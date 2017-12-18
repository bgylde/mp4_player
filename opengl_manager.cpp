#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <string.h>
#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <pthread.h>
#include <binder/IPCThreadState.h>
#include <binder/ProcessState.h>
#include <binder/IServiceManager.h>

#include "opengl_manager.h"
#include "android_surface.h"

using namespace android;

#define MAX_MANAGER_MESSAGE_NUM 64

typedef struct
{
   GLfloat   m[4][4];
} ESMatrix;


typedef enum _MANAGER_MSG
{
    MANAGER_MSG_NONE = 0,
    MANAGER_MSG_INIT = 1,
    MANAGER_MSG_PREPARE = 2,
    MANAGER_MSG_VARY = 3,
    MANAGER_MSG_PUSH_DATA = 4,
    MANAGER_MSG_CLEAR = 5,
    MANAGER_MSG_UNINT = 6,

} MANAGER_MSG;

typedef struct _VIDEO_RECT
{
    int32_t height;
    int32_t width;
    int32_t x;
    int32_t y;
}VIDEO_RECT_T,*VIDEO_RECT_P;

typedef struct _Manager_Message_Context
{
    pthread_t message_thread;
    int32_t  task_running;

    MANAGER_MSG  event_type[MAX_MANAGER_MESSAGE_NUM];
    void *event_param[MAX_MANAGER_MESSAGE_NUM];
    unsigned long long  p_event_num;//producer
    unsigned long long  c_event_num;//consumer
    int32_t clearflag;
}Manager_Message_Context;

typedef struct _OpenGL_Manager_Context
{
    EGLDisplay        display; 
    EGLContext        context;
    EGLint            program;                     
    EGLSurface        surface;
    EGLint            width;
    EGLint            height;
    EGLint            major;
    EGLint            minor;
    EGLint            numConfigs;
    EGLConfig         config[256];

    GLuint            _vertexBuffer;
    GLuint            _texCoordBuffer;
    GLuint            _indexBuffer;

    ESMatrix          mvpMatrix;                   //变换矩阵 4X4
    GLfloat           angle;                       //绕向量旋转角度
    GLfloat           lrangle;                     //绕左右旋转角度
    GLfloat           udangle;                     //绕上下旋转角度
    GLfloat           zdistance;
    
    int32_t           vr;
    int32_t           tex_created;
    u_int32_t*        textureid;                   //OpenGL ES 纹理ID，目前为3个
    int32_t           vertexnum;                   //顶点数量，用于glDrawElements

    Manager_Message_Context *  gl_message_context;
    
    int32_t           video_width;
    int32_t           video_height;
    int32_t           x;
    int32_t           y;
}OpenGL_Manager_Context;

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define PI 3.1415926535897932384626433832795f


const char vShaderStr[] =
        "#version 300 es                            \n"
        "uniform mat4 u_mvpMatrix;                  \n"
        "layout(location = 0) in vec4 a_position;   \n"
        "layout(location = 1) in vec2 a_texCoord;   \n"
        "out vec2 v_texCoord;                       \n"
        "void main()                                \n"
        "{                                          \n"
        "   gl_Position = u_mvpMatrix * a_position; \n"
        "   v_texCoord = a_texCoord;                \n"
        "}                                          \n";                        

const char fShaderStr[] =
        "#version 300 es                                     \n"
        "precision mediump float;                            \n"
        "in vec2 v_texCoord;                                 \n"
        "layout(location = 0) out vec4 outColor;             \n"
        "uniform sampler2D SamplerY;                         \n"  
        "uniform sampler2D SamplerU;                         \n"  
        "uniform sampler2D SamplerV;                         \n"  
        "void main()                                         \n"
        "{                                                   \n"
        "    mediump vec3 yuv;                               \n"  
        "    lowp vec3 rgb;                                  \n"  
        "    yuv.x = texture(SamplerY, v_texCoord).r;        \n"  
        "    yuv.y = texture(SamplerU, v_texCoord).r - 0.5;  \n"  
        "    yuv.z = texture(SamplerV, v_texCoord).r - 0.5;  \n"  
        "    rgb = mat3( 1,   1,   1,                        \n"  
        "                0, -0.39465, 2.03211,               \n"  
        "                1.13983, -0.58060, 0) * yuv;        \n"  
        "    outColor = vec4(rgb, 1);                        \n"  
        "}                                                   \n";


//static OpenGL_Manager_Context *context = NULL;

#if 0
static int window_width = 1920;
static int window_height = 1080;
#else
static int window_width = 1480;
static int window_height = 920;
#endif

/*************************************************
 Function: opengl_manager_LoadShader
 Description: 加载并编译着色器
 Input: type: 枚举着色器类型，顶点或者片元. shaderSrc: 着色器代码
 Output: 无
 Return: int,success为shader ID，fail为0
 Others: 无
 *************************************************/

static GLuint opengl_manager_LoadShader(GLenum type, const char *shaderSrc)
{
    GLuint shader;
    GLint compiled;
    shader = glCreateShader ( type );// Create the shader object
    if ( shader == 0 ) {
        printf("Create Sharder error %d",glGetError());
        return 0;
    }
    glShaderSource ( shader, 1, &shaderSrc, NULL );// Load the shader source
    glCompileShader ( shader );// Compile the shader
    glGetShaderiv ( shader, GL_COMPILE_STATUS, &compiled );// Check the compile status
    if ( !compiled ) {
        GLint infoLen = 0;
        glGetShaderiv ( shader, GL_INFO_LOG_LENGTH, &infoLen );
        if ( infoLen > 1 ) {
            char *infoLog = (char*)malloc ( sizeof ( char ) * infoLen );
            glGetShaderInfoLog ( shader, infoLen, NULL, infoLog );
            printf( "Error compiling shader:\n%s", infoLog );
            free ( infoLog );
        }
        glDeleteShader ( shader );
        return 0;
    }
    return shader;
}
/*************************************************
Function: opengl_manager_LoadProgram
Description: 加载program，加载并编译着色器
Input: vertShaderSrc:顶点着色器代码 . fragShaderSrc: 片元着色器代码
Output: 无
Return: int,success为program ID，fail为0
Others: 无
*************************************************/

static GLuint opengl_manager_LoadProgram(const char *vertShaderSrc, const char *fragShaderSrc)
{
    GLuint vertexShader;
    GLuint fragmentShader;
    GLuint programObject;
    GLint linked;
    vertexShader = opengl_manager_LoadShader ( GL_VERTEX_SHADER, vertShaderSrc ); // Load the vertex/fragment shaders
    if ( vertexShader == 0 ) {
        printf("Load Shader error");
        return 0;
    }
    fragmentShader = opengl_manager_LoadShader ( GL_FRAGMENT_SHADER, fragShaderSrc );
    if ( fragmentShader == 0 ) {
        printf("Load Shader error\n");
        glDeleteShader ( vertexShader );
        return 0;
    }
    programObject = glCreateProgram ( );// Create the program object
    if ( programObject == 0 ) {
        printf("Create Program error");
        return 0;
    }
    glAttachShader ( programObject, vertexShader );
    glAttachShader ( programObject, fragmentShader );
    glLinkProgram ( programObject );// Link the program
    glGetProgramiv ( programObject, GL_LINK_STATUS, &linked );// Check the link status
    if ( !linked ) {
        GLint infoLen = 0;
        glGetProgramiv ( programObject, GL_INFO_LOG_LENGTH, &infoLen );
        if ( infoLen > 1 ) {
            char *infoLog = (char*)malloc( sizeof ( char ) * infoLen );
            glGetProgramInfoLog ( programObject, infoLen, NULL, infoLog );
            printf( "Error linking program:\n%s", infoLog );
            free ( infoLog );
        }
        glDeleteProgram ( programObject );
        return 0;
    }
    glDeleteShader ( vertexShader );// Free up no longer needed shader resources
    glDeleteShader ( fragmentShader );
    return programObject;
}

/*************************************************
Function: opengl_manager_MatrixMultiply
Description: 矩阵相乘
Input: srcA: A矩阵，srcB: B矩阵
Output: result: 结果矩阵
Return: 无
Others: 没判空，要求输入都不为空
*************************************************/

void 
opengl_manager_MatrixMultiply ( ESMatrix *result, ESMatrix *srcA, ESMatrix *srcB )
{
   ESMatrix    tmp;
   int         i;

   for ( i = 0; i < 4; i++ )
   {
      tmp.m[i][0] =  ( srcA->m[i][0] * srcB->m[0][0] ) +
                     ( srcA->m[i][1] * srcB->m[1][0] ) +
                     ( srcA->m[i][2] * srcB->m[2][0] ) +
                     ( srcA->m[i][3] * srcB->m[3][0] ) ;

      tmp.m[i][1] =  ( srcA->m[i][0] * srcB->m[0][1] ) +
                     ( srcA->m[i][1] * srcB->m[1][1] ) +
                     ( srcA->m[i][2] * srcB->m[2][1] ) +
                     ( srcA->m[i][3] * srcB->m[3][1] ) ;

      tmp.m[i][2] =  ( srcA->m[i][0] * srcB->m[0][2] ) +
                     ( srcA->m[i][1] * srcB->m[1][2] ) +
                     ( srcA->m[i][2] * srcB->m[2][2] ) +
                     ( srcA->m[i][3] * srcB->m[3][2] ) ;

      tmp.m[i][3] =  ( srcA->m[i][0] * srcB->m[0][3] ) +
                     ( srcA->m[i][1] * srcB->m[1][3] ) +
                     ( srcA->m[i][2] * srcB->m[2][3] ) +
                     ( srcA->m[i][3] * srcB->m[3][3] ) ;
   }

   memcpy ( result, &tmp, sizeof ( ESMatrix ) );
}

/*************************************************
Function: opengl_manager_Frustum
Description: 函数定义一个平截头体，它计算一个用于实现透视投影的矩阵，并把它与当前的投影矩阵（一般是单位矩阵）相乘。
也即是该函数构造了一个视景体用来将模型进行投影，来裁剪模型，决定模型哪些在视景体里面，哪些在视景体的外面，在视景体之外的就不可见。
Input: left: 左边所在X轴上坐标. right: 右边所在X轴坐标. bottom: 下边所在Y轴坐标. top: 上边所在Y轴坐标. 
       nearZ: 近处剪切面距离眼坐标(原点)位置. farZ: 远处剪切面距离眼坐标(原点)位置. 
Output: result: 结果矩阵
Return: 无
Others: 没判空，要求输入都不为空
*************************************************/
void 
opengl_manager_Frustum ( ESMatrix *result, float left, float right, float bottom, float top, float nearZ, float farZ )
{
   float       deltaX = right - left;
   float       deltaY = top - bottom;
   float       deltaZ = farZ - nearZ;
   ESMatrix    frust;

   if ( ( nearZ <= 0.0f ) || ( farZ <= 0.0f ) ||
         ( deltaX <= 0.0f ) || ( deltaY <= 0.0f ) || ( deltaZ <= 0.0f ) )
   {
      return;
   }

   frust.m[0][0] = 2.0f * nearZ / deltaX;
   frust.m[0][1] = frust.m[0][2] = frust.m[0][3] = 0.0f;

   frust.m[1][1] = 2.0f * nearZ / deltaY;
   frust.m[1][0] = frust.m[1][2] = frust.m[1][3] = 0.0f;

   frust.m[2][0] = ( right + left ) / deltaX;
   frust.m[2][1] = ( top + bottom ) / deltaY;
   frust.m[2][2] = - ( nearZ + farZ ) / deltaZ;
   frust.m[2][3] = -1.0f;

   frust.m[3][2] = -2.0f * nearZ * farZ / deltaZ;
   frust.m[3][0] = frust.m[3][1] = frust.m[3][3] = 0.0f;

   opengl_manager_MatrixMultiply ( result, &frust, result );
}


/*************************************************
Function: opengl_manager_MatrixMultiply
Description: 和opengl_manager_Frustum相似，只是参数不同
Input: fovy: 视场角(FOV). aspect: 宽高比.  
       nearZ: 近处剪切面距离眼坐标(原点)位置. farZ: 远处剪切面距离眼坐标(原点)位置. 
Output: result: 结果矩阵
Return: 无
Others: 没判空，要求输入都不为空
*************************************************/
void 
opengl_manager_Perspective ( ESMatrix *result, float fovy, float aspect, float nearZ, float farZ )
{
   GLfloat frustumW, frustumH;

   frustumH = tanf ( fovy / 360.0f * PI ) * nearZ;
   frustumW = frustumH * aspect;

   opengl_manager_Frustum ( result, -frustumW, frustumW, -frustumH, frustumH, nearZ, farZ );
}

/*************************************************
Function: opengl_manager_MatrixLoadIdentity
Description: 生成一个单位矩阵
Input: 无
Output: result: 结果矩阵
Return: 无
Others: 没判空，要求输入都不为空
*************************************************/
void 
opengl_manager_MatrixLoadIdentity ( ESMatrix *result )
{
   memset ( result, 0x0, sizeof ( ESMatrix ) );
   result->m[0][0] = 1.0f;
   result->m[1][1] = 1.0f;
   result->m[2][2] = 1.0f;
   result->m[3][3] = 1.0f;
}

/*************************************************
Function: opengl_manager_Translate
Description: 将眼坐标移动到(tx,ty,tz)
Input: tx: X轴坐标. ty: Y轴坐标. tz: Z轴坐标.
Output: result: 结果矩阵
Return: 无
Others: 没判空，要求输入都不为空
*************************************************/
void 
opengl_manager_Translate ( ESMatrix *result, GLfloat tx, GLfloat ty, GLfloat tz )
{
   result->m[3][0] += ( result->m[0][0] * tx + result->m[1][0] * ty + result->m[2][0] * tz );
   result->m[3][1] += ( result->m[0][1] * tx + result->m[1][1] * ty + result->m[2][1] * tz );
   result->m[3][2] += ( result->m[0][2] * tx + result->m[1][2] * ty + result->m[2][2] * tz );
   result->m[3][3] += ( result->m[0][3] * tx + result->m[1][3] * ty + result->m[2][3] * tz );
}

/*************************************************
Function: opengl_manager_Translate
Description: 将顶点绕着向量(tx,ty,tz)旋转angle度
Input: angle:角度. x: X轴坐标. y: Y轴坐标. z: Z轴坐标.
Output: result: 结果矩阵
Return: 无
Others: 没判空，要求输入都不为空
*************************************************/
void 
opengl_manager_Rotate ( ESMatrix *result, GLfloat angle, GLfloat x, GLfloat y, GLfloat z )
{
   GLfloat sinAngle, cosAngle;
   GLfloat mag = sqrtf ( x * x + y * y + z * z );//求模长

   sinAngle = sinf ( angle * PI / 180.0f );
   cosAngle = cosf ( angle * PI / 180.0f );

   if ( mag > 0.0f )
   {
      GLfloat xx, yy, zz, xy, yz, zx, xs, ys, zs;
      GLfloat oneMinusCos;
      ESMatrix rotMat;

      x /= mag;
      y /= mag;
      z /= mag;

      xx = x * x;
      yy = y * y;
      zz = z * z;
      xy = x * y;
      yz = y * z;
      zx = z * x;
      xs = x * sinAngle;
      ys = y * sinAngle;
      zs = z * sinAngle;
      oneMinusCos = 1.0f - cosAngle;

      rotMat.m[0][0] = ( oneMinusCos * xx ) + cosAngle;
      rotMat.m[0][1] = ( oneMinusCos * xy ) - zs;
      rotMat.m[0][2] = ( oneMinusCos * zx ) + ys;
      rotMat.m[0][3] = 0.0F;

      rotMat.m[1][0] = ( oneMinusCos * xy ) + zs;
      rotMat.m[1][1] = ( oneMinusCos * yy ) + cosAngle;
      rotMat.m[1][2] = ( oneMinusCos * yz ) - xs;
      rotMat.m[1][3] = 0.0F;

      rotMat.m[2][0] = ( oneMinusCos * zx ) - ys;
      rotMat.m[2][1] = ( oneMinusCos * yz ) + xs;
      rotMat.m[2][2] = ( oneMinusCos * zz ) + cosAngle;
      rotMat.m[2][3] = 0.0F;

      rotMat.m[3][0] = 0.0F;
      rotMat.m[3][1] = 0.0F;
      rotMat.m[3][2] = 0.0F;
      rotMat.m[3][3] = 1.0F;

      opengl_manager_MatrixMultiply ( result, &rotMat, result );
   }
}

 /*************************************************
 Function: opengl_manager_vary_t
 Description: 将球体投影并将其绕着X轴或者Y轴旋转value度
 Input: manager_instance:当前manager_instance. value: 旋转角度. type: 1为绕着左右，2为绕着上下.
 Output: 无
 Return: 无
 Others: 没判空，要求输入都不为空
 *************************************************/

 void opengl_manager_vary_t(OpenGL_Manager_Context *context, GLfloat value, int32_t type)
 {
    if(context == NULL)
    {
        return;
    }
    OpenGL_Manager_Context * manager_instance = context;
    ESMatrix perspective;
    ESMatrix modelview;
    float    aspect;

    printf("------in---------value=%f,type=%d",value,type);
    if(type == 1)
    {
       // Compute a rotation angle based on time to rotate the cube
       manager_instance->lrangle += value;

       if ( manager_instance->lrangle >= 360.0f )
       {
          manager_instance->lrangle -= 360.0f;
       }
    }
    else if(type == 2)
    {
        manager_instance->udangle += value;

       if ( manager_instance->udangle >= 360.0f )
       {
          manager_instance->udangle -= 360.0f;
       }
    }
    
    printf("lrangle = %f, udangle=%f" ,manager_instance->lrangle ,manager_instance->udangle );
    
    // Compute the window aspect ratio
    aspect = ( GLfloat ) manager_instance->video_width / ( GLfloat ) manager_instance->video_height;

    // Generate a perspective matrix with a 60 degree FOV
    opengl_manager_MatrixLoadIdentity ( &perspective );
    opengl_manager_Perspective ( &perspective, 85.0f, aspect, 0.1f, 400.0f );

    // Generate a model view matrix to rotate/translate the cube
    opengl_manager_MatrixLoadIdentity ( &modelview );
    

    // Translate away from the viewer
    opengl_manager_Translate ( &modelview, 0.0, 0.0, manager_instance->zdistance);
    // Rotate the cube

    opengl_manager_Rotate ( &modelview, manager_instance->lrangle, modelview.m[0][1], modelview.m[1][1], modelview.m[2][1]);

    opengl_manager_Rotate ( &modelview, manager_instance->udangle, modelview.m[0][0], modelview.m[1][0], modelview.m[2][0]);
 
      


    // Compute the final MVP by multiplying the
    // modevleiw and perspective matrices together
    opengl_manager_MatrixMultiply ( &manager_instance->mvpMatrix, &modelview, &perspective );

   
 }
 /*************************************************
 Function: opengl_manager_vary
 Description: 将球体投影并将其绕着向量(x,y,z)旋转value度
 Input: manager_instance:当前manager_instance. value: 旋转角度.  x: X轴坐标. y: Y轴坐标. z: Z轴坐标.
 Output: 无
 Return: 无
 Others: 没判空，要求输入都不为空
 *************************************************/

 void opengl_manager_vary(OpenGL_Manager_Context *context, GLfloat value, GLfloat x, GLfloat y, GLfloat z )
{
     if(context == NULL)
    {
        return;
    }
    OpenGL_Manager_Context * manager_instance = context;
   ESMatrix perspective;
   ESMatrix modelview;
   float    aspect;

   // Compute a rotation angle based on time to rotate the cube
   manager_instance->angle += value;

   if ( manager_instance->angle >= 360.0f )
   {
      manager_instance->angle -= 360.0f;
   }

    // Compute the window aspect ratio
    aspect = ( GLfloat ) 1960 / ( GLfloat ) 1080;

    // Generate a perspective matrix with a 60 degree FOV
    opengl_manager_MatrixLoadIdentity ( &perspective );
    opengl_manager_Perspective ( &perspective, 85.0f, aspect, 0.1f, 400.0f );

    // Generate a model view matrix to rotate/translate the cube
    opengl_manager_MatrixLoadIdentity ( &modelview );

    // Translate away from the viewer
    opengl_manager_Translate ( &modelview, 0.0, 0.0, manager_instance->zdistance);
    // Rotate the cube
    opengl_manager_Rotate ( &modelview, manager_instance->angle, x, y, z);


    // Compute the final MVP by multiplying the
    // modevleiw and perspective matrices together
    opengl_manager_MatrixMultiply ( &manager_instance->mvpMatrix, &modelview, &perspective );

    /*
    int i;
   for(i = 0 ; i < 4 ; i++)
   {
       printf("update end %f %f %f %f\n",manager_instance->mvpMatrix.m[i][0],manager_instance->mvpMatrix.m[i][1],manager_instance->mvpMatrix.m[i][2],manager_instance->mvpMatrix.m[i][3]);
   }
   */

}
     /*************************************************
      Function: opengl_manager_vary
      Description: 将球体投影并将其绕着向量(x,y,z)旋转value度
      Input: manager_instance:当前manager_instance. value: 旋转角度.  x: X轴坐标. y: Y轴坐标. z: Z轴坐标.
      Output: 无
      Return: 无
      Others: 没判空，要求输入都不为空
      *************************************************/
     
  void opengl_manager_vary_c(OpenGL_Manager_Context *context, GLfloat value )
 {
     if(context == NULL)
    {
        return;
    }
    OpenGL_Manager_Context * manager_instance = context;
    ESMatrix perspective;
    ESMatrix modelview;
    float    aspect;

    manager_instance->zdistance += value;
    if(manager_instance->zdistance <= -100.0)
    {
        manager_instance->zdistance = -100.0;
    }
    printf("manager_instance->zdistance = %f",manager_instance->zdistance);
    
    // Compute the window aspect ratio
    
    aspect = ( GLfloat ) manager_instance->video_width / ( GLfloat ) manager_instance->video_height;
    

    // Generate a perspective matrix with a 60 degree FOV
    opengl_manager_MatrixLoadIdentity ( &perspective );
    opengl_manager_Perspective ( &perspective, 85.0f, aspect, 0.1f, 400.0f );

    // Generate a model view matrix to rotate/translate the cube
    opengl_manager_MatrixLoadIdentity ( &modelview );
    

    // Translate away from the viewer
    opengl_manager_Translate ( &modelview, 0.0, 0.0, manager_instance->zdistance);
    // Rotate the cube

    opengl_manager_Rotate ( &modelview, manager_instance->lrangle, modelview.m[0][1], modelview.m[1][1], modelview.m[2][1]);

    opengl_manager_Rotate ( &modelview, manager_instance->udangle, modelview.m[0][0], modelview.m[1][0], modelview.m[2][0]);


    // Compute the final MVP by multiplying the
    // modevleiw and perspective matrices together
    opengl_manager_MatrixMultiply ( &manager_instance->mvpMatrix, &modelview, &perspective );
 
 }

 

/*************************************************
Function: opengl_manager_createTexture
Description: 创建Y U V纹理ID并且设置纹理属性。
Input: 无
Output: 无
Return: 纹理ID指针
Others: 无
*************************************************/

u_int32_t* opengl_manager_createTexture()  
{  
    GLuint* textureid =(GLuint*) malloc(3*sizeof(GLuint));  
    
    glGenTextures ( 3, textureid ); 
    //绑定Y纹理
    glActiveTexture ( GL_TEXTURE0 );
    glBindTexture ( GL_TEXTURE_2D, textureid[0] );   
  
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );  
      
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );  
     
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );  
      
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );   

    //绑定U纹理
    glActiveTexture ( GL_TEXTURE1 );
    glBindTexture(GL_TEXTURE_2D, textureid[1] );
    
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );  
      
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );  
     
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );  
      
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

    //绑定V纹理
    glActiveTexture ( GL_TEXTURE2 );
    glBindTexture(GL_TEXTURE_2D, textureid[2] );
     
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );  
      
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );  
     
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );  
      
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

    return (u_int32_t*)textureid;  
}  

/*************************************************
Function: opengl_manager_updateTexture
Description: 根据buffer创建或者更新YUV glTexImage2D
Input: textureid: 纹理ID指针. buffer:视频buffer . width:视频宽 . height:视频高 .
Output: 无
Return: 无
Others: 无
*************************************************/

void opengl_manager_updateTexture(OpenGL_Manager_Context *context, u_int32_t* textureid,u_int8_t  *buffer, GLuint width , GLuint height)
{
    if(context->tex_created == 0)
    {
        //printf("update TexImage2D ");
        glActiveTexture ( GL_TEXTURE0 );
        glBindTexture ( GL_TEXTURE_2D, textureid[0] );  
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer);
        
        glActiveTexture ( GL_TEXTURE1 );
        glBindTexture(GL_TEXTURE_2D, textureid[1] );
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width/2, height/2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer + width * height);
        
        glActiveTexture ( GL_TEXTURE2 );
        glBindTexture(GL_TEXTURE_2D, textureid[2] );  
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width/2, height/2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer + width * height * 5 / 4);
        context->tex_created = 1;
    }
    else
    {
        //printf("update SubImage2D ");
        glActiveTexture ( GL_TEXTURE0 );
        glBindTexture ( GL_TEXTURE_2D, textureid[0] );  
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer);
        
        glActiveTexture ( GL_TEXTURE1 );
        glBindTexture ( GL_TEXTURE_2D, textureid[1] );  
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width/2, height/2, GL_LUMINANCE, GL_UNSIGNED_BYTE,  buffer + width * height);
        
        glActiveTexture ( GL_TEXTURE2 );
        glBindTexture ( GL_TEXTURE_2D, textureid[2] );  
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width/2, height/2, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer + width * height * 5 / 4);
    }
}

void opengl_manager_prepare(OpenGL_Manager_Context *context, int32_t width, int32_t height, int32_t x, int32_t y)
{
    printf("--in---- ,%d ,%d\n",width,height);
    if(context == NULL)
    {
        return;
    }
    
    OpenGL_Manager_Context * manager_instance = context;
    manager_instance->video_width = width;
    manager_instance->video_height= height;
    manager_instance->x = x;
    manager_instance->y = y;
    if(manager_instance->vr == FALSE)
    {
        printf("in not vr\n");

        GLfloat dst_x1 =   (GLfloat)(width + x)/(GLfloat)manager_instance->width ;//(2* 1536/1920 -1 = 0.6 1)
        GLfloat dst_y1 =   (GLfloat)(height - y)/(GLfloat)manager_instance->height;//(2*648/1080 - 1 = 0.2 1)
        GLfloat dst_x2 = - (GLfloat)(width - x)/(GLfloat)manager_instance->width;//(2 * 192/1920 -1 = -0.8  -1)
        GLfloat dst_y2 = - (GLfloat)(height + y)/(GLfloat)manager_instance->height;//2* 82/1080 -1 = -0.85 -1)
        printf("dst_x1: %f, dst_y1: %f, dst_x2: %f, dst_y2: %f\n", (float)dst_x1, (float)dst_y1, (float)dst_x2, (float)dst_y2);
        GLfloat verts[] =
        {
        #if 0
            dst_x2,dst_y2,0.0,
            dst_x2,dst_y1,0.0,
            dst_x1,dst_y1,0.0,
            dst_x1,dst_y2,0.0
        #else
            -1.0, -1.0, 0.3,
            -1.0,  1.0, 0.3,
             1.0,  1.0, 0.3,
             1.0, -1.0, 0.3
        #endif

        };
        GLfloat texcoord[] =
        {
            0.0,1.0, 
            0.0,0.0,
            1.0,0.0,
            1.0,1.0
        };

        GLushort indices[] =
        {
            0,3,1,2
            //0, 1, 2, 3
        };
        manager_instance->vertexnum = 4;
        
       // 加载顶点坐标数据
        glGenBuffers(1, &manager_instance->_vertexBuffer); // 申请内存
        glBindBuffer(GL_ARRAY_BUFFER, manager_instance->_vertexBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) *12 ,verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT,GL_FALSE, 0, 0);

        glGenBuffers(1, &manager_instance->_texCoordBuffer); // 申请内存
        glBindBuffer(GL_ARRAY_BUFFER, manager_instance->_texCoordBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8,texcoord, GL_STATIC_DRAW);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT,GL_FALSE, 0, 0);

        glGenBuffers(1, &manager_instance->_indexBuffer); // 申请内存
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, manager_instance->_indexBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort)* 4, indices, GL_STATIC_DRAW);
    }
    else
    {
        printf("in  vr");
        GLfloat *positions;
        GLfloat *uvs;
        GLushort *indices;

        GLint slices = 100;
        GLint stacks = 100;
        float radius = 300.0;

        GLint vertex_count = (slices + 1) * stacks;

        const int component_size = sizeof (GLfloat) * vertex_count;

        positions = (GLfloat *) malloc (component_size * 3);
        uvs = (GLfloat *) malloc (component_size * 2);

        GLfloat *v = positions;
        GLfloat *t = uvs;

        float const J = 1. / (float) (stacks - 1);
        float const I = 1. / (float) (slices - 1);
        int i,j;
        for (i = 0; i < slices; i++) {
            float const theta = PI * i * I;
            for (j = 0; j < stacks; j++) {
              float const phi = 2 * PI * j * J + PI / 2.0;

              float const x = sin (theta) * cos (phi);
              float const y = -cos (theta);
              float const z = sin (phi) * sin (theta);

              *v++ = x * radius;
              *v++ = y * radius;
              *v++ = z * radius;

              *t++ = j * J;
              *t++ = 1.0 - i * I;
             // printf("theta=%f,phi=%f,(%f,%f,%f)\n",theta,phi,x * radius,y * radius,z * radius);
            }
        }

        /* index */
        GLint index_size = (slices - 1) * stacks * 2;

        indices = (GLushort *) malloc (sizeof (GLushort) * index_size);
        GLushort *indextemp = indices;

        // -3 = minus caps slices - one to iterate over strips
        for (i = 0; i < slices - 1; i++) {
            for (j = 0; j < stacks; j++) {
              *indextemp++ = i * stacks + j;
              *indextemp++ = (i + 1) * stacks + j;
            }
        }
    


        // 加载顶点坐标数据
        glGenBuffers(1, &manager_instance->_vertexBuffer); // 申请内存
        glBindBuffer(GL_ARRAY_BUFFER, manager_instance->_vertexBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertex_count * 3,positions, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT,GL_FALSE, 0, 0);
        


        glGenBuffers(1, &manager_instance->_texCoordBuffer); // 申请内存
        glBindBuffer(GL_ARRAY_BUFFER, manager_instance->_texCoordBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertex_count * 2,uvs, GL_STATIC_DRAW);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT,GL_FALSE, 0, 0);
        

        
        glGenBuffers(1, &manager_instance->_indexBuffer); // 申请内存
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, manager_instance->_indexBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort)*index_size,indices, GL_STATIC_DRAW);
    


        free(positions);
        free(uvs);
        free(indices);
        
        manager_instance->vertexnum = index_size;

                 
        glEnable(GL_CULL_FACE); 
        glCullFace(GL_FRONT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_SRC_COLOR);

        
    }
    // Set the viewport
    glViewport ((window_width - width)/2, (window_height - height)/2, width, height);
    
    //glViewport (0, 0, width, height);
    glClearColor(1.0f,0.0f,0.0f,0.3f);
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glEnable(GL_TEXTURE_2D);

    // Use the program object
    glUseProgram ( manager_instance->program );
  
    printf("2. testtesttesttest!\n");
    glUniform1i(glGetUniformLocation (manager_instance->program, "SamplerY"), 0);
    glUniform1i(glGetUniformLocation (manager_instance->program, "SamplerU"), 1); 
    glUniform1i(glGetUniformLocation (manager_instance->program, "SamplerV"), 2); 

    printf("3. testtesttesttest!\n");
    manager_instance->textureid = opengl_manager_createTexture();
    printf("4. testtesttesttest!\n");
    // Load the MVP matrix
    glUniformMatrix4fv (glGetUniformLocation (manager_instance->program, "u_mvpMatrix"), 1, GL_FALSE, ( GLfloat * ) &manager_instance->mvpMatrix.m[0][0] );
    printf("----out----");
}

void opengl_manager_change(OpenGL_Manager_Context *context, int32_t vr)
{
    printf("-----------in---------");
    if(context == NULL)
    {
        return;
    }
    OpenGL_Manager_Context * manager_instance = context;
    manager_instance->vr = vr;
    // Delete texture object
    glDeleteBuffers(1, &manager_instance->_vertexBuffer);    
    glDeleteBuffers(1, &manager_instance->_texCoordBuffer);
    glDeleteBuffers(1, &manager_instance->_indexBuffer);

    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );


    if(manager_instance->vr == FALSE)
     {
        printf("in init");
         manager_instance->angle = 0.0f;
         manager_instance->lrangle = 0.0f;
         manager_instance->udangle = 0.0f;
         manager_instance->zdistance = -100.0f;
         opengl_manager_MatrixLoadIdentity(&manager_instance->mvpMatrix);
     }
     else
     {
         manager_instance->angle = 0.0f;
         manager_instance->lrangle = 0.0f;
         manager_instance->udangle = 0.0f;
         manager_instance->zdistance = -100.0f;
         opengl_manager_vary(manager_instance, 0.0,0.0,1.0,0.0);
        // opengl_manager_MatrixLoadIdentity(&manager_instance->mvpMatrix);
     }

    if(manager_instance->vr == FALSE)
    {
        printf("in not vr");
        GLfloat verts[] =
        {
            -1.0,-1.0,0.0,
            -1.0,1.0,0.0,
            1.0,1.0,0.0,
            1.0,-1.0,0.0
        };
        GLfloat texcoord[] =
        {
            0.0,1.0, 
            0.0,0.0,
            1.0,0.0,
            1.0,1.0           
        };

        GLushort indices[] =
        {
            0,3,1,2
        };
        manager_instance->vertexnum = 4;
         // 加载顶点坐标数据
        glGenBuffers(1, &manager_instance->_vertexBuffer); // 申请内存
        glBindBuffer(GL_ARRAY_BUFFER, manager_instance->_vertexBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) *12 ,verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT,GL_FALSE, 0, 0);

        glGenBuffers(1, &manager_instance->_texCoordBuffer); // 申请内存
        glBindBuffer(GL_ARRAY_BUFFER, manager_instance->_texCoordBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8,texcoord, GL_STATIC_DRAW);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT,GL_FALSE, 0, 0);

        glGenBuffers(1, &manager_instance->_indexBuffer); // 申请内存
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, manager_instance->_indexBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort)* 4, indices, GL_STATIC_DRAW);
    
    }
    else
    {
        printf("in  vr");
        GLfloat *positions;
        GLfloat *uvs;
        GLushort *indices;

        GLint slices = 100;
        GLint stacks = 100;
        float radius = 300.0;

        GLint vertex_count = (slices + 1) * stacks;

        const int component_size = sizeof (GLfloat) * vertex_count;

        positions = (GLfloat *) malloc (component_size * 3);
        uvs = (GLfloat *) malloc (component_size * 2);

        GLfloat *v = positions;
        GLfloat *t = uvs;

        float const J = 1. / (float) (stacks - 1);
        float const I = 1. / (float) (slices - 1);
        int i,j;
        for (i = 0; i < slices; i++) {
            float const theta = PI * i * I;
            for (j = 0; j < stacks; j++) {
              float const phi = 2 * PI * j * J + PI / 2.0;

              float const x = sin (theta) * cos (phi);
              float const y = -cos (theta);
              float const z = sin (phi) * sin (theta);

              *v++ = x * radius;
              *v++ = y * radius;
              *v++ = z * radius;

              *t++ = j * J;
              *t++ = 1.0 - i * I;
             // printf("theta=%f,phi=%f,(%f,%f,%f)\n",theta,phi,x * radius,y * radius,z * radius);
            }
        }

        /* index */
        GLint index_size = (slices - 1) * stacks * 2;

        indices = (GLushort *) malloc (sizeof (GLushort) * index_size);
        GLushort *indextemp = indices;

        // -3 = minus caps slices - one to iterate over strips
        for (i = 0; i < slices - 1; i++) {
            for (j = 0; j < stacks; j++) {
              *indextemp++ = i * stacks + j;
              *indextemp++ = (i + 1) * stacks + j;
            }
        }

        // 加载顶点坐标数据
        glGenBuffers(1, &manager_instance->_vertexBuffer); // 申请内存
        glBindBuffer(GL_ARRAY_BUFFER, manager_instance->_vertexBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertex_count * 3,positions, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT,GL_FALSE, 0, 0);

        glGenBuffers(1, &manager_instance->_texCoordBuffer); // 申请内存
        glBindBuffer(GL_ARRAY_BUFFER, manager_instance->_texCoordBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertex_count * 2,uvs, GL_STATIC_DRAW);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT,GL_FALSE, 0, 0);
             
        glGenBuffers(1, &manager_instance->_indexBuffer); // 申请内存
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, manager_instance->_indexBuffer); // 将命名的缓冲对象绑定到指定的类型上去
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort)*index_size,indices, GL_STATIC_DRAW);

        free(positions);
        free(uvs);
        free(indices);
        
        manager_instance->vertexnum = index_size;
        printf("----out----");
    }
}


int32_t opengl_manager_init(OpenGL_Manager_Context *context)
{
    printf("--------in---------\n");
    
    int32_t ret = TRUE;
    if(context == NULL)
    {
        return FALSE;
    }
    OpenGL_Manager_Context * instance = context;
    //配置属性
     EGLint configAttribs[] = {
        EGL_SAMPLES,        0,
        EGL_SAMPLE_BUFFERS, 0,
        EGL_RED_SIZE,       8,
        EGL_GREEN_SIZE,     8,
        EGL_BLUE_SIZE,      8,
        EGL_ALPHA_SIZE,     8,
        EGL_LUMINANCE_SIZE, 0,
        EGL_BUFFER_SIZE,    32,
        EGL_DEPTH_SIZE,     0,
        EGL_STENCIL_SIZE,   0,
        EGL_SURFACE_TYPE,   EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE,EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };  
    
    EGLint contextAttribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    
    printf("-------getdisplay begin----------\n");
    //获取EGLdisplay 
    instance->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if(EGL_NO_DISPLAY == instance->display)
    {
        printf("-------getdisplay fail----------\n");
        ret = FALSE;
    }
    else
    {
        printf("-------getdisplay successful----------\n");
    }
    
    //初始化，获取版本号
    if(!eglInitialize(instance->display, &instance->major, &instance->minor))
    {
        printf("-------eglInitialize fail----------\n");
        ret = FALSE;
    }
    else
    {
        printf("-------eglInitialize successful ----------\n");
    }
    
    //通过属性获取EGLconfig
    if(!eglChooseConfig(instance->display, configAttribs, NULL, 0, &instance->numConfigs))
    {
        printf("eglChooseConfig error 0x%x\n", eglGetError());
        ret = FALSE; 
    }
    else
    {
        printf("eglChooseConfig successful.\n");
    }
    
    if(!eglChooseConfig(instance->display, configAttribs, instance->config, instance->numConfigs, &instance->numConfigs))
    {
        printf("-------getconfig fail----------\n");
        ret = FALSE;
    }
    else
    {
        printf("eglChooseConfig successful.\n");
    }
    
    //绑定opengles API
     if(!eglBindAPI (EGL_OPENGL_ES_API))
     {
        printf("-------bindapi fail----------\n");
        ret = FALSE;
     } else
     {
         printf("-------eglBindAPI successful ----------\n");
     }

    //创建context
    instance->context = eglCreateContext (instance->display, instance->config[0], EGL_NO_CONTEXT, contextAttribs);
    if (EGL_NO_CONTEXT == instance->context)
    {
        printf("-------createcontext fail----------\n");
        ret = FALSE;
    }
    else
    {
        printf("-------createcontext successful----------\n");

    }
    //创建EGLNativeWindowType
    //TODO
    EGLNativeWindowType WindowTypes = (EGLNativeWindowType) getNativeWindow(window_width, window_height, 0, 0, 0);
    printf("-------getNativeWindow %p----------\n",WindowTypes);
    if(NULL == WindowTypes)
    {
        printf("-------getNativeWindow fail----------\n");
        ret = FALSE;
    }
    //创建surface
    instance->surface = eglCreateWindowSurface(instance->display, instance->config[0], WindowTypes, NULL);
    if(EGL_NO_SURFACE == instance->surface)
    {
        printf("-------createsuface fail----------0x%x\n",eglGetError());
        ret = FALSE;
    }
    //绑定context和surface
    if(!eglMakeCurrent(instance->display, instance->surface, instance->surface, instance->context))
    {
        printf("-------makecurrent fail----------0x%x\n",eglGetError());
        ret = FALSE;
    }
    //查询宽高
    eglQuerySurface(instance->display,instance->surface,EGL_HEIGHT,&instance->height);
    eglQuerySurface(instance->display,instance->surface,EGL_WIDTH,&instance->width);
    //创建program
    instance->program = opengl_manager_LoadProgram(vShaderStr, fShaderStr);
    if(0 == instance->program)
    {
         printf("-------esLoadProgram fail----------\n");
         ret = FALSE;
    }
    else
    {
        printf("esLoadProgram success = %d\n",instance->program);
    }

    if(ret == TRUE)
    {
        instance ->vr = FALSE;
        if(instance->vr == FALSE)
        {
            instance->angle = 0.0f;
            instance->lrangle = 0.0f;
            instance->udangle = 0.0f;
            instance->zdistance = -100.0f;
            opengl_manager_MatrixLoadIdentity(&instance->mvpMatrix);
        }
        else
        {
            instance->angle = 0.0f;
            instance->lrangle = 0.0f;
            instance->udangle = 0.0f;
            instance->zdistance = -100.0f;
            opengl_manager_vary(context, 0.0,0.0,1.0,0.0);
            //opengl_manager_MatrixLoadIdentity(&manager_instance->mvpMatrix);
        }
    }
    
    printf("---------out---------\n");
    return ret;
}
void opengl_manager_clear(OpenGL_Manager_Context *context)
{
    printf("-------in--------");
    if(context == NULL)
    {
        return;
    }
    OpenGL_Manager_Context * manager_instance = context;

    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    eglSwapBuffers ( manager_instance->display, manager_instance->surface );
    
    manager_instance ->vr = FALSE;
    if(manager_instance->vr == FALSE)
    {
        manager_instance->angle = 0.0f;
        manager_instance->lrangle = 0.0f;
        manager_instance->udangle = 0.0f;
        manager_instance->zdistance = -100.0f;
        opengl_manager_MatrixLoadIdentity(&manager_instance->mvpMatrix);
    }
    else
    {
        manager_instance->angle = 0.0f;
        manager_instance->lrangle = 0.0f;
        manager_instance->udangle = 0.0f;
        manager_instance->zdistance = -100.0f;
        opengl_manager_vary(manager_instance, 0.0,0.0,1.0,0.0);
        //opengl_manager_MatrixLoadIdentity(&manager_instance->mvpMatrix);
    }
    int i;
    for(i = 0; i < MAX_MANAGER_MESSAGE_NUM; i++)
    {
        manager_instance->gl_message_context->event_type[i] = MANAGER_MSG_NONE;
        manager_instance->gl_message_context->event_param[i] = NULL;
    }
    printf("handle NUM = %lld ,store NUM = %lld",manager_instance->gl_message_context->c_event_num,manager_instance->gl_message_context->p_event_num);
    manager_instance->gl_message_context->c_event_num = 0;
    manager_instance->gl_message_context->p_event_num = 0;
    glFlush();
    // Delete texture object
    glDeleteTextures ( 3, manager_instance->textureid );
    glDeleteBuffers(1, &manager_instance->_vertexBuffer);    
    glDeleteBuffers(1, &manager_instance->_texCoordBuffer);
    glDeleteBuffers(1, &manager_instance->_indexBuffer);
    free(manager_instance->textureid);
  
    printf("-------out---------");
}
void opengl_manager_uninit(OpenGL_Manager_Context *context)
{
    if(context == NULL)
    {
        return;
    }
    
    OpenGL_Manager_Context * manager_instance = context;
    glFlush();
    // Delete texture object
    glDeleteTextures ( 3, manager_instance->textureid );
    glDeleteBuffers(1, &manager_instance->_vertexBuffer);    
    glDeleteBuffers(1, &manager_instance->_texCoordBuffer);
    glDeleteBuffers(1, &manager_instance->_indexBuffer);

    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Delete program object
    glDeleteProgram ( manager_instance->program );
    eglSwapBuffers ( manager_instance->display, manager_instance->surface );

    if (manager_instance->display != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(manager_instance->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (manager_instance->context != EGL_NO_CONTEXT)
        {
            eglDestroyContext(manager_instance->display, manager_instance->context);
        }
        
        if (manager_instance->surface != EGL_NO_SURFACE)
        {
            eglDestroySurface(manager_instance->display, manager_instance->surface);
        }
        
        eglTerminate(manager_instance->display);
    }

    if(NULL != manager_instance->gl_message_context)
    {
        free(manager_instance->gl_message_context);
        manager_instance->gl_message_context = NULL;
    }

    manager_instance->display = EGL_NO_DISPLAY;
    manager_instance->surface = EGL_NO_SURFACE;
    manager_instance->context = EGL_NO_CONTEXT;

    disposeNativeWindow();
    disposeNativeWindow2();
    
    free(manager_instance);
    manager_instance = NULL;
}

static void opengl_manager_msg_func(OpenGL_Manager_Context *context, MANAGER_MSG msg, void *param)
{
    switch(msg)
    {
        case MANAGER_MSG_INIT:
            opengl_manager_init(context);
            break;
        case MANAGER_MSG_PREPARE:
            {
                VIDEO_RECT_P pRect = (VIDEO_RECT_P)param;
                opengl_manager_prepare(context, pRect->width,pRect->height,pRect->x,pRect->y);
                free(pRect);
             }
            break;    
        case MANAGER_MSG_VARY:
            {
                int32_t vary = *((int32_t *)param);
                int32_t type = vary/1000;
                int32_t value = vary%1000;
                switch (type){
                    case 1:
                            if(context->vr == TRUE)
                            {
                                opengl_manager_vary_t(context, (float)value,1);
                            }
                        break;
                        
                    case 2:
                            if(context->vr == TRUE)
                            {
                                opengl_manager_vary_t(context, -(float)value,1);
                            }
                        break;
                        
                    case 3:
                            if(context->vr == TRUE)
                            {
                                opengl_manager_vary_t(context, (float)value,2);
                            }
                        break;
                        
                    case 4:
                            if(context->vr == TRUE)
                            {
                                opengl_manager_vary_t(context, -(float)value,2);
                            } 
                        break;
                        
                    case 5:
                            if(context->vr == TRUE)
                            {
                                opengl_manager_vary_c(context, -(float)value);
                            }
                        break;
                        
                    case 6:
                            if(context->vr == TRUE)
                            {
                                opengl_manager_vary_c(context, (float)value);
                            }
                       break;
                        
                    case 7:
                           if(context->vr == TRUE)
                           {
                             context->vr = FALSE;
                           }
                           else
                           {
                             context->vr = TRUE;
                           }
                           opengl_manager_change(context, context->vr);
                        break;
                        
                    default:
                        break;
                        
                    }
            }
            break;
        case MANAGER_MSG_PUSH_DATA:
            opengl_manager_push_data((OPENGL_MGR_HANDLE)context, (char* )param);
            break;
        case MANAGER_MSG_CLEAR:
            opengl_manager_clear(context);
            break; 
        case MANAGER_MSG_UNINT:
            opengl_manager_uninit(context);
            break;
        default:
            break;
    }
}

static void* opengl_manager_msg_task(void *arg)
{   
    OpenGL_Manager_Context * context = NULL;
    Manager_Message_Context  *cb_context = NULL;
    unsigned int    event_num = 0;

    context = (OpenGL_Manager_Context *)arg;
    cb_context = (Manager_Message_Context *)context->gl_message_context;
    if(cb_context == NULL)
    {
        return NULL;
    }
    
    char tname[32];
    sprintf(tname, "opengl_manager_msg_task(%d)", getpid());
 //   prctl(PR_SET_NAME, (unsigned long)tname);

    while(cb_context->task_running)
    {
        if(cb_context->clearflag)
        {
            opengl_manager_clear(context);
            cb_context->clearflag = FALSE;
            continue;
        }
        event_num = cb_context->c_event_num%MAX_MANAGER_MESSAGE_NUM;

        if(MANAGER_MSG_NONE == cb_context->event_type[event_num])
        {
            usleep(1000*100);
            continue;
        }

       // printf(" event_type( %d ), c_event_num( %lld ) .\n", cb_context->event_type[event_num],cb_context->c_event_num);
        opengl_manager_msg_func(context, cb_context->event_type[event_num], cb_context->event_param[event_num]);       
        cb_context->event_type[event_num] = MANAGER_MSG_NONE;
        cb_context->c_event_num++;
        usleep(1000*10);
    }
    return NULL;
}

 int32_t opengl_manager_start()
 {
     printf("--------in---------");
     
     OpenGL_Manager_Context * instance =(OpenGL_Manager_Context *)malloc(sizeof(OpenGL_Manager_Context));
     memset(instance, 0, sizeof(OpenGL_Manager_Context));
     
     /* for gstreamer message callback */
     Manager_Message_Context  *msg_context = NULL;
     msg_context = (Manager_Message_Context *)malloc(sizeof(Manager_Message_Context));
     if(msg_context == NULL)
     {
         return FALSE;
     }
     
     memset(msg_context->event_type, MANAGER_MSG_NONE, sizeof(msg_context->event_type));
     memset(msg_context->event_param, 0, sizeof(msg_context->event_param));
     msg_context->p_event_num = 0;
     msg_context->c_event_num = 0;
     msg_context->task_running = TRUE;
     msg_context->clearflag = FALSE;
 
     instance->gl_message_context = msg_context;
     //context = instance;
     pthread_create(&msg_context->message_thread, NULL, opengl_manager_msg_task, (void *)(instance));
     
     printf("-------out---------");
     return TRUE;
 }

 void opengl_manager_message_store(OpenGL_Manager_Context *context, MANAGER_MSG msg, void *param)
{
    Manager_Message_Context  *    message_context = NULL;
    unsigned int    event_num = 0;

    if(context && context->gl_message_context)
            message_context = context->gl_message_context;
    
    if(message_context == NULL)
    {
            printf ("store gstreamer message can NOT find cb_context.\n");
        return;
    }
    
    event_num = message_context->p_event_num%MAX_MANAGER_MESSAGE_NUM;
    
    while(message_context->event_type[event_num] != MANAGER_MSG_NONE)
    {
        usleep(1000*10);
    }

    if(NULL != message_context->event_param[event_num])
    {
        message_context->event_param[event_num] = NULL; 
    }

    /* massage filter */
    switch (msg)
    {
        /* TODO: when recieve error msg, can remove all message stroed in the array, upload error message immediately */
        /* all massage will upload righy now, temporarily, not cope with @param */
        case MANAGER_MSG_CLEAR:
            message_context->clearflag = TRUE;
            break;
        default:
           // printf ("store gstreamer message( %d ).\n",msg);
            message_context->event_type[event_num] = msg;
            message_context->event_param[event_num] = param;
            message_context->p_event_num++;
            break;
    }
    //usleep(1000*50);
    return ;
}

int  getYV12Data(FILE * fp, char * pYUVData, int size, int offset)
{
    int ret = -1;
    //FILE *fp = fopen(path,"rb");
    if(fp == NULL)
    {
        printf("fp == NULL!\n");
        return -1;
    }
    
    ret = fseek(fp, size * offset, SEEK_SET);
    if(ret < 0)
    {
        return -1;
    }
    
    fread(pYUVData, size, 1, fp);
    //fclose(fp);
    
    return 0;
}

OPENGL_MGR_HANDLE initOpenGL(int yuv_width, int yuv_height)
{
    //int yuv_width = 1280;
    //int yuv_height = 720;
    //int size = yuv_width * yuv_height * 3/2;
    //char * data = NULL;
    //char * path = "/data/temp/yuv_video/1280x720_yuv420p.yuv";
    //char * path = "/mnt/usb/967CEBDC7CEBB4E1/yuv/1920x1080_video_yuv.yuv";
    //OpenGL_Manager_Context *context = NULL;
    OpenGL_Manager_Context * context = NULL; 
    
    context = (OpenGL_Manager_Context *)malloc(sizeof(OpenGL_Manager_Context));
    memset(context, 0, sizeof(OpenGL_Manager_Context));

    opengl_manager_init(context);
    opengl_manager_prepare(context, yuv_width, yuv_height, 0, 0);

    return (OPENGL_MGR_HANDLE)context;
}

void uninitOpenGL(OPENGL_MGR_HANDLE *handle)
{
    OpenGL_Manager_Context * context = (OpenGL_Manager_Context *)*handle;

    opengl_manager_uninit(context);
    *handle = NULL;
}


void opengl_manager_push_data(OPENGL_MGR_HANDLE context, char *data)
{
    if(context == NULL)
    {
        return ;
    }
    OpenGL_Manager_Context * manager_instance = (OpenGL_Manager_Context *)context;
    opengl_manager_updateTexture(manager_instance, manager_instance->textureid, (u_int8_t *)data, manager_instance->video_width, manager_instance->video_height); 

    // Draw 
    glUniformMatrix4fv(glGetUniformLocation (manager_instance->program,  "u_mvpMatrix"), 1, GL_FALSE, ( GLfloat * ) &manager_instance->mvpMatrix.m[0][0]);
    glDrawElements ( GL_TRIANGLE_STRIP,manager_instance->vertexnum, GL_UNSIGNED_SHORT, 0);
    //printf("glvsink->vertexnum = %d",glvsink->vertexnum);
    eglSwapBuffers ( manager_instance->display, manager_instance->surface );
}


#if 0
int main(int argc, char **argv)
{
    int yuv_width = 1280;
    int yuv_height = 720;
    int ret = 0;
    int size = yuv_width * yuv_height * 3/2;
    unsigned int i = 0;
    char * data = NULL;
    char * path = "/data/temp/yuv_video/1280x720_yuv420p.yuv";
    //char * path = "/mnt/usb/967CEBDC7CEBB4E1/yuv/1920x1080_video_yuv.yuv";
    OpenGL_Manager_Context *context = NULL;
    FILE *fp = NULL;
    data = (char *)malloc(size); 
    
    context = (OpenGL_Manager_Context *)malloc(sizeof(OpenGL_Manager_Context));
    memset(context, 0, sizeof(OpenGL_Manager_Context));

    opengl_manager_init(context);
    opengl_manager_prepare(context, yuv_width, yuv_height, 0, 0);

    fp = fopen(path,"rb");
    do
    {
        for(i = 1; ret >= 0; i++)
        {
            memset(data, 0, size);
            ret = getYV12Data(fp, data, size, i);//get yuv data from file;
            if(ret < 0)
            {
                printf("play end\n");
                break;
            }
            
            opengl_manager_push_data(context, data, context->surface);
        }

        ret = 0;
    }while(1);
    
    //IPCThreadState::self()->joinThreadPool();//可以保证画面一直显示，否则瞬间消失
    //IPCThreadState::self()->stopProcess();
EXIT:
    if(NULL == fp)
    {
        fclose(fp);
        fp = NULL;
    }
    
    if(NULL == data)
    {
        free(data);
        data = NULL;
    }
    
    opengl_manager_uninit(context);
    
    return 0;
}
#endif
