#ifndef _OPENGL_MANAGER_H_
#define _OPENGL_MANAGER_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void* OPENGL_MGR_HANDLE;
void opengl_manager_push_data(OPENGL_MGR_HANDLE context, char *data);
void uninitOpenGL(OPENGL_MGR_HANDLE *handle);
OPENGL_MGR_HANDLE initOpenGL(int yuv_width, int yuv_height);

int  getYV12Data(FILE * fp, char * pYUVData, int size, int offset);


#ifdef __cplusplus
}
#endif

#endif
