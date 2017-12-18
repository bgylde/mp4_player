#ifndef __ANDROID_SURFACE_H__
#define __ANDROID_SURFACE_H__


#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

EGLNativeWindowType getNativeWindow(int width, int hight, int position_x, int position_y, int type);
void disposeNativeWindow(void);

EGLNativeWindowType getNativeWindow2(int width, int hight, int position_x, int position_y, int type);
void disposeNativeWindow2(void);


#endif
