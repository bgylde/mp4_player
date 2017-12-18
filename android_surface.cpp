#include <stdio.h>
#include <include/SoftwareRenderer.h>

#include <cutils/memory.h>

#include <unistd.h>
#include <utils/Log.h>

#include <binder/IPCThreadState.h>
#include <binder/ProcessState.h>
#include <binder/IServiceManager.h>

#include <gui/Surface.h>
#include <gui/SurfaceComposerClient.h>
#include <gui/ISurfaceComposer.h>
#include <ui/DisplayInfo.h>
#include <android/native_window.h>
#include <media/stagefright/MetaData.h>
#include <media/stagefright/Utils.h>
#include <media/stagefright/foundation/AMessage.h>


#include <ui/GraphicBuffer.h>
#include <gui/Surface.h>
#include <gui/ISurfaceComposer.h>
#include <gui/Surface.h>
#include <gui/SurfaceComposerClient.h>


using namespace android;


sp<SurfaceComposerClient>  mComposerClient = NULL;
sp<SurfaceControl>   mSurfaceControl = NULL;

EGLNativeWindowType getNativeWindow(int width, int hight, int position_x, int position_y, int type)
{
    DisplayInfo dinfo;
    
    mComposerClient = new SurfaceComposerClient();
    sp<IBinder> dtoken(SurfaceComposerClient::getBuiltInDisplay(
                        ISurfaceComposer::eDisplayIdMain));

    status_t status = SurfaceComposerClient::getDisplayInfo(dtoken, &dinfo);
    printf("w=%d,h=%d,xdpi=%f,ydpi=%f,fps=%f,ds=%f\n", 
                dinfo.w, dinfo.h, dinfo.xdpi, dinfo.ydpi, dinfo.fps, dinfo.density);

    mSurfaceControl = mComposerClient->createSurface(
        String8("Test Surface"),
        dinfo.w, dinfo.h,
        PIXEL_FORMAT_RGBA_8888, 0);

    SurfaceComposerClient::openGlobalTransaction();
    mSurfaceControl->setLayer(100000);//设定Z坐标
    mSurfaceControl->setPosition(position_x, position_y);
    mSurfaceControl->setSize(width, hight);

    SurfaceComposerClient::closeGlobalTransaction();

    sp<ANativeWindow> window = mSurfaceControl->getSurface();

    return window.get();
}


void disposeNativeWindow(void)
{
    if (mComposerClient != NULL) {
        mComposerClient->dispose();
        mComposerClient = NULL;
        mSurfaceControl = NULL;
    }
}


sp<SurfaceComposerClient>  mComposerClient2 = NULL;
sp<SurfaceControl>   mSurfaceControl2 = NULL;
EGLNativeWindowType getNativeWindow2(int width, int hight, int position_x, int position_y, int type)
{
    DisplayInfo dinfo;
    
    mComposerClient2 = new SurfaceComposerClient();
    sp<IBinder> dtoken(SurfaceComposerClient::getBuiltInDisplay(
                        ISurfaceComposer::eDisplayIdMain));

    status_t status = SurfaceComposerClient::getDisplayInfo(dtoken, &dinfo);
    printf("w=%d,h=%d,xdpi=%f,ydpi=%f,fps=%f,ds=%f\n", 
                dinfo.w, dinfo.h, dinfo.xdpi, dinfo.ydpi, dinfo.fps, dinfo.density);

    mSurfaceControl2 = mComposerClient2->createSurface(
        String8("Test Surface"),
        dinfo.w, dinfo.h,
        PIXEL_FORMAT_RGBA_8888, 0);

    SurfaceComposerClient::openGlobalTransaction();
    mSurfaceControl2->setLayer(100000);//设定Z坐标
    mSurfaceControl2->setPosition(position_x, position_y);
    mSurfaceControl2->setSize(width, hight);

    SurfaceComposerClient::closeGlobalTransaction();

    sp<ANativeWindow> window = mSurfaceControl2->getSurface();

    return window.get();
}

void disposeNativeWindow2(void)
{
    if (mComposerClient2 != NULL) {
        mComposerClient2->dispose();
        mComposerClient2 = NULL;
        mSurfaceControl2 = NULL;
    }
}