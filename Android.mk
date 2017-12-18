LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_ADDITIONAL_DEPENDENCIES := $(LOCAL_PATH)/Android.mk
LOCAL_SRC_FILES:= \
		codec.cpp \
		SimplePlayer.cpp \
		opengl_manager.cpp \
		android_surface.cpp 

LOCAL_SHARED_LIBRARIES := \
		libstagefright liblog libutils libbinder libstagefright_foundation \
		libmedia libgui libcutils libui \
		libEGL \
		libGLESv2 \
		libbinder \
		libcutils \
		libgui \
		libmedia \
		libstagefright \
		libstagefright_foundation \
		libstagefright_omx \
		libsync \
		libui \
		libutils \
		liblog 
		

LOCAL_C_INCLUDES:= \
		frameworks/av/media/libstagefright \
		frameworks/av/media/libstagefright/include \
		$(TOP)/frameworks/native/include/media/openmax

#LOCAL_CFLAGS += -Wno-multichar -Werror -Wall
LOCAL_CLANG := true

LOCAL_MODULE_TAGS := mp4_player

LOCAL_MODULE:= mp4_player

include $(BUILD_EXECUTABLE)
