LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

LOCAL_SRC_FILES:= \
		codec.cpp \
		SimplePlayer.cpp

LOCAL_SHARED_LIBRARIES := \
		libstagefright liblog libutils libbinder libstagefright_foundation \
		libmedia libgui libcutils libui

LOCAL_C_INCLUDES:= \
		frameworks/av/media/libstagefright \
		$(TOP)/frameworks/native/include/media/openmax

#LOCAL_CFLAGS += -Wno-multichar -Werror -Wall
LOCAL_CLANG := true

LOCAL_MODULE_TAGS := test_player

LOCAL_MODULE:= test_player

include $(BUILD_EXECUTABLE)
