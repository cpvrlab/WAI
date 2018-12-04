#ifndef WAI_H
#define WAI_H

#include <map>

#include <WAIMath.h>
#include <WAISensor.h>
#include <WAIMode.h>
#include <WAIModeAruco.h>
#include <WAIModeOrbSlam2.h>
#include <WAIFileSystem.h>

#ifdef __APPLE__
#    include <TargetConditionals.h>
#    if TARGET_OS_IOS
#        define WAI_OS_MACIOS
#    else
#        define WAI_OS_MACOS
#    endif
#    define WAI_LOG(...) printf(__VA_ARGS__)
#elif defined(ANDROID) || defined(ANDROID_NDK)
#    include <android/log.h>
#    define WAI_OS_ANDROID
#    define WAI_LOG(...) __android_log_print(ANDROID_LOG_INFO, "lib-WAI", __VA_ARGS__)
#elif defined(_WIN32)
#    define WAI_OS_WINDOWS
#    define STDCALL __stdcall
#    define WAI_LOG(...) printf(__VA_ARGS__)
#elif defined(linux) || defined(__linux) || defined(__linux__)
#    define WAI_OS_LINUX
#    define WAI_LOG(...) printf(__VA_ARGS__)
#else
#    error "WAI has not been ported to this OS"
#endif

namespace WAI
{

#if WAI_BUILD_DEBUG
#    define wai_assert(expression) \
        if (!(expression)) { *(int*)0 = 0; }
#else
#    define wai_assert(expression)
#endif

class WAI
{
    public:
    WAI(std::string dataRoot);
    void  activateSensor(SensorType sensorType, void* sensorInfo);
    void  updateSensor(SensorType type, void* value);
    bool  whereAmI(cv::Mat* pose);
    Mode* setMode(ModeType mode);

    private:
    std::string                   _dataRoot;
    Mode*                         _mode = nullptr;
    std::map<SensorType, Sensor*> _sensors;
};
}

#endif
