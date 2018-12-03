#ifndef WAI_H
#define WAI_H

#include <map>

#include <WAIMath.h>
#include <WAISensor.h>
#include <WAIMode.h>
#include <WAIModeAruco.h>
#include <WAIModeOrbSlam2.h>

#ifdef __APPLE__
#    include <TargetConditionals.h>
#    if TARGET_OS_IOS
#        define WAI_OS_MACIOS
#    else
#        define WAI_OS_MACOS
#    endif
#elif defined(ANDROID) || defined(ANDROID_NDK)
#    define WAI_OS_ANDROID
#elif defined(_WIN32)
#    define WAI_OS_WINDOWS
#    define STDCALL __stdcall
#elif defined(linux) || defined(__linux) || defined(__linux__)
#    define WAI_OS_LINUX
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
    WAI() {}
    void  activateSensor(SensorType sensorType, void* sensorInfo);
    void  updateSensor(SensorType type, void* value);
    bool  whereAmI(cv::Mat* pose);
    Mode* setMode(ModeType mode);

    private:
    Mode*                         _mode = 0;
    std::map<SensorType, Sensor*> _sensors;
};
}

#endif
