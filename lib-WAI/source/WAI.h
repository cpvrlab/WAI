#ifndef WAI_H
#define WAI_H

#include <map>

#include <WAIMath.h>
#include <WAISensor.h>
#include <WAIMode.h>
#include <WAIModeAruco.h>
#include <WAIModeOrbSlam2.h>

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
