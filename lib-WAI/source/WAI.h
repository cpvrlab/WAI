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
    void setMode(ModeType mode);
    void activateSensor(SensorType sensorType, void* sensorInfo);
    void updateSensor(SensorType type, void* value);
    bool getDebugInfo(DebugInfoType type, void* memory);
    bool whereAmI(cv::Mat* pose);

    private:
    Mode*                         _mode = 0;
    std::map<SensorType, Sensor*> _sensors;
};
}

#endif
