#ifndef WAI_H
#define WAI_H

#include <map>

#include <WAIHelper.h>
#include <WAIMath.h>
#include <WAISensor.h>
#include <WAIMode.h>
#include <WAIModeAruco.h>
#include <WAIModeOrbSlam2.h>
#include <WAIModeOrbSlam2DataOriented.h>
#include <KPextractor.h>

namespace WAI
{

class WAI_API WAI
{
    public:
    WAI(std::string dataRoot);
    void  setDataRoot(std::string dataRoot);
    void  activateSensor(SensorType sensorType, void* sensorInfo);
    void  updateSensor(SensorType type, void* value);
    bool  whereAmI(cv::Mat* pose);
    Mode* setMode(ModeType mode);
    Mode* getCurrentMode();

    private:
    std::string                   _dataRoot;
    Mode*                         _mode = nullptr;
    std::map<SensorType, Sensor*> _sensors;
    KPextractor                   *_extractor;
};
}

#endif
