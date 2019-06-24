
#include <AppWAISingleton.h>
#include <WAI.h>
#include <WAIModeOrbSlam2.h>
#include <WAICalibration.h>
#include <WAIAutoCalibration.h>

AppWAISingleton* AppWAISingleton::_instance = 0;

AppWAISingleton::AppWAISingleton()
{
}

AppWAISingleton* AppWAISingleton::instance()
{
    if (_instance == nullptr)
    {
        _instance = new AppWAISingleton();
    }
    return _instance;
}

void AppWAISingleton::load(int width, int height, std::string path)
{
    scrWidth  = width;
    scrHeight = height;
    root_path = path;

    wai         = new WAI::WAI(path + "/data");
    appWaiScene = new AppWAIScene();
    wc          = new WAICalibration();

    wc->changeImageSize(width, height);
    // TODO(jan): this needs to be dynamic
    //wc->loadFromFile(root_path + "/data/calibrations/cam_calibration_unity_drone.xml");
    WAI::CameraCalibration calibration;
    calibration = wc->getCameraCalibration();
    wai->activateSensor(WAI::SensorType_Camera, &calibration);
}

void AppWAISingleton::load(int width, int height, std::string path, WAICalibration* c)
{
    scrWidth  = width;
    scrHeight = height;
    root_path = path;

    wai         = new WAI::WAI(path + "/data");
    appWaiScene = new AppWAIScene();
    wc          = c;

    // TODO(jan): need a better way to define calibrations
    wc->changeImageSize(width, height);
    wc->loadFromFile(root_path + "/data/calibrations/cam_calibration_main.xml");
    WAI::CameraCalibration calibration;
    calibration = wc->getCameraCalibration();
    wai->activateSensor(WAI::SensorType_Camera, &calibration);
}