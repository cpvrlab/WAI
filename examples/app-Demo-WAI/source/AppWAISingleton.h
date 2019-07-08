#ifndef APPWAISINGLETON
#define APPWAISINGLETON
#include <WAI.h>
#include <WAICalibration.h>
#include <WAIAutoCalibration.h>
#include <AppWAIScene.h>

class AppWAISingleton
{
    public:
    static AppWAISingleton* instance();

    void            load(int width, int height, std::string path);
    void            load(int width, int height, std::string path, WAICalibration* calibration);
    void            load(int width, int height, std::string path, std::string externalPath);
    std::string     root_path;
    WAI::WAI*       wai;
    WAICalibration* wc;
    AppWAIScene*    appWaiScene;
    int             scrWidth;
    int             scrHeight;

    cv::VideoWriter videoWriter;
    cv::VideoWriter videoWriterInfo;

    private:
    AppWAISingleton();
    static AppWAISingleton* _instance;
};
#endif
