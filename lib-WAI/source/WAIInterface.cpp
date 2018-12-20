#include <WAI.h>
#include <WAIMath.h>

static WAI::WAI           wai("");
static WAI::ModeOrbSlam2* mode = nullptr;

typedef void (*DebugCallback)(const char* str);
DebugCallback gDebugCallback;

void debugInUnity(std::string message)
{
    if (gDebugCallback)
    {
        gDebugCallback(message.c_str());
    }
}

extern "C" {
WAI_API void wai_setMode(WAI::ModeType modeType)
{
    debugInUnity("setMode called");
    mode = (WAI::ModeOrbSlam2*)wai.setMode(modeType);
}

WAI_API void wai_activateSensor(WAI::SensorType sensorType, void* sensorInfo)
{
    fprintf(stdout, "activateSensor called\n");
    fflush(stdout);
    wai.activateSensor(sensorType, sensorInfo);
}

WAI_API void wai_updateCamera(WAI::CameraFrame* frameRGB, WAI::CameraFrame* frameGray)
{
    fprintf(stdout, "updateCamera called\n");
    fflush(stdout);
    cv::Mat cvFrameRGB  = cv::Mat(frameRGB->height,
                                 frameRGB->width,
                                 CV_8UC3,
                                 frameRGB->memory,
                                 frameRGB->pitch);
    cv::Mat cvFrameGray = cv::Mat(frameGray->height,
                                  frameGray->width,
                                  CV_8UC1,
                                  frameGray->memory,
                                  frameGray->pitch);

    WAI::CameraData sensorData = {&cvFrameGray, &cvFrameRGB};

    wai.updateSensor(WAI::SensorType_Camera, &sensorData);
}

WAI_API bool wai_whereAmI(WAI::M4x4* pose)
{
    fprintf(stdout, "whereAmI called\n");
    fflush(stdout);
    bool result = 0;

    cv::Mat cvPose = cv::Mat(4, 4, CV_32F);
    result         = wai.whereAmI(&cvPose);

    if (result)
    {
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                pose->e[x][y] = cvPose.at<float>(x, y);
            }
        }
    }

    return result;
}

WAI_API int wai_getState(char* buffer, int size)
{
    fprintf(stdout, "getState called\n");
    fflush(stdout);
    int result = 0;

    if (mode)
    {
        std::string state = mode->getPrintableState();

        if ((state.size() + 1) < size)
        {
            size = state.size() + 1;
        }

        result        = size;
        char*       c = buffer;
        const char* s = state.c_str();

        strncpy(c, s, size);
    }

    return result;
}

WAI_API void wai_registerDebugCallback(DebugCallback callback)
{
    if (callback)
    {
        gDebugCallback = callback;
    }
}
}
