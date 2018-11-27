#ifndef WAI_MODE_H
#define WAI_MODE_H

#include <opencv2/core.hpp>

#include <WAIMath.h>

namespace WAI
{

enum ModeType
{
    ModeType_None,
    ModeType_Aruco,
    ModeType_ORB_SLAM2
};

enum DebugInfoType
{
    DebugInfoType_None,
    DebugInfoType_Mappoints,
    DebugInfoType_MappointsMatched,
    DebugInfoType_MappointsLocal,
    DebugInfoType_Keyframes
};

class Mode
{
    public:
    virtual ~Mode()                                             = 0;
    virtual bool getPose(cv::Mat* pose)                         = 0;
    virtual void notifyUpdate()                                 = 0;
    virtual bool getDebugInfo(DebugInfoType type, void* memory) = 0;
};
}

#endif
