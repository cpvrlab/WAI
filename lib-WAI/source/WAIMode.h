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
    ModeType_ORB_SLAM2,
    ModeType_ORB_SLAM2_DATA_ORIENTED
};

class Mode
{
    public:
    virtual ~Mode()                     = 0;
    virtual bool getPose(cv::Mat* pose) = 0;
    virtual void notifyUpdate()         = 0;
};
}

#endif
