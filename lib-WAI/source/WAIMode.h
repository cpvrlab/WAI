#ifndef WAI_MODE_H
#define WAI_MODE_H

#include <WAIMath.h>

namespace WAI
{

enum ModeType
{
    ModeType_None,
    ModeType_Aruco,
    ModeType_ORB_SLAM2
};

class Mode
{
    public:
    virtual ~Mode()                  = 0;
    virtual bool getPose(M4x4* pose) = 0;
    virtual void notifyUpdate()      = 0;
};
}

#endif
