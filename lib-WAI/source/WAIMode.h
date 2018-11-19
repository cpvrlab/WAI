#ifndef WAI_MODE_H

#    include <WAIMath.h>

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
    virtual bool getPose(M4x4* pose) = 0;
    virtual ~Mode()                  = 0;
};
}

#    define WAI_MODE_H
#endif
