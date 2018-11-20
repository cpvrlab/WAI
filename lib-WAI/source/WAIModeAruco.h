#ifndef WAI_MODE_ARUCO_H

#    include <WAIMode.h>
#    include <WAISensorCamera.h>

namespace WAI
{

class ModeAruco : public Mode
{
    public:
    ModeAruco(SensorCamera* camera);
    ~ModeAruco() {}
    bool getPose(M4x4* pose);
    void notifyUpdate() {}

    private:
    SensorCamera*                          _camera;
    cv::Ptr<cv::aruco::DetectorParameters> _arucoParams;
    cv::Ptr<cv::aruco::Dictionary>         _dictionary;
    float                                  _edgeLength;
};
}

#    define WAI_MODE_ARUCO_H
#endif
