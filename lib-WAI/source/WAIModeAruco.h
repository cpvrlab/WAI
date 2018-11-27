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
    bool getPose(cv::Mat* pose);
    void notifyUpdate() {}
    bool getDebugInfo(DebugInfoType type, void* memory) { return false; }

    private:
    SensorCamera*                          _camera;
    cv::Ptr<cv::aruco::DetectorParameters> _arucoParams;
    cv::Ptr<cv::aruco::Dictionary>         _dictionary;
    float                                  _edgeLength;
};
}

#    define WAI_MODE_ARUCO_H
#endif
