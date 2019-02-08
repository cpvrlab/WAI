#ifndef WAI_MODE_ORB_SLAM_2_DATA_ORIENTED
#define WAI_MODE_ORB_SLAM_2_DATA_ORIENTED

#include <WAIPlatform.h>
#include <WAIOrbPattern.h>
#include <WAISensorCamera.h>

enum OrbSlamStatus
{
    OrbSlamStatus_None,
    OrbSlamStatus_Initializing
};

struct OrbSlamState
{
    OrbSlamStatus          status;
    i32                    pyramidScaleLevels;
    std::vector<r32>       pyramidScaleFactors;
    std::vector<r32>       inversePyramidScaleFactors;
    i32                    edgeThreshold;
    i32                    numberOfFeatures;
    std::vector<i32>       numberOfFeaturesPerScaleLevel;
    i32                    initialFastThreshold;
    i32                    minimalFastThreshold;
    i32                    orbOctTreePatchSize;
    i32                    orbOctTreeHalfPatchSize;
    std::vector<i32>       umax;
    std::vector<cv::Point> pattern;
    r32                    fx, fy, cx, cy;
    r32                    invfx, invfy;
    r32                    invGridElementWidth, invGridElementHeight;
};

namespace WAI
{

class WAI_API ModeOrbSlam2DataOriented : public Mode
{
    public:
    ModeOrbSlam2DataOriented(SensorCamera* camera);
    void notifyUpdate();
    bool getPose(cv::Mat* pose) { return false; }

    private:
    SensorCamera* _camera;
    OrbSlamState  _state;
};
}

#endif