#ifndef WAI_MODE_ORB_SLAM_2_DATA_ORIENTED
#define WAI_MODE_ORB_SLAM_2_DATA_ORIENTED

#include <WAIPlatform.h>
#include <WAIOrbPattern.h>
#include <WAISensorCamera.h>

#define FRAME_GRID_ROWS 36 //48
#define FRAME_GRID_COLS 64

enum OrbSlamStatus
{
    OrbSlamStatus_None,
    OrbSlamStatus_Initializing
};

struct KeyFrame
{
    i32                       numberOfKeyPoints;
    std::vector<cv::KeyPoint> keyPoints; // only used for visualization
    std::vector<cv::KeyPoint> undistortedKeyPoints;
    std::vector<size_t>       keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    cv::Mat                   descriptors;
};

struct ImagePyramidStats
{
    i32              numberOfScaleLevels;
    std::vector<r32> scaleFactors;
    std::vector<r32> inverseScaleFactors;
    std::vector<i32> numberOfFeaturesPerScaleLevel;
};

struct GridConstraints
{
    r32 minX;
    r32 minY;
    r32 invGridElementWidth;
    r32 invGridElementHeight;
};

struct OrbSlamState
{
    OrbSlamStatus status;
    KeyFrame*     referenceKeyFrame = nullptr; // keyframe no. 0

    // pyramid + orb stuff
    i32                    edgeThreshold;
    i32                    numberOfFeatures;
    i32                    initialFastThreshold;
    i32                    minimalFastThreshold;
    i32                    orbOctTreePatchSize;
    i32                    orbOctTreeHalfPatchSize;
    std::vector<i32>       umax;
    std::vector<cv::Point> pattern;

    // initialization stuff
    std::vector<cv::Point2f> previouslyMatchedKeyPoints;
    std::vector<i32>         initializationMatches;

    // camera stuff
    r32 fx, fy, cx, cy;
    r32 invfx, invfy;

    ImagePyramidStats imagePyramidStats;
    GridConstraints   gridConstraints;
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