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
    OrbSlamStatus_Initializing,
    OrbSlamStatus_Tracking
};

struct MapPoint
{
    cv::Mat                          position;
    cv::Mat                          normalVector;
    cv::Mat                          descriptor;
    std::vector<std::pair<i32, i32>> observations; // each pair contains the index of the keyframe in _state.keyframes and the index of the keyframes keypoint
};

struct KeyFrame
{
    i32 numberOfKeyPoints;

    std::vector<cv::KeyPoint> keyPoints; // only used for visualization
    std::vector<cv::KeyPoint> undistortedKeyPoints;
    std::vector<i32>          mapPointIndices;   // same size as keyPoints, initialized with -1
    std::vector<bool32>       mapPointIsOutlier; // same size as keyPoints

    std::vector<size_t> keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    cv::Mat             descriptors;
    cv::Mat             cTw;
};

struct PyramidOctTreeCell
{
    r32 minX, maxX, minY, maxY;
    i32 imagePyramidLevel;
    i32 xOffset, yOffset;
};

struct PyramidOctTreeLevel
{
    i32                             minBorderX;
    i32                             minBorderY;
    i32                             maxBorderX;
    i32                             maxBorderY;
    std::vector<PyramidOctTreeCell> cells;
};

struct PyramidOctTree
{
    std::vector<PyramidOctTreeLevel> levels;
};

struct ImagePyramidStats
{
    i32              numberOfScaleLevels;
    std::vector<r32> scaleFactors;
    std::vector<r32> inverseScaleFactors;
    std::vector<r32> sigmaSquared;
    std::vector<r32> inverseSigmaSquared;
    std::vector<i32> numberOfFeaturesPerScaleLevel;
};

struct FastFeatureConstraints
{
    i32 initialThreshold;
    i32 minimalThreshold;
    i32 numberOfFeatures;
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

    // pyramid + orb stuff
    i32                    edgeThreshold;
    i32                    orbOctTreePatchSize;
    i32                    orbOctTreeHalfPatchSize;
    std::vector<i32>       umax;
    std::vector<cv::Point> pattern;

    // initialization stuff
    std::vector<cv::Point2f> previouslyMatchedKeyPoints;

    // camera stuff
    r32 fx, fy, cx, cy;
    r32 invfx, invfy;

    ImagePyramidStats      imagePyramidStats;
    GridConstraints        gridConstraints;
    FastFeatureConstraints fastFeatureConstraints;

    std::vector<KeyFrame> keyFrames;
    i32                   keyFrameCount;
    std::vector<MapPoint> mapPoints;
};

namespace WAI
{

class WAI_API ModeOrbSlam2DataOriented : public Mode
{
    public:
    ModeOrbSlam2DataOriented(SensorCamera* camera);
    void notifyUpdate();
    bool getPose(cv::Mat* pose);

    std::vector<MapPoint> getMapPoints();
    i32                   getMapPointCount() { return _state.mapPoints.size(); }

    private:
    SensorCamera* _camera;
    OrbSlamState  _state;
    cv::Mat       _pose;
};
}

#endif