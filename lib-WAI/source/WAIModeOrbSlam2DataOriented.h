#ifndef WAI_MODE_ORB_SLAM_2_DATA_ORIENTED
#define WAI_MODE_ORB_SLAM_2_DATA_ORIENTED

#include <WAIPlatform.h>
#include <WAIOrbPattern.h>
#include <WAISensorCamera.h>

#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

#define FRAME_GRID_ROWS 36 //48
#define FRAME_GRID_COLS 64

enum OrbSlamStatus
{
    OrbSlamStatus_None,
    OrbSlamStatus_Initializing,
    OrbSlamStatus_Tracking
};

struct MapPointTrackingInfos
{
    bool32 inView;
    r32    projX;
    r32    projY;
    r32    scaleLevel;
    r32    viewCos;
};

struct KeyFrame;

struct MapPoint
{
    i32 index;

    cv::Mat                  position;
    cv::Mat                  normalVector;
    cv::Mat                  descriptor;
    std::map<KeyFrame*, i32> observations; // key is pointer to a keyframe, value is index into that keyframes keypoint vector

    bool32 bad;

    r32 maxDistance;
    r32 minDistance;

    KeyFrame* firstObservationKeyFrame;
    i32       foundInKeyFrameCounter;
    i32       visibleInKeyFrameCounter;

    i32 trackReferenceForFrame;
    i32 lastFrameSeen;

    MapPointTrackingInfos trackingInfos; // TODO(jan): this should not be in here
};

struct KeyFrame
{
    i32 index;
    i32 numberOfKeyPoints;

    std::vector<cv::KeyPoint> keyPoints; // only used for visualization
    std::vector<cv::KeyPoint> undistortedKeyPoints;
    std::vector<MapPoint*>    mapPointMatches;   // same size as keyPoints, initialized with -1
    std::vector<bool32>       mapPointIsOutlier; // same size as keyPoints

    std::vector<size_t> keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    cv::Mat             descriptors;
    cv::Mat             cTw;
    cv::Mat             wTc;
    cv::Mat             worldOrigin;

    KeyFrame*              parent;
    std::vector<KeyFrame*> children; // children in covisibility graph
    i32                    trackReferenceForFrame;

    std::vector<KeyFrame*>   orderedConnectedKeyFrames;
    std::vector<i32>         orderedWeights;
    std::map<KeyFrame*, i32> connectedKeyFrameWeights;

    KeyFrame* referenceKeyFrame;

    DBoW2::BowVector     bowVector;
    DBoW2::FeatureVector featureVector;
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
    r32 maxX;
    r32 maxY;
    r32 invGridElementWidth;
    r32 invGridElementHeight;
};

struct LocalMappingState
{
    std::list<KeyFrame*> newKeyFrames; // TODO(jan): replace with vector?
};

struct OrbSlamState
{
    OrbSlamStatus status;
    bool32        trackingWasOk;

    // local map
    std::vector<KeyFrame*> localKeyFrames;
    std::vector<MapPoint*> localMapPoints;

    // pyramid + orb stuff
    i32                    edgeThreshold;
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

    ImagePyramidStats      imagePyramidStats;
    GridConstraints        gridConstraints;
    FastFeatureConstraints fastFeatureConstraints;

    std::vector<KeyFrame*> keyFrames;
    std::vector<MapPoint*> mapPoints;
    i32                    nextKeyFrameId;
    i32                    nextMapPointId;

    KeyFrame* referenceKeyFrame;

    i32 lastKeyFrameId;
    i32 lastRelocalizationKeyFrameId;

    r32 scaleFactor;

    i32 frameCounter;

    ORBVocabulary* orbVocabulary;

    LocalMappingState localMapping;
};

static inline cv::Mat getKeyFrameRotation(const KeyFrame* keyFrame)
{
    cv::Mat result = keyFrame->cTw.rowRange(0, 3).colRange(0, 3).clone();

    return result;
}

static inline cv::Mat getKeyFrameTranslation(const KeyFrame* keyFrame)
{
    cv::Mat result = keyFrame->cTw.rowRange(0, 3).col(3).clone();

    return result;
}

static inline cv::Mat getKeyFrameCameraCenter(const KeyFrame* keyFrame)
{
    cv::Mat result = keyFrame->worldOrigin.clone();

    return result;
}

namespace WAI
{

class WAI_API ModeOrbSlam2DataOriented : public Mode
{
    public:
    ModeOrbSlam2DataOriented(SensorCamera* camera, std::string vocabularyPath);
    void notifyUpdate();
    bool getPose(cv::Mat* pose);

    std::vector<MapPoint*> getMapPoints();
    i32                    getMapPointCount() { return _state.mapPoints.size(); }

    private:
    SensorCamera* _camera;
    OrbSlamState  _state;
    cv::Mat       _pose;
};
}

#endif