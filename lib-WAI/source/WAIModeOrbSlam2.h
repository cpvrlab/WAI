#ifndef WAI_MODE_ORB_SLAM_2
#define WAI_MODE_ORB_SLAM_2

#include <thread>

#include <WAIMode.h>
#include <WAISensorCamera.h>
#include <WAIKeyFrameDB.h>
#include <WAIMap.h>
#include <WAIOrbVocabulary.h>

#include <OrbSlam/LocalMapping.h>
#include <OrbSlam/LoopClosing.h>

namespace WAI
{

class ModeOrbSlam2 : public Mode
{
    public:
    ModeOrbSlam2(SensorCamera* camera, bool serial);
    ~ModeOrbSlam2();
    bool getPose(M4x4* pose);

    private:
    SensorCamera*             _camera;
    ORB_SLAM2::ORBVocabulary* mpVocabulary;
    WAIKeyFrameDB*            mpKeyFrameDatabase;
    WAIMap*                   _map;
    bool                      _initialized;
    ORB_SLAM2::ORBextractor*  _extractor;
    ORB_SLAM2::ORBextractor*  mpIniORBextractor;
    ORB_SLAM2::LocalMapping*  mpLocalMapper;
    ORB_SLAM2::LoopClosing*   mpLoopCloser;
    std::thread*              mptLocalMapping;
    std::thread*              mptLoopClosing;
    bool                      _serial;
};
}

#endif
