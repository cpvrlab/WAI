#ifndef WAI_INITIALIZER_ORBSLAM_H
#define WAI_INITIALIZER_ORBSLAM_H

#include <WAIInitializer.h>
#include <WAISensorCamera.h>
#include <WAIMap.h>
#include <WAIKeyFrameDB.h>

#include <OrbSlam/ORBVocabulary.h>
#include <OrbSlam/Initializer.h>

#include <KPextractor.h>

namespace WAI
{

class WAIInitializerOrbSlam : public WAIInitializer
{
    public:
    WAIInitializerOrbSlam(WAIMap*                   map,
                          WAIKeyFrameDB*            keyFrameDB,
                          KPextractor*              kpExtractor,
                          ORB_SLAM2::ORBVocabulary* orbVoc,
                          SensorCamera*             camera,
                          bool                      serial,
                          bool                      retainImg);
    InitializationResult initialize();
    void                 reset();

    private:
    WAIMap*        _map        = nullptr;
    WAIKeyFrameDB* _keyFrameDB = nullptr;
    WAIFrame       _initialFrame;
    WAIFrame       _currentFrame;
    SensorCamera*  _camera = nullptr;

    ORB_SLAM2::ORBVocabulary* _orbVoc      = nullptr;
    ORB_SLAM2::Initializer*   _initializer = nullptr;

    KPextractor* _kpExtractor = nullptr;

    bool _serial;
    bool _retainImg;

    std::vector<cv::Point2f> _prevMatched;
    std::vector<int>         _iniMatches;
    std::vector<cv::Point3f> _3DpointsIniMatched;

    bool createInitialMap(WAIKeyFrame** kfIni,
                          WAIKeyFrame** kfCur);
};

};

#endif
