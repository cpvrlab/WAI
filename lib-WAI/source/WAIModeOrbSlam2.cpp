#include <WAIModeOrbSlam2.h>

WAI::ModeOrbSlam2::ModeOrbSlam2(SensorCamera* camera, bool serial) : _camera(camera),
                                                                     _serial(serial)
{
    //load visual vocabulary for relocalization
    mpVocabulary = WAIOrbVocabulary::get();

    //instantiate and load slam map
    mpKeyFrameDatabase = new WAIKeyFrameDB(*mpVocabulary);

    _map = new WAIMap("Map");

    //setup file system and check for existing files
    //WAIMapStorage::init();
    //make new map
    //WAIMapStorage::newMap();

    if (_map->KeyFramesInMap())
        _initialized = true;
    else
        _initialized = false;

    int   nFeatures    = 1000;
    float fScaleFactor = 1.2;
    int   nLevels      = 8;
    int   fIniThFAST   = 20;
    int   fMinThFAST   = 7;

    //instantiate Orb extractor
    _extractor        = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    mpIniORBextractor = new ORB_SLAM2::ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    //instantiate local mapping
    mpLocalMapper = new ORB_SLAM2::LocalMapping(_map, 1, mpVocabulary);
    mpLoopCloser  = new ORB_SLAM2::LoopClosing(_map, mpKeyFrameDatabase, mpVocabulary, false, false);

    mpLocalMapper->SetLoopCloser(mpLoopCloser);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    if (!_serial)
    {
        mptLocalMapping = new std::thread(&LocalMapping::Run, mpLocalMapper);
        mptLoopClosing  = new std::thread(&LoopClosing::Run, mpLoopCloser);
    }
}

WAI::ModeOrbSlam2::~ModeOrbSlam2()
{
}

bool WAI::ModeOrbSlam2::getPose(M4x4* pose)
{
    bool result = 0;

    return result;
}
