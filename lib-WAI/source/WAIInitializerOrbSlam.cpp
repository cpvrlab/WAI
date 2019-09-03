#include <mutex>

#include <WAIInitializerOrbSlam.h>

#include <OrbSlam/Optimizer.h>
#include <OrbSlam/ORBmatcher.h>

WAI::WAIInitializerOrbSlam::WAIInitializerOrbSlam(WAIMap*                   map,
                                                  WAIKeyFrameDB*            keyFrameDB,
                                                  KPextractor*              kpExtractor,
                                                  ORB_SLAM2::ORBVocabulary* orbVoc,
                                                  SensorCamera*             camera,
                                                  bool                      serial,
                                                  bool                      retainImg)
  : _map(map),
    _kpExtractor(kpExtractor),
    _orbVoc(orbVoc),
    _camera(camera),
    _serial(serial),
    _retainImg(retainImg),
    _keyFrameDB(keyFrameDB)
{
}

WAI::InitializationResult WAI::WAIInitializerOrbSlam::initialize()
{
    //1. if there are more than 100 keypoints in the current frame, the Initializer is instantiated
    //2. if there are less than 100 keypoints in the next frame, the Initializer is deinstantiated again
    //3. else if there are more than 100 keypoints we try to match the keypoints in the current with the initial frame
    //4. if we found less than 100 matches between the current and the initial keypoints, the Initializer is deinstantiated
    //5. else we try to initializer: that means a homograhy and a fundamental matrix are calculated in parallel and 3D points are triangulated initially
    //6. if the initialization (by homograhy or fundamental matrix) was successful an inital map is created:
    //  - two keyframes are generated from the initial and the current frame and added to keyframe database and map
    //  - a mappoint is instantiated from the triangulated 3D points and all necessary members are calculated (distinctive descriptor, depth and normal, add observation reference of keyframes)
    //  - a global bundle adjustment is applied
    //  - the two new keyframes are added to the local mapper and the local mapper is started twice
    //  - the tracking state is changed to TRACKING/INITIALIZED

    InitializationResult result = {};

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(_map->mMutexMapUpdate, std::defer_lock);
    if (!_serial)
    {
        lock.lock();
    }

    cv::Mat cameraMat     = _camera->getCameraMatrix();
    cv::Mat distortionMat = _camera->getDistortionMatrix();
    _currentFrame         = WAIFrame(_camera->getImageGray(),
                             0.0,
                             _kpExtractor,
                             cameraMat,
                             distortionMat,
                             _orbVoc,
                             _retainImg);

#if 0 // TODO(dgj1): reactivate for marker correction
    //cv::Mat markerCorrectedPose;
    std::vector<int> markerMatchesCurrentFrame;
    if (_markerCorrected)
    {
#    if 0
        if (!findChessboardPose(markerCorrectedPose))
        {
            return;
        }
#    endif
        ORBmatcher               matcher(0.9, true);
        std::vector<cv::Point2f> prevMatched(_markerFrame.mvKeysUn.size());
        for (size_t i = 0; i < _markerFrame.mvKeysUn.size(); i++)
            prevMatched[i] = _markerFrame.mvKeysUn[i].pt;

        std::vector<int> markerMatchesToCurrentFrame;
        int              matchCount = matcher.SearchForInitialization(_markerFrame, _currentFrame, prevMatched, markerMatchesToCurrentFrame, 100);
        WAI_LOG("matchCount: %i", matchCount);

        if (matchCount > 100)
        {
            std::vector<cv::KeyPoint> matches;
            for (int i = 0; i < markerMatchesToCurrentFrame.size(); i++)
            {
                if (markerMatchesToCurrentFrame[i] >= 0)
                {
                    matches.push_back(_currentFrame.mvKeys[markerMatchesToCurrentFrame[i]]);
                    markerMatchesCurrentFrame.push_back(i);
                }
            }

            _currentFrame = WAIFrame(_camera->getImageGray(),
                                     mpIniORBextractor,
                                     cameraMat,
                                     distortionMat,
                                     matches,
                                     mpVocabulary,
                                     _retainImg);
        }
        else
        {
            return;
        }
    }
#endif

    if (!_initializer)
    {
        // Set Reference Frame
        if (_currentFrame.mvKeys.size() > 100)
        {
            _initialFrame = WAIFrame(_currentFrame);
            //mLastFrame    = WAIFrame(_currentFrame); // TODO(dgj1): do we need the lastFrame at all?
            _prevMatched.resize(_currentFrame.mvKeysUn.size());
            //ghm1: we store the undistorted keypoints of the initial frame in an extra vector
            //todo: why not using _initialFrame.mvKeysUn????
            for (size_t i = 0; i < _currentFrame.mvKeysUn.size(); i++)
                _prevMatched[i] = _currentFrame.mvKeysUn[i].pt;

            // TODO(jan): is this necessary?
            if (_initializer)
                delete _initializer;

            _initializer = new ORB_SLAM2::Initializer(_currentFrame, 1.0, 200);
            //ghm1: clear _iniMatches. it contains the index of the matched keypoint in the current frame
            fill(_iniMatches.begin(), _iniMatches.end(), -1);

#if 0 // TODO(dgj1): reactivate for marker correction \
      //_initialFrameChessboardPose = markerCorrectedPose;
            if (_markerCorrected)
            {
                _initialFrameToMarkerMatches = markerMatchesCurrentFrame;
            }
#endif

            return;
        }
    }
    else
    {
        // Try to initialize
        if ((int)_currentFrame.mvKeys.size() <= 100)
        {
            delete _initializer;
            _initializer = static_cast<Initializer*>(NULL);
            fill(_iniMatches.begin(), _iniMatches.end(), -1);
            return;
        }

        int matchCount = 0;
#if 0 // TODO(dgj1): reactivate for marker correction
        if (_markerCorrected)
        {
            _iniMatches = std::vector<int>(_initialFrame.mvKeysUn.size(), -1);
            for (int i = 0; i < _initialFrameToMarkerMatches.size(); i++)
            {
                for (int j = 0; j < markerMatchesCurrentFrame.size(); j++)
                {
                    if (_initialFrameToMarkerMatches[i] == markerMatchesCurrentFrame[j])
                    {
                        _iniMatches[i] = j;
                        matchCount++;
                    }
                }
            }

            for (int i = 0; i < _iniMatches.size(); i++)
            {
                if (_iniMatches[i] >= 0)
                {
                    _prevMatched[i] = _currentFrame.mvKeysUn[_iniMatches[i]].pt;
                }
            }
        }
        else
        {
#endif
        // Find correspondences
        ORBmatcher matcher(0.9, true);
        matchCount = matcher.SearchForInitialization(_initialFrame, _currentFrame, _prevMatched, _iniMatches, 10);
#if 0 // TODO(dgj1): reactivate for marker correction
    }
#endif

        WAI_LOG("matchCount for initialization: %i", matchCount);

        // Check if there are enough correspondences
        if (matchCount < 100)
        {
            delete _initializer;
            _initializer = static_cast<Initializer*>(NULL);
            return;
        }

        for (unsigned int i = 0; i < _initialFrame.mvKeys.size(); i++)
        {
            cv::rectangle(_camera->getImageRGB(),
                          _initialFrame.mvKeys[i].pt,
                          cv::Point(_initialFrame.mvKeys[i].pt.x + 3, _initialFrame.mvKeys[i].pt.y + 3),
                          cv::Scalar(0, 0, 255));
        }

        //ghm1: decorate image with tracked matches
        for (unsigned int i = 0; i < _iniMatches.size(); i++)
        {
            if (_iniMatches[i] >= 0)
            {
                cv::line(_camera->getImageRGB(),
                         _initialFrame.mvKeys[i].pt,
                         _currentFrame.mvKeys[_iniMatches[i]].pt,
                         cv::Scalar(0, 255, 0));
            }
        }

        cv::Mat      Rcw;               // Current Camera Rotation
        cv::Mat      tcw;               // Current Camera Translation
        vector<bool> matchTriangulated; // Triangulated Correspondences (_iniMatches)

        if (_initializer->Initialize(_currentFrame, _iniMatches, Rcw, tcw, _3DpointsIniMatched, matchTriangulated))
        {
            for (size_t i = 0, iend = _iniMatches.size(); i < iend; i++)
            {
                if (_iniMatches[i] >= 0 && !matchTriangulated[i])
                {
                    _iniMatches[i] = -1;
                    matchCount--;
                }
            }

            WAI_LOG("%i triangulated", matchCount);

            // Set Frame Poses
            _initialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            _currentFrame.SetPose(Tcw);

            bool mapInitializedSuccessfully = createInitialMap(&result.kfIni, &result.kfCur);
            if (mapInitializedSuccessfully)
            {
                //mark tracking as initialized
                result.success      = true;
                result.currentFrame = &_currentFrame;
#if 0
                if (_markerCorrected)
                {
                    cv::Mat t1, t2;
                    t1 = _initialFrameChessboardPose.rowRange(0, 3).col(3);
                    t2 = markerCorrectedPose.rowRange(0, 3).col(3);

                    cv::Mat t1w, t2w;
                    t1w = mInitialFrame.GetCameraCenter();
                    t2w = mCurrentFrame.GetCameraCenter();

                    float distCorrected   = cv::norm(t1, t2);
                    float distUncorrected = cv::norm(t1w, t2w);

                    float scaleFactor = distUncorrected / distCorrected;
                    //float scaleFactor = 1.0f;

                    cv::Mat scaledMarkerCorrection               = _initialFrameChessboardPose.clone();
                    scaledMarkerCorrection.col(3).rowRange(0, 3) = scaledMarkerCorrection.col(3).rowRange(0, 3) * scaleFactor;
                    _markerCorrectionTransformation              = scaledMarkerCorrection;
                }
#endif
            }
        }
    }

    return result;
}

bool WAI::WAIInitializerOrbSlam::createInitialMap(WAIKeyFrame** kfIniPtr,
                                                  WAIKeyFrame** kfCurPtr)
{
    // Create KeyFrames
    WAIKeyFrame* kfIni = new WAIKeyFrame(_initialFrame, _map, _keyFrameDB);
    WAIKeyFrame* kfCur = new WAIKeyFrame(_currentFrame, _map, _keyFrameDB);

    WAI_LOG("pKFini num keypoints: %i", _initialFrame.N);
    WAI_LOG("pKFcur num keypoints: %i", _currentFrame.N);

    kfIni->ComputeBoW(_orbVoc);
    kfCur->ComputeBoW(_orbVoc);

    // Insert KFs in the map
    _map->AddKeyFrame(kfIni);
    _map->AddKeyFrame(kfCur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < _iniMatches.size(); i++)
    {
        if (_iniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(_3DpointsIniMatched[i]);

        WAIMapPoint* pMP = new WAIMapPoint(worldPos, kfCur, _map);

        kfIni->AddMapPoint(pMP, i);
        kfCur->AddMapPoint(pMP, _iniMatches[i]);

        pMP->AddObservation(kfIni, i);
        pMP->AddObservation(kfCur, _iniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        _currentFrame.mvpMapPoints[_iniMatches[i]] = pMP;
        _currentFrame.mvbOutlier[_iniMatches[i]]   = false;

        //Add to Map
        _map->AddMapPoint(pMP);
    }

    // Update Connections
    kfIni->UpdateConnections();
    kfCur->UpdateConnections();

    // Bundle Adjustment
    WAI_LOG("New Map created with %i points", _map->MapPointsInMap());

    Optimizer::GlobalBundleAdjustemnt(_map, 20);

    // Set median depth to 1
    float medianDepth    = kfIni->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth < 0 || kfCur->TrackedMapPoints(1) < 10) //80)
    {
        WAI_LOG("Wrong initialization, reseting...");
        reset();
        return false;
    }

#if 1 // TODO(dgj1): REACTIVATE THIS FOR REGULAR INITIALIZATION
    // Scale initial baseline
    cv::Mat Tc2w               = kfCur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    kfCur->SetPose(Tc2w);

    // Scale points
    vector<WAIMapPoint*> vpAllMapPoints = kfIni->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            WAIMapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }
#endif

    _currentFrame.SetPose(kfCur->GetPose());

#if 0 // TODO(dgj1): do this in mode orbslam 2
    localMapper->InsertKeyFrame(pKFini);
    localMapper->InsertKeyFrame(pKFcur);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame   = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = _map->GetAllMapPoints();

    mpReferenceKF               = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = WAIFrame(mCurrentFrame);

    _map->SetReferenceMapPoints(mvpLocalMapPoints);

    _map->mvpKeyFrameOrigins.push_back(pKFini);

    //ghm1: run local mapping once
    if (_serial)
    {
        mpLocalMapper->RunOnce();
        mpLocalMapper->RunOnce();
    }

    // Bundle Adjustment
    WAI_LOG("Number of Map points after local mapping: %i", _map->MapPointsInMap());

    std::cout << pKFini->getObjectMatrix() << std::endl;
    std::cout << pKFcur->getObjectMatrix() << std::endl;

    //ghm1: add keyframe to scene graph. this position is wrong after bundle adjustment!
    //set map dirty, the map will be updated in next decoration
    _mapHasChanged = true;
#endif

    *kfIniPtr = kfIni;
    *kfCurPtr = kfCur;

    return true;
}

void WAI::WAIInitializerOrbSlam::reset()
{
    if (_initializer)
    {
        delete _initializer;
        _initializer = nullptr;
    }
}
