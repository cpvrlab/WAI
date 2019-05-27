#include <WAI.h>
#include <WAIMath.h>
#include <Utils.h>

static WAI::WAI           wai("");
static WAI::ModeOrbSlam2* mode      = nullptr;
static std::string        mapPrefix = "slam-map-";
static std::string        mapsDir;
static int                nextId;

struct WAIMapPointCoordinate
{
    float x, y, z;
};

extern "C" {
WAI_API void wai_setDataRoot(const char* dataRoot)
{
    WAI_LOG("dataroot set to %s", dataRoot);
    wai.setDataRoot(dataRoot);
    mapsDir = std::string(dataRoot) + "maps/";
}

WAI_API void wai_setMode(WAI::ModeType modeType)
{
    //WAI_LOG("setMode called");
    mode = (WAI::ModeOrbSlam2*)wai.setMode(modeType);
}

WAI_API void wai_getMapPoints(WAIMapPointCoordinate** mapPointCoordinatePtr,
                              int*                    mapPointCount)
{
    if (!mode)
    {
        WAI_LOG("mode not set. Call wai_setMode first.");
        return;
    }

    std::vector<WAIMapPoint*> mapPoints       = mode->getMapPoints();
    *mapPointCoordinatePtr                    = (WAIMapPointCoordinate*)malloc(mapPoints.size() * sizeof(WAIMapPointCoordinate));
    WAIMapPointCoordinate* mapPointCoordinate = *mapPointCoordinatePtr;

    int count = 0;

    for (WAIMapPoint* mapPoint : mapPoints)
    {
        if (!mapPoint->isBad())
        {
            *mapPointCoordinate = {
              mapPoint->worldPosVec().x,
              mapPoint->worldPosVec().y,
              mapPoint->worldPosVec().z};

            mapPointCoordinate++;
            count++;
        }
    }

    *mapPointCount = count;
}

WAI_API void wai_getKeyFrames(WAIMapPointCoordinate** keyFrameCoordinatePtr,
                              int*                    keyFrameCount)
{
    if (!mode)
    {
        WAI_LOG("mode not set. Call wai_setMode first.");
        return;
    }

    std::vector<WAIKeyFrame*> keyFrames       = mode->getKeyFrames();
    *keyFrameCoordinatePtr                    = (WAIMapPointCoordinate*)malloc(keyFrames.size() * sizeof(WAIMapPointCoordinate));
    WAIMapPointCoordinate* keyFrameCoordinate = *keyFrameCoordinatePtr;

    int count = 0;

    for (WAIKeyFrame* keyFrame : keyFrames)
    {
        if (!keyFrame->isBad())
        {
            cv::Mat worldPos = keyFrame->GetCameraCenter();

            *keyFrameCoordinate = {
              worldPos.at<float>(0, 0),
              worldPos.at<float>(1, 0),
              worldPos.at<float>(2, 0)};

            keyFrameCoordinate++;
            count++;
        }
    }

    *keyFrameCount = count;
}

WAI_API void wai_releaseMapPoints(WAIMapPointCoordinate** mapPointCoordinatePtr)
{
    delete *mapPointCoordinatePtr;
}

WAI_API void wai_releaseKeyFrames(WAIMapPointCoordinate** keyFrameCoordinatePtr)
{
    delete *keyFrameCoordinatePtr;
}

WAI_API void wai_activateSensor(WAI::SensorType sensorType, void* sensorInfo)
{
    //WAI_LOG("activateSensor called");
    wai.activateSensor(sensorType, sensorInfo);
}

WAI_API void wai_updateCameraWithKnownPose(WAI::CameraFrame* frameRGB,
                                           WAI::CameraFrame* frameGray,
                                           WAI::M4x4         knownPose)
{
    //WAI_LOG("updateCameraWithKnownPose called");
    cv::Mat cvFrameRGB  = cv::Mat(frameRGB->height,
                                 frameRGB->width,
                                 CV_8UC3,
                                 frameRGB->memory,
                                 frameRGB->pitch);
    cv::Mat cvFrameGray = cv::Mat(frameGray->height,
                                  frameGray->width,
                                  CV_8UC1,
                                  frameGray->memory,
                                  frameGray->pitch);

    cv::Mat cvKnownPose = cv::Mat(4, 4, CV_32F);
    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 4; x++)
        {
            cvKnownPose.at<float>(x, y) = knownPose.e[y][x];
        }
    }

    WAI::CameraData sensorData = {&cvFrameGray,
                                  &cvFrameRGB,
                                  true,
                                  &cvKnownPose};

    wai.updateSensor(WAI::SensorType_Camera, &sensorData);
}

WAI_API void wai_updateCamera(WAI::CameraFrame* frameRGB,
                              WAI::CameraFrame* frameGray)
{
    //WAI_LOG("updateCamera called");
    cv::Mat cvFrameRGB  = cv::Mat(frameRGB->height,
                                 frameRGB->width,
                                 CV_8UC3,
                                 frameRGB->memory,
                                 frameRGB->pitch);
    cv::Mat cvFrameGray = cv::Mat(frameGray->height,
                                  frameGray->width,
                                  CV_8UC1,
                                  frameGray->memory,
                                  frameGray->pitch);

    WAI::CameraData sensorData = {&cvFrameGray,
                                  &cvFrameRGB,
                                  false};

    wai.updateSensor(WAI::SensorType_Camera, &sensorData);
}

WAI_API bool wai_whereAmI(WAI::M4x4* pose)
{
    //WAI_LOG("whereAmI called");
    bool result = 0;

    cv::Mat cvPose = cv::Mat(4, 4, CV_32F);
    result         = wai.whereAmI(&cvPose);

    if (result)
    {
        //WAI_LOG("WAI knows where I am");
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                pose->e[x][y] = cvPose.at<float>(x, y);
            }
        }
    }

    return result;
}

WAI_API int wai_getState(char* buffer, int size)
{
    //WAI_LOG("getState called");
    int result = 0;

    if (mode)
    {
        std::string state = mode->getPrintableState();

        if ((state.size() + 1) < size)
        {
            size = state.size() + 1;
        }

        result        = size;
        char*       c = buffer;
        const char* s = state.c_str();

        strncpy(c, s, size);
    }

    return result;
}

WAI_API void wai_registerDebugCallback(DebugLogCallback callback)
{
    registerDebugCallback(callback);
}

WAI_API bool32 wai_saveMap(bool32 saveAndLoadImages)
{
    //save keyframes (without graph/neigbourhood information)
    std::vector<WAIKeyFrame*> kfs = mode->getKeyFrames();
    if (kfs.size())
    {
        std::string mapDir   = mapsDir + mapPrefix + std::to_string(nextId) + "/";
        std::string filename = mapDir + mapPrefix + std::to_string(nextId) + ".json";
        std::string imgDir   = mapDir + "imgs";

        WAI_LOG("Saving map to %s\n", mapDir.c_str());

        if (!Utils::dirExists(mapDir))
        {
            if (!Utils::makeDir(mapDir))
            {
                WAI_LOG("Failed to create directory at %s\n", mapDir.c_str());
            }
        }
        else
        {
            if (Utils::fileExists(filename))
            {
                Utils::deleteFile(filename);
            }
        }

        if (!Utils::dirExists(imgDir))
        {
            Utils::makeDir(imgDir);
        }
        else
        {
            std::vector<std::string> content = Utils::getFileNamesInDir(imgDir);
            for (std::string path : content)
            {
                Utils::deleteFile(path);
            }
        }

        cv::FileStorage fs(filename, cv::FileStorage::WRITE);

        cv::Mat cvOm = cv::Mat::eye(4, 4, CV_32F);
#if 0
        SLMat4f om           = _mapNode->om();
        cvOm.at<float>(0, 0) = om.m(0);
        cvOm.at<float>(0, 1) = -om.m(1);
        cvOm.at<float>(0, 2) = -om.m(2);
        cvOm.at<float>(0, 3) = om.m(12);
        cvOm.at<float>(1, 0) = om.m(4);
        cvOm.at<float>(1, 1) = -om.m(5);
        cvOm.at<float>(1, 2) = -om.m(6);
        cvOm.at<float>(1, 3) = -om.m(13);
        cvOm.at<float>(2, 0) = om.m(8);
        cvOm.at<float>(2, 1) = -om.m(9);
        cvOm.at<float>(2, 2) = -om.m(10);
        cvOm.at<float>(2, 3) = -om.m(14);
        cvOm.at<float>(3, 3) = 1.0f;
#endif
        fs << "mapNodeOm" << cvOm;

        //start sequence keyframes
        fs << "KeyFrames"
           << "[";
        for (int i = 0; i < kfs.size(); ++i)
        {
            WAIKeyFrame* kf = kfs[i];
            if (kf->isBad())
                continue;

            fs << "{"; //new map keyFrame
                       //add id
            fs << "id" << (int)kf->mnId;
            if (kf->mnId != 0) //kf with id 0 has no parent
                fs << "parentId" << (int)kf->GetParent()->mnId;
            else
                fs << "parentId" << -1;
            //loop edges: we store the id of the connected kf
            auto loopEdges = kf->GetLoopEdges();
            if (loopEdges.size())
            {
                std::vector<int> loopEdgeIds;
                for (auto loopEdgeKf : loopEdges)
                {
                    loopEdgeIds.push_back(loopEdgeKf->mnId);
                }
                fs << "loopEdges" << loopEdgeIds;
            }

            // world w.r.t camera
            fs << "Tcw" << kf->GetPose();
            fs << "featureDescriptors" << kf->mDescriptors;
            fs << "keyPtsUndist" << kf->mvKeysUn;

            //scale factor
            fs << "scaleFactor" << kf->mfScaleFactor;
            //number of pyriamid scale levels
            fs << "nScaleLevels" << kf->mnScaleLevels;
            //fs << "fx" << kf->fx;
            //fs << "fy" << kf->fy;
            //fs << "cx" << kf->cx;
            //fs << "cy" << kf->cy;
            fs << "K" << kf->mK;

            //debug print
            //std::cout << "fx" << kf->fx << std::endl;
            //std::cout << "fy" << kf->fy << std::endl;
            //std::cout << "cx" << kf->cx << std::endl;
            //std::cout << "cy" << kf->cy << std::endl;
            //std::cout << "K" << kf->mK << std::endl;

            fs << "nMinX" << kf->mnMinX;
            fs << "nMinY" << kf->mnMinY;
            fs << "nMaxX" << kf->mnMaxX;
            fs << "nMaxY" << kf->mnMaxY;

            fs << "}"; //close map

            //save the original frame image for this keyframe
            if (saveAndLoadImages)
            {
                cv::Mat imgColor;
                if (!kf->imgGray.empty())
                {
                    std::stringstream ss;
                    ss << imgDir << "kf" << (int)kf->mnId << ".jpg";

                    cv::cvtColor(kf->imgGray, imgColor, cv::COLOR_GRAY2BGR);
                    cv::imwrite(ss.str(), imgColor);

                    //if this kf was never loaded, we still have to set the texture path
                    kf->setTexturePath(ss.str());
                }
            }
        }
        fs << "]"; //close sequence keyframes

        std::vector<WAIMapPoint*> mpts = mode->getMapPoints();
        //start map points sequence
        fs << "MapPoints"
           << "[";
        for (int i = 0; i < mpts.size(); ++i)
        {
            WAIMapPoint* mpt = mpts[i];
            if (mpt->isBad())
                continue;

            fs << "{"; //new map for MapPoint
                       //add id
            fs << "id" << (int)mpt->mnId;
            //add position
            fs << "mWorldPos" << mpt->GetWorldPos();
            //save keyframe observations
            auto        observations = mpt->GetObservations();
            vector<int> observingKfIds;
            vector<int> corrKpIndices; //corresponding keypoint indices in observing keyframe
            for (auto it : observations)
            {
                if (!it.first->isBad())
                {
                    observingKfIds.push_back(it.first->mnId);
                    corrKpIndices.push_back(it.second);
                }
            }
            fs << "observingKfIds" << observingKfIds;
            fs << "corrKpIndices" << corrKpIndices;
            //(we calculate mean descriptor and mean deviation after loading)

            //reference key frame (I think this is the keyframe from which this
            //map point was generated -> first reference?)
            fs << "refKfId" << (int)mpt->refKf()->mnId;

            fs << "}"; //close map
        }
        fs << "]";

        // explicit close
        fs.release();

        nextId++;

        return true;
    }

    return false;
}

WAI_API bool32 wai_loadMap(int selectedMapId, bool32 saveAndLoadImages)
{
    std::string idString = std::to_string(selectedMapId);
    std::string mapName  = mapPrefix + idString;

    //load selected map
    cv::Mat cvOm = cv::Mat(4, 4, CV_32F);

    mode->requestStateIdle();
    while (!mode->hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    mode->reset();

    std::string mapDir   = mapsDir + mapName + "/";
    std::string filename = mapDir + mapName + ".json";
    std::string imgDir   = mapDir + "imgs";

    WAI_LOG("Loading map from %s\n", mapDir.c_str());

    //check if dir and file exist
    if (Utils::dirExists(mapDir))
    {
        WAI_LOG("Map dir exists\n");
        if (Utils::fileExists(filename))
        {
            WAI_LOG("Map file exists\n");

            cv::FileStorage fs(filename, cv::FileStorage::READ);

#if 0
    // TODO(jan): This must happen in the visualization software
                fs["mapNodeOm"] >> cvOm;
                SLMat4f om;
                om.setMatrix(cvOm.at<float>(0, 0),
                             -cvOm.at<float>(0, 1),
                             -cvOm.at<float>(0, 2),
                             cvOm.at<float>(0, 3),
                             cvOm.at<float>(1, 0),
                             -cvOm.at<float>(1, 1),
                             -cvOm.at<float>(1, 2),
                             cvOm.at<float>(1, 3),
                             cvOm.at<float>(2, 0),
                             -cvOm.at<float>(2, 1),
                             -cvOm.at<float>(2, 2),
                             cvOm.at<float>(2, 3),
                             cvOm.at<float>(3, 0),
                             -cvOm.at<float>(3, 1),
                             -cvOm.at<float>(3, 2),
                             cvOm.at<float>(3, 3));
                _mapNode->om(om);
#endif

            //mapping of keyframe pointer by their id (used during map points loading)
            map<int, WAIKeyFrame*>    kfsMap;
            std::vector<WAIKeyFrame*> keyFrames;
            std::vector<WAIMapPoint*> mapPoints;
            int                       numLoopClosings = 0;

            {
                cv::FileNode n = fs["KeyFrames"];
                if (n.type() != cv::FileNode::SEQ)
                {
                    WAI_LOG("strings is not a sequence! FAIL\n");
                }

                //the id of the parent is mapped to the kf id because we can assign it not before all keyframes are loaded
                std::map<int, int> parentIdMap;
                //vector of keyframe ids of connected loop edge candidates mapped to kf id that they are connected to
                std::map<int, std::vector<int>> loopEdgesMap;
                //reserve space in kfs
                for (auto it = n.begin(); it != n.end(); ++it)
                {
                    int id = (*it)["id"];
                    //load parent id
                    if (!(*it)["parentId"].empty())
                    {
                        int parentId    = (*it)["parentId"];
                        parentIdMap[id] = parentId;
                    }
                    //load ids of connected loop edge candidates
                    if (!(*it)["loopEdges"].empty() && (*it)["loopEdges"].isSeq())
                    {
                        cv::FileNode     les = (*it)["loopEdges"];
                        std::vector<int> loopEdges;
                        for (auto itLes = les.begin(); itLes != les.end(); ++itLes)
                        {
                            loopEdges.push_back((int)*itLes);
                        }
                        loopEdgesMap[id] = loopEdges;
                    }
                    // Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
                    // world w.r.t. camera pose -> wTc
                    cv::Mat Tcw; //has to be here!
                    (*it)["Tcw"] >> Tcw;

                    cv::Mat featureDescriptors; //has to be here!
                    (*it)["featureDescriptors"] >> featureDescriptors;

                    //load undistorted keypoints in frame
                    //todo: braucht man diese wirklich oder kann man das umgehen, indem zus�tzliche daten im MapPoint abgelegt werden (z.B. octave/level siehe UpdateNormalAndDepth)
                    std::vector<cv::KeyPoint> keyPtsUndist;
                    (*it)["keyPtsUndist"] >> keyPtsUndist;

                    //ORB extractor information
                    float scaleFactor;
                    (*it)["scaleFactor"] >> scaleFactor;
                    //number of pyriamid scale levels
                    int nScaleLevels = -1;
                    (*it)["nScaleLevels"] >> nScaleLevels;
                    //calculation of scaleFactors , levelsigma2, invScaleFactors and invLevelSigma2
                    //(copied from ORBextractor ctor)

                    //vectors for precalculation of scalefactors
                    std::vector<float> vScaleFactor;
                    std::vector<float> vInvScaleFactor;
                    std::vector<float> vLevelSigma2;
                    std::vector<float> vInvLevelSigma2;
                    vScaleFactor.clear();
                    vLevelSigma2.clear();
                    vScaleFactor.resize(nScaleLevels);
                    vLevelSigma2.resize(nScaleLevels);
                    vScaleFactor[0] = 1.0f;
                    vLevelSigma2[0] = 1.0f;
                    for (int i = 1; i < nScaleLevels; i++)
                    {
                        vScaleFactor[i] = vScaleFactor[i - 1] * scaleFactor;
                        vLevelSigma2[i] = vScaleFactor[i] * vScaleFactor[i];
                    }

                    vInvScaleFactor.resize(nScaleLevels);
                    vInvLevelSigma2.resize(nScaleLevels);
                    for (int i = 0; i < nScaleLevels; i++)
                    {
                        vInvScaleFactor[i] = 1.0f / vScaleFactor[i];
                        vInvLevelSigma2[i] = 1.0f / vLevelSigma2[i];
                    }

                    //calibration information
                    //load camera matrix
                    cv::Mat K;
                    (*it)["K"] >> K;
                    float fx, fy, cx, cy;
                    fx = K.at<float>(0, 0);
                    fy = K.at<float>(1, 1);
                    cx = K.at<float>(0, 2);
                    cy = K.at<float>(1, 2);

                    //image bounds
                    float nMinX, nMinY, nMaxX, nMaxY;
                    (*it)["nMinX"] >> nMinX;
                    (*it)["nMinY"] >> nMinY;
                    (*it)["nMaxX"] >> nMaxX;
                    (*it)["nMaxY"] >> nMaxY;

                    WAIKeyFrame* newKf = new WAIKeyFrame(Tcw,
                                                         id,
                                                         fx,
                                                         fy,
                                                         cx,
                                                         cy,
                                                         keyPtsUndist.size(),
                                                         keyPtsUndist,
                                                         featureDescriptors,
                                                         WAIOrbVocabulary::get(),
                                                         nScaleLevels,
                                                         scaleFactor,
                                                         vScaleFactor,
                                                         vLevelSigma2,
                                                         vInvLevelSigma2,
                                                         nMinX,
                                                         nMinY,
                                                         nMaxX,
                                                         nMaxY,
                                                         K,
                                                         mode->getKfDB(),
                                                         mode->getMap());

                    if (saveAndLoadImages)
                    {
                        stringstream ss;
                        ss << imgDir << "kf" << id << ".jpg";
                        //newKf->imgGray = kfImg;
                        if (Utils::fileExists(ss.str()))
                        {
                            newKf->setTexturePath(ss.str());
                            cv::Mat imgColor = cv::imread(ss.str());
                            cv::cvtColor(imgColor, newKf->imgGray, cv::COLOR_BGR2GRAY);
                        }
                    }

#if 1
                    keyFrames.push_back(newKf);

                    //pointer goes out of scope und wird invalid!!!!!!
                    //map pointer by id for look-up
                    kfsMap[newKf->mnId] = newKf;
#else
                    //kfs.push_back(newKf);
                    _map->AddKeyFrame(newKf);

                    //Update keyframe database:
                    //add to keyframe database
                    _kfDB->add(newKf);

#endif
                }

                //set parent keyframe pointers into keyframes
                for (WAIKeyFrame* kf : keyFrames)
                {
                    if (kf->mnId != 0)
                    {
                        auto itParentId = parentIdMap.find(kf->mnId);
                        if (itParentId != parentIdMap.end())
                        {
                            int  parentId   = itParentId->second;
                            auto itParentKf = kfsMap.find(parentId);
                            if (itParentKf != kfsMap.end())
                                kf->ChangeParent(itParentKf->second);
                            else
                                cerr << "[WAIMapIO] loadKeyFrames: Parent does not exist! FAIL" << endl;
                        }
                        else
                            cerr << "[WAIMapIO] loadKeyFrames: Parent does not exist! FAIL" << endl;
                    }
                }

                int numberOfLoopClosings = 0;
                //set loop edge pointer into keyframes
                for (WAIKeyFrame* kf : keyFrames)
                {
                    auto it = loopEdgesMap.find(kf->mnId);
                    if (it != loopEdgesMap.end())
                    {
                        const auto& loopEdgeIds = it->second;
                        for (int loopKfId : loopEdgeIds)
                        {
                            auto loopKfIt = kfsMap.find(loopKfId);
                            if (loopKfIt != kfsMap.end())
                            {
                                kf->AddLoopEdge(loopKfIt->second);
                                numberOfLoopClosings++;
                            }
                            else
                                cerr << "[WAIMapIO] loadKeyFrames: Loop keyframe id does not exist! FAIL" << endl;
                        }
                    }
                }
                //there is a loop edge in the keyframe and the matched keyframe -> division by 2
                numLoopClosings = numberOfLoopClosings / 2;
            }

            {
                cv::FileNode n = fs["MapPoints"];
                if (n.type() != cv::FileNode::SEQ)
                {
                    cerr << "strings is not a sequence! FAIL" << endl;
                }

                //reserve space in mapPts
                //mapPts.reserve(n.size());
                //read and add map points
                for (auto it = n.begin(); it != n.end(); ++it)
                {
                    //newPt->id( (int)(*it)["id"]);
                    int id = (int)(*it)["id"];

                    cv::Mat mWorldPos; //has to be here!
                    (*it)["mWorldPos"] >> mWorldPos;

                    WAIMapPoint* newPt = new WAIMapPoint(id, mWorldPos, mode->getMap());
                    //get observing keyframes
                    vector<int> observingKfIds;
                    (*it)["observingKfIds"] >> observingKfIds;
                    //get corresponding keypoint indices in observing keyframe
                    vector<int> corrKpIndices;
                    (*it)["corrKpIndices"] >> corrKpIndices;

#if 1
                    mapPoints.push_back(newPt);
#else
                    _map->AddMapPoint(newPt);
#endif

                    //get reference keyframe id
                    int refKfId = (int)(*it)["refKfId"];

                    //find and add pointers of observing keyframes to map point
                    {
                        WAIMapPoint* mapPt = newPt;
                        for (int i = 0; i < observingKfIds.size(); ++i)
                        {
                            const int kfId = observingKfIds[i];
                            if (kfsMap.find(kfId) != kfsMap.end())
                            {
                                WAIKeyFrame* kf = kfsMap[kfId];
                                kf->AddMapPoint(mapPt, corrKpIndices[i]);
                                mapPt->AddObservation(kf, corrKpIndices[i]);
                            }
                            else
                            {
                                WAI_LOG("keyframe with id %i not found\n", i);
                            }
                        }

                        //todo: is the reference keyframe only a currently valid variable or has every keyframe a reference keyframe?? Is it necessary for tracking?
                        //map reference key frame pointer
                        if (kfsMap.find(refKfId) != kfsMap.end())
                        {
                            mapPt->refKf(kfsMap[refKfId]);
                        }
                        else
                        {
                            WAI_LOG("no reference keyframe found!");
                            if (observingKfIds.size())
                            {
                                //we use the first of the observing keyframes
                                int kfId = observingKfIds[0];
                                if (kfsMap.find(kfId) != kfsMap.end())
                                    mapPt->refKf(kfsMap[kfId]);
                            }
                            else
                            {
                                int stop = 0;
                            }
                        }
                    }
                }
            }

            //update the covisibility graph, when all keyframes and mappoints are loaded
            WAIKeyFrame* firstKF           = nullptr;
            bool         buildSpanningTree = false;
            for (WAIKeyFrame* kf : keyFrames)
            {
                // Update links in the Covisibility Graph, do not build the spanning tree yet
                kf->UpdateConnections(false);
                if (kf->mnId == 0)
                {
                    firstKF = kf;
                }
                else if (kf->GetParent() == NULL)
                {
                    buildSpanningTree = true;
                }
            }

            wai_assert(firstKF && "Could not find keyframe with id 0\n");

            // Build spanning tree if keyframes have no parents (legacy support)
            if (buildSpanningTree)
            {
                //QueueElem: <unconnected_kf, graph_kf, weight>
                using QueueElem                 = std::tuple<WAIKeyFrame*, WAIKeyFrame*, int>;
                auto                   cmpQueue = [](const QueueElem& left, const QueueElem& right) { return (std::get<2>(left) < std::get<2>(right)); };
                auto                   cmpMap   = [](const pair<WAIKeyFrame*, int>& left, const pair<WAIKeyFrame*, int>& right) { return left.second < right.second; };
                std::set<WAIKeyFrame*> graph;
                std::set<WAIKeyFrame*> unconKfs;
                for (auto& kf : keyFrames)
                    unconKfs.insert(kf);

                //pick first kf
                graph.insert(firstKF);
                unconKfs.erase(firstKF);

                while (unconKfs.size())
                {
                    std::priority_queue<QueueElem, std::vector<QueueElem>, decltype(cmpQueue)> q(cmpQueue);
                    //update queue with keyframes with neighbous in the graph
                    for (auto& unconKf : unconKfs)
                    {
                        const std::map<WAIKeyFrame*, int>& weights = unconKf->GetConnectedKfWeights();
                        for (auto& graphKf : graph)
                        {
                            auto it = weights.find(graphKf);
                            if (it != weights.end())
                            {
                                QueueElem newElem = std::make_tuple(unconKf, it->first, it->second);
                                q.push(newElem);
                            }
                        }
                    }
                    //extract keyframe with shortest connection
                    QueueElem topElem = q.top();
                    //remove it from unconKfs and add it to graph
                    WAIKeyFrame* newGraphKf = std::get<0>(topElem);
                    unconKfs.erase(newGraphKf);
                    newGraphKf->ChangeParent(std::get<1>(topElem));
                    std::cout << "Added kf " << newGraphKf->mnId << " with parent " << std::get<1>(topElem)->mnId << std::endl;
                    //update parent
                    graph.insert(newGraphKf);
                }
            }

            //compute resulting values for map points
            for (WAIMapPoint*& mp : mapPoints)
            {
                //mean viewing direction and depth
                mp->UpdateNormalAndDepth();
                mp->ComputeDistinctiveDescriptors();
            }

            mode->loadMapData(keyFrames, mapPoints, numLoopClosings);
            mode->resume();

            return true;
        }
    }

    return false;
}
}
