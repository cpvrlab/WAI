#include <SLApplication.h>
#include <SLBox.h>
#include <SLLightSpot.h>
#include <SLCoordAxis.h>
#include <SLPoints.h>

#include <SLCVTrackedChessboard.h>
#include <SLKeyframeCamera.h>
#include <SLCVCapture.h>
#include <SLFileSystem.h>

#include <WAIMapStorage.h>

#include <AppWAIScene.h>
#include <AppDemoGui.h>
#include <AppDemoGuiTrackedMapping.h>
#include <AppDemoGuiMapStorage.h>
#include <AppDemoGuiInfosMapNodeTransform.h>
#include <AppDemoGuiInfosTracking.h>
#include <AppWAISceneView.h>

WAISceneView::WAISceneView(SLCVCalibration* calib, std::string externalDir, std::string dataRoot)
  : wai(dataRoot), _externalDir(externalDir)
{
    WAIMapStorage::init(externalDir);
}

void WAISceneView::setAppWAIScene(AppWAIScene* appWaiScene)
{
    _appWaiScene = appWaiScene;
}

//-----------------------------------------------------------------------------
static void onLoadCalibration(SLScene* s, SLSceneView* sv)
{
    s->name("Calibrate Main Cam.");
    SLApplication::activeCalib->clear();

    // Material
    //SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));

    // set the edge length of a chessboard square
    SLfloat e1 = 0.028f;
    SLfloat e3 = e1 * 3.0f;
    SLfloat e9 = e3 * 3.0f;

    // Create a scene group node
    SLNode* scene = new SLNode("scene node");

    // Create a camera node
    SLCamera* cam1 = new SLCamera();
    cam1->name("camera node");
    cam1->translation(0, 0, 5);
    cam1->lookAt(0, 0, 0);
    cam1->focalDist(5);
    cam1->clipFar(10);
    cam1->fov(SLApplication::activeCalib->cameraFovDeg());
    cam1->background().texture(s->videoTexture());
    cam1->setInitialState();
    scene->addChild(cam1);

    // Create a light source node
    SLLightSpot* light1 = new SLLightSpot(e1 * 0.5f);
    light1->translate(e9, e9, e9);
    light1->name("light node");
    scene->addChild(light1);

    // Create OpenCV Tracker for the box node
    s->trackers().push_back(new SLCVTrackedChessboard(cam1));

    // pass the scene group as root node
    s->root3D(scene);

    // Set active camera
    sv->camera(cam1);
    sv->doWaitOnIdle(false);

    sv->onInitialize();
    s->onAfterLoad();
}

static void onLoadTrackChessboard(SLScene* s, SLSceneView* sv)
{
    s->name("Track Chessboard (main cam.)");
    // Material
    SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));

    // set the edge length of a chessboard square
    SLfloat e1 = 0.028f;
    SLfloat e3 = e1 * 3.0f;
    SLfloat e9 = e3 * 3.0f;

    // Create a scene group node
    SLNode* scene = new SLNode("scene node");

    // Create a camera node
    SLCamera* cam1 = new SLCamera();
    cam1->name("camera node");
    cam1->translation(0, 0, 5);
    cam1->lookAt(0, 0, 0);
    cam1->focalDist(5);
    cam1->clipFar(10);
    cam1->fov(SLApplication::activeCalib->cameraFovDeg());
    cam1->background().texture(s->videoTexture());
    cam1->setInitialState();
    scene->addChild(cam1);

    // Create a light source node
    SLLightSpot* light1 = new SLLightSpot(e1 * 0.5f);
    light1->translate(e9, e9, e9);
    light1->name("light node");
    scene->addChild(light1);

    // Build mesh & node
    SLBox*  box     = new SLBox(0.0f, 0.0f, 0.0f, e3, e3, e3, "Box", yellow);
    SLNode* boxNode = new SLNode(box, "Box Node");
    boxNode->setDrawBitsRec(SL_DB_CULLOFF, true);
    SLNode* axisNode = new SLNode(new SLCoordAxis(), "Axis Node");
    axisNode->setDrawBitsRec(SL_DB_WIREMESH, false);
    axisNode->scale(e3);
    boxNode->addChild(axisNode);
    scene->addChild(boxNode);

    s->trackers().push_back(new SLCVTrackedChessboard(cam1));
    s->root3D(scene);

    // Set active camera
    sv->camera(cam1);
    sv->doWaitOnIdle(false);

    sv->onInitialize();
    s->onAfterLoad();
}

//-----------------------------------------------------------------------------
static void onLoadScenePoseEstimation(SLScene* s, SLSceneView* sv)
{
    WAISceneView* waiSceneView = (WAISceneView*)sv;
    AppWAIScene*  appWaiScene  = new AppWAIScene();
    waiSceneView->setAppWAIScene(appWaiScene);

    // Set scene name and info string
    s->name("Track Keyframe based Features");
    s->info("Example for loading an existing pose graph with map points.");

    //make some light
    SLLightSpot* light1 = new SLLightSpot(1, 1, 1, 0.3f);
    light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
    light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
    light1->specular(SLCol4f(1, 1, 1));
    light1->attenuation(1, 0, 0);

    //always equal for tracking
    //setup tracking camera
    appWaiScene->cameraNode = new SLCamera("Camera 1");
    appWaiScene->cameraNode->translation(0, 0, 0.1f);
    appWaiScene->cameraNode->lookAt(0, 0, 0);
    //for tracking we have to use the field of view from calibration
    appWaiScene->cameraNode->fov(SLApplication::activeCalib->cameraFovDeg());
    appWaiScene->cameraNode->clipNear(0.001f);
    appWaiScene->cameraNode->clipFar(1000000.0f); // Increase to infinity?
    appWaiScene->cameraNode->setInitialState();
    appWaiScene->cameraNode->background().texture(s->videoTexture());

    // Save no energy
    sv->doWaitOnIdle(false); //for constant video feed
    sv->camera(appWaiScene->cameraNode);

    //add yellow box and axis for augmentation
    SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
    SLfloat     l = 0.593f, b = 0.466f, h = 0.257f;
    SLBox*      box1     = new SLBox(0.0f, 0.0f, 0.0f, l, h, b, "Box 1", yellow);
    SLNode*     boxNode  = new SLNode(box1, "boxNode");
    SLNode*     axisNode = new SLNode(new SLCoordAxis(), "axis node");
    boxNode->addChild(axisNode);
    //boxNode->translation(0.0f, 0.0f, -2.0f);
    appWaiScene->mapNode->rotate(180, 1, 0, 0);
    appWaiScene->mapNode->addChild(appWaiScene->cameraNode);

    //setup scene
    SLNode* scene = new SLNode("scene");
    scene->addChild(light1);
    scene->addChild(boxNode);
    scene->addChild(appWaiScene->mapNode);

    s->root3D(scene);

    sv->onInitialize();
    s->onAfterLoad();

    WAI::CameraCalibration calibration = {SLApplication::activeCalib->fx(),
                                          SLApplication::activeCalib->fy(),
                                          SLApplication::activeCalib->cx(),
                                          SLApplication::activeCalib->cy(),
                                          SLApplication::activeCalib->k1(),
                                          SLApplication::activeCalib->k2(),
                                          SLApplication::activeCalib->p1(),
                                          SLApplication::activeCalib->p2()};

    waiSceneView->wai.activateSensor(WAI::SensorType_Camera, &calibration);

#if DATA_ORIENTED
    waiSceneView->setMode((WAI::ModeOrbSlam2DataOriented*)waiSceneView->wai.setMode(WAI::ModeType_ORB_SLAM2_DATA_ORIENTED));
#else
    waiSceneView->setMode((WAI::ModeOrbSlam2*)waiSceneView->wai.setMode(WAI::ModeType_ORB_SLAM2));
    auto trackedMapping = std::make_shared<AppDemoGuiTrackedMapping>("Tracked mapping", waiSceneView->getMode());
    AppDemoGui::addInfoDialog(trackedMapping);
    auto mapStorage = std::make_shared<AppDemoGuiMapStorage>("Map Storage",
                                                             waiSceneView->getMode(),
                                                             appWaiScene->mapNode,
                                                             waiSceneView->getExternalDir() + "slam-maps");
    AppDemoGui::addInfoDialog(mapStorage);
    auto mapTransform = std::make_shared<AppDemoGuiInfosMapNodeTransform>("Map transform",
                                                                          appWaiScene->mapNode,
                                                                          waiSceneView->getMode(),
                                                                          waiSceneView->getExternalDir());
    AppDemoGui::addInfoDialog(mapTransform);
    auto trackingInfos = std::make_shared<AppDemoGuiInfosTracking>("Tracking Infos",
                                                                   waiSceneView,
                                                                   waiSceneView->getMode());
    AppDemoGui::addInfoDialog(trackingInfos);
#endif
}

//-----------------------------------------------------------------------------
void onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid)
{
    SLApplication::sceneID = sid;
    s->init();

#if LIVE_VIDEO

    s->videoType(VT_MAIN);

    if (sid == SID_VideoCalibrateMain)
    {
        onLoadCalibration(s, sv);
    }
    else if (SLApplication::sceneID == SID_VideoTrackChessMain)
    {
        onLoadTrackChessboard(s, sv);
    }
    else
    {
        onLoadScenePoseEstimation(s, sv);
    }
#else
    SLstring calibFileName = "cam_calibration_main_huawei_p10_640_360.xml";
    SLApplication::calibVideoFile.load(SLFileSystem::externalDir() + "calibrations/", calibFileName, false, false);
    SLApplication::calibVideoFile.loadCalibParams();

    s->videoType(VT_FILE);
    SLCVCapture::videoFilename = "street3.mp4";
    SLCVCapture::videoLoops    = true;

    onLoadScenePoseEstimation(s, sv);
#endif
}
//-----------------------------------------------------------------------------
void WAISceneView::update()
{
    if (SLApplication::sceneID == SID_VideoCalibrateMain ||
        SLApplication::sceneID == SID_VideoTrackChessMain)
        return;

    cv::Mat pose          = cv::Mat(4, 4, CV_32F);
    bool    iKnowWhereIAm = wai.whereAmI(&pose);

    //update tracking infos visualization
    updateTrackingVisualization(iKnowWhereIAm);

    if (iKnowWhereIAm)
    {
        // update camera node position
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);

        Rwc = pose.rowRange(0, 3).colRange(0, 3).t();
        twc = -Rwc * pose.rowRange(0, 3).col(3);

        SLMat4f slPose((SLfloat)Rwc.at<float>(0, 0),
                       (SLfloat)Rwc.at<float>(0, 1),
                       (SLfloat)Rwc.at<float>(0, 2),
                       (SLfloat)twc.at<float>(0, 0),
                       (SLfloat)Rwc.at<float>(1, 0),
                       (SLfloat)Rwc.at<float>(1, 1),
                       (SLfloat)Rwc.at<float>(1, 2),
                       (SLfloat)twc.at<float>(1, 0),
                       (SLfloat)Rwc.at<float>(2, 0),
                       (SLfloat)Rwc.at<float>(2, 1),
                       (SLfloat)Rwc.at<float>(2, 2),
                       (SLfloat)twc.at<float>(2, 0),
                       0.0f,
                       0.0f,
                       0.0f,
                       1.0f);
        slPose.rotate(180, 1, 0, 0);

        _appWaiScene->cameraNode->om(slPose);
    }
}
//-----------------------------------------------------------------------------
void WAISceneView::updateTrackingVisualization(const bool iKnowWhereIAm)
{
    //update keypoints visualization (2d image points):
    //TODO: 2d visualization is still done in mode... do we want to keep it there?
    _mode->showKeyPoints(_showKeyPoints);
    _mode->showKeyPointsMatched(_showKeyPointsMatched);

    //update map point visualization:
    //if we still want to visualize the point cloud
    if (_showMapPC)
    {
        //get new points and add them
        renderMapPoints("MapPoints",
                        _mode->getMapPoints(),
                        _appWaiScene->mapPC,
                        _appWaiScene->mappointsMesh,
                        _appWaiScene->redMat);
    }
    else if (_appWaiScene->mappointsMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        _appWaiScene->mapPC->deleteMesh(_appWaiScene->mappointsMesh);
    }

    //update visualization of local map points:
    //only update them with a valid pose from WAI
    if (_showLocalMapPC && iKnowWhereIAm)
    {
        renderMapPoints("LocalMapPoints",
                        _mode->getLocalMapPoints(),
                        _appWaiScene->mapLocalPC,
                        _appWaiScene->mappointsLocalMesh,
                        _appWaiScene->blueMat);
    }
    else if (_appWaiScene->mappointsLocalMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        _appWaiScene->mapLocalPC->deleteMesh(_appWaiScene->mappointsLocalMesh);
    }

    //update visualization of matched map points
    //only update them with a valid pose from WAI
    if (_showMatchesPC && iKnowWhereIAm)
    {
        renderMapPoints("MatchedMapPoints",
                        _mode->getMatchedMapPoints(),
                        _appWaiScene->mapMatchedPC,
                        _appWaiScene->mappointsMatchedMesh,
                        _appWaiScene->greenMat);
    }
    else if (_appWaiScene->mappointsMatchedMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        _appWaiScene->mapMatchedPC->deleteMesh(_appWaiScene->mappointsMatchedMesh);
    }

    //update keyframe visualization
    _appWaiScene->keyFrameNode->deleteChildren();
    if (_showKeyFrames)
    {
        renderKeyframes();
    }

    //update pose graph visualization
    renderGraphs();
}
//-----------------------------------------------------------------------------
void WAISceneView::updateCamera(WAI::CameraData* cameraData)
{
    if (SLApplication::sceneID == SID_VideoCalibrateMain ||
        SLApplication::sceneID == SID_VideoTrackChessMain)
        return;
    wai.updateSensor(WAI::SensorType_Camera, cameraData);
}
//-----------------------------------------------------------------------------
void WAISceneView::updateMinNumOfCovisibles(int n)
{
    _minNumOfCovisibles = n;
}
//-----------------------------------------------------------------------------
void WAISceneView::renderMapPoints(std::string                      name,
                                   const std::vector<WAIMapPoint*>& pts,
                                   SLNode*&                         node,
                                   SLPoints*&                       mesh,
                                   SLMaterial*&                     material)
{
    //remove old mesh, if it exists
    if (mesh)
        node->deleteMesh(mesh);

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        std::vector<SLVec3f> points, normals;
        for (auto mapPt : pts)
        {
            WAI::V3 wP = mapPt->worldPosVec();
            WAI::V3 wN = mapPt->normalVec();
            points.push_back(SLVec3f(wP.x, wP.y, wP.z));
            normals.push_back(SLVec3f(wN.x, wN.y, wN.z));
        }

        mesh = new SLPoints(points, normals, name, material);
        node->addMesh(mesh);
        node->updateAABBRec();
    }
}
//-----------------------------------------------------------------------------
void WAISceneView::renderKeyframes()
{
#if DATA_ORIENTED
    std::vector<KeyFrame*> keyframes = _mode->getKeyFrames();

    // TODO(jan): delete keyframe textures
    for (KeyFrame* kf : keyframes)
    {
        SLCVCamera* cam = new SLCVCamera("KeyFrame " + std::to_string(kf->index));

        cv::Mat Twc = kf->wTc.clone();
        SLMat4f om;
        om.setMatrix(Twc.at<float>(0, 0),
                     -Twc.at<float>(0, 1),
                     -Twc.at<float>(0, 2),
                     Twc.at<float>(0, 3),
                     Twc.at<float>(1, 0),
                     -Twc.at<float>(1, 1),
                     -Twc.at<float>(1, 2),
                     Twc.at<float>(1, 3),
                     Twc.at<float>(2, 0),
                     -Twc.at<float>(2, 1),
                     -Twc.at<float>(2, 2),
                     Twc.at<float>(2, 3),
                     Twc.at<float>(3, 0),
                     -Twc.at<float>(3, 1),
                     -Twc.at<float>(3, 2),
                     Twc.at<float>(3, 3));

        cam->om(om);

        //calculate vertical field of view
        SLfloat fy     = SLApplication::activeCalib->fx();
        SLfloat cy     = SLApplication::activeCalib->cy();
        SLfloat fovDeg = 2 * (SLfloat)atan2(cy, fy) * SL_RAD2DEG;
        cam->fov(fovDeg);
        cam->focalDist(0.11);
        cam->clipNear(0.1);
        cam->clipFar(1000.0);
        _keyFrameNode->addChild(cam);
    }
#else
    std::vector<WAIKeyFrame*> keyframes = _mode->getKeyFrames();

    // TODO(jan): delete keyframe textures
    for (WAIKeyFrame* kf : keyframes)
    {
        if (kf->isBad())
            continue;

        SLKeyframeCamera* cam = new SLKeyframeCamera("KeyFrame " + std::to_string(kf->mnId));
        //set background
        if (kf->getTexturePath().size())
        {
            // TODO(jan): textures are saved in a global textures vector (scene->textures)
            // and should be deleted from there. Otherwise we have a yuuuuge memory leak.
#    if 0
        SLGLTexture* texture = new SLGLTexture(kf->getTexturePath());
        _kfTextures.push_back(texture);
        cam->background().texture(texture);
#    endif
        }

        cv::Mat Twc = kf->getObjectMatrix();
        SLMat4f om;
        om.setMatrix(Twc.at<float>(0, 0),
                     -Twc.at<float>(0, 1),
                     -Twc.at<float>(0, 2),
                     Twc.at<float>(0, 3),
                     Twc.at<float>(1, 0),
                     -Twc.at<float>(1, 1),
                     -Twc.at<float>(1, 2),
                     Twc.at<float>(1, 3),
                     Twc.at<float>(2, 0),
                     -Twc.at<float>(2, 1),
                     -Twc.at<float>(2, 2),
                     Twc.at<float>(2, 3),
                     Twc.at<float>(3, 0),
                     -Twc.at<float>(3, 1),
                     -Twc.at<float>(3, 2),
                     Twc.at<float>(3, 3));

        cam->om(om);

        //calculate vertical field of view
        SLfloat fy     = (SLfloat)kf->fy;
        SLfloat cy     = (SLfloat)kf->cy;
        SLfloat fovDeg = 2 * (SLfloat)atan2(cy, fy) * SL_RAD2DEG;
        cam->fov(fovDeg);
        cam->focalDist(0.11f);
        cam->clipNear(0.1f);
        cam->clipFar(1000.0f);
        _appWaiScene->keyFrameNode->addChild(cam);
    }
#endif
}
//-----------------------------------------------------------------------------
void WAISceneView::renderGraphs()
{
#ifndef DATA_ORIENTED
    std::vector<WAIKeyFrame*> kfs = _mode->getKeyFrames();

    SLVVec3f covisGraphPts;
    SLVVec3f spanningTreePts;
    SLVVec3f loopEdgesPts;
    for (auto* kf : kfs)
    {
        cv::Mat Ow = kf->GetCameraCenter();

        //covisibility graph
        const vector<WAIKeyFrame*> vCovKFs = kf->GetCovisiblesByWeight(_minNumOfCovisibles);
        if (!vCovKFs.empty())
        {
            for (vector<WAIKeyFrame*>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
            {
                if ((*vit)->mnId < kf->mnId)
                    continue;
                cv::Mat Ow2 = (*vit)->GetCameraCenter();

                covisGraphPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
                covisGraphPts.push_back(SLVec3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2)));
            }
        }

        //spanning tree
        WAIKeyFrame* parent = kf->GetParent();
        if (parent)
        {
            cv::Mat Owp = parent->GetCameraCenter();
            spanningTreePts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            spanningTreePts.push_back(SLVec3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2)));
        }

        //loop edges
        std::set<WAIKeyFrame*> loopKFs = kf->GetLoopEdges();
        for (set<WAIKeyFrame*>::iterator sit = loopKFs.begin(), send = loopKFs.end(); sit != send; sit++)
        {
            if ((*sit)->mnId < kf->mnId)
                continue;
            cv::Mat Owl = (*sit)->GetCameraCenter();
            loopEdgesPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            loopEdgesPts.push_back(SLVec3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2)));
        }
    }

    if (_appSceneView->covisibilityGraphMesh)
        _appSceneView->covisibilityGraph->deleteMesh(_appSceneView->covisibilityGraphMesh);

    if (covisGraphPts.size())
    {
        _appSceneView->covisibilityGraphMesh = new SLPolyline(covisGraphPts, false, "CovisibilityGraph", _appSceneView->covisibilityGraphMat);
        _appSceneView->covisibilityGraph->addMesh(_appSceneView->covisibilityGraphMesh);
        _appSceneView->covisibilityGraph->updateAABBRec();
    }

    if (_appSceneView->spanningTreeMesh)
        _appSceneView->spanningTree->deleteMesh(_appSceneView->spanningTreeMesh);

    if (spanningTreePts.size())
        ;
    {
        _appSceneView->spanningTreeMesh = new SLPolyline(spanningTreePts, false, "SpanningTree", _appSceneView->spanningTreeMat);
        _appSceneView->spanningTree->addMesh(_appSceneView->spanningTreeMesh);
        _appSceneView->spanningTree->updateAABBRec();
    }

    if (_appSceneView->loopEdgesMesh)
        _appSceneView->loopEdges->deleteMesh(_appSceneView->loopEdgesMesh);

    if (loopEdgesPts.size())
    {
        _appSceneView->loopEdgesMesh = new SLPolyline(loopEdgesPts, false, "LoopEdges", _appSceneView->loopEdgesMat);
        _appSceneView->loopEdges->addMesh(_appSceneView->loopEdgesMesh);
        _appSceneView->loopEdges->updateAABBRec();
    }

#else
    std::vector<WAIKeyFrame*> kfs = _mode->getKeyFrames();

    SLVVec3f covisGraphPts;
    SLVVec3f spanningTreePts;
    SLVVec3f loopEdgesPts;
    for (auto* kf : kfs)
    {
        cv::Mat Ow = kf->GetCameraCenter();

        //covisibility graph
        const vector<WAIKeyFrame*> vCovKFs = kf->GetCovisiblesByWeight(_minNumOfCovisibles);
        if (!vCovKFs.empty())
        {
            for (vector<WAIKeyFrame*>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
            {
                if ((*vit)->mnId < kf->mnId)
                    continue;
                cv::Mat Ow2 = (*vit)->GetCameraCenter();

                covisGraphPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
                covisGraphPts.push_back(SLVec3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2)));
            }
        }

        //spanning tree
        WAIKeyFrame* parent = kf->GetParent();
        if (parent)
        {
            cv::Mat Owp = parent->GetCameraCenter();
            spanningTreePts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            spanningTreePts.push_back(SLVec3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2)));
        }

        //loop edges
        std::set<WAIKeyFrame*> loopKFs = kf->GetLoopEdges();
        for (set<WAIKeyFrame*>::iterator sit = loopKFs.begin(), send = loopKFs.end(); sit != send; sit++)
        {
            if ((*sit)->mnId < kf->mnId)
                continue;
            cv::Mat Owl = (*sit)->GetCameraCenter();
            loopEdgesPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            loopEdgesPts.push_back(SLVec3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2)));
        }
    }

    if (_appWaiScene->covisibilityGraphMesh)
        _appWaiScene->covisibilityGraph->deleteMesh(_appWaiScene->covisibilityGraphMesh);

    if (covisGraphPts.size() && _showCovisibilityGraph)
    {
        _appWaiScene->covisibilityGraphMesh = new SLPolyline(covisGraphPts, false, "CovisibilityGraph", _appWaiScene->covisibilityGraphMat);
        _appWaiScene->covisibilityGraph->addMesh(_appWaiScene->covisibilityGraphMesh);
        _appWaiScene->covisibilityGraph->updateAABBRec();
    }

    if (_appWaiScene->spanningTreeMesh)
        _appWaiScene->spanningTree->deleteMesh(_appWaiScene->spanningTreeMesh);

    if (spanningTreePts.size() && _showSpanningTree)
    {
        _appWaiScene->spanningTreeMesh = new SLPolyline(spanningTreePts, false, "SpanningTree", _appWaiScene->spanningTreeMat);
        _appWaiScene->spanningTree->addMesh(_appWaiScene->spanningTreeMesh);
        _appWaiScene->spanningTree->updateAABBRec();
    }

    if (_appWaiScene->loopEdgesMesh)
        _appWaiScene->loopEdges->deleteMesh(_appWaiScene->loopEdgesMesh);

    if (loopEdgesPts.size() && _showLoopEdges)
    {
        _appWaiScene->loopEdgesMesh = new SLPolyline(loopEdgesPts, false, "LoopEdges", _appWaiScene->loopEdgesMat);
        _appWaiScene->loopEdges->addMesh(_appWaiScene->loopEdgesMesh);
        _appWaiScene->loopEdges->updateAABBRec();
    }
#endif
}
