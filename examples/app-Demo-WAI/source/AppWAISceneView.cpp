#include <SLApplication.h>
#include <SLBox.h>
#include <SLLightSpot.h>
#include <SLCoordAxis.h>
#include <SLPoints.h>

#include <SLCVTrackedChessboard.h>
#include <SLKeyframeCamera.h>
#include <SLCVCapture.h>
#include <Utils.h>

#include <WAIMapStorage.h>

#include <AppWAIScene.h>
#include <AppWAISingleton.h>
#include <AppDemoGui.h>
#include <AppDemoGuiTrackedMapping.h>
#include <AppDemoGuiMapStorage.h>
#include <AppDemoGuiInfosMapNodeTransform.h>
#include <AppDemoGuiInfosTracking.h>
#include <AppWAISceneView.h>

WAISceneView::WAISceneView(std::string externalDir, std::string dataRoot)
  : _externalDir(externalDir)
{
    WAIMapStorage::init(externalDir);
}

//-----------------------------------------------------------------------------
static void onLoadScenePoseEstimation(SLScene* s, SLSceneView* sv)
{
    WAISceneView* waiSceneView = (WAISceneView*)sv;

    AppWAIScene* waiScene = AppWAISingleton::instance()->appWaiScene;
    waiScene->rebuild();

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
    //waiScene->cameraNode = new SLCamera("This is a Camera");
    waiScene->cameraNode->translation(0, 0, 0.1f);
    waiScene->cameraNode->lookAt(0, 0, 0);
    //for tracking we have to use the field of view from calibration
    waiScene->cameraNode->fov(AppWAISingleton::instance()->wc->calcCameraFOV());
    waiScene->cameraNode->clipNear(0.001f);
    waiScene->cameraNode->clipFar(1000000.0f); // Increase to infinity?
    waiScene->cameraNode->setInitialState();
    waiScene->cameraNode->background().texture(s->videoTexture());

    // Save no energy
    sv->doWaitOnIdle(false); //for constant video feed
    sv->camera(waiScene->cameraNode);

    //add yellow box and axis for augmentation
    SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
    SLfloat     l = 0.071f, b = 0.071f, h = 0.071f;
    SLBox*      box1    = new SLBox(-l / 2, -h / 2, 0.0f, l / 2, h / 2, b, "Box 1", yellow);
    SLNode*     boxNode = new SLNode(box1, "boxNode");
    //SLNode*     axisNode = new SLNode(new SLCoordAxis(), "axis node");
    //boxNode->addChild(axisNode);
    //boxNode->translation(0.0f, 0.0f, -2.0f);
    waiScene->mapNode->rotate(180, 1, 0, 0);
    waiScene->mapNode->addChild(waiScene->cameraNode);

    //setup scene
    SLNode* scene = new SLNode("scene");
    scene->addChild(light1);
    scene->addChild(boxNode);
    //scene->addChild(waiScene->cameraNode);
    scene->addChild(waiScene->mapNode);

    s->root3D(scene);

    sv->onInitialize();
    s->onAfterLoad();

    WAI::WAI* wai = AppWAISingleton::instance()->wai;

    waiSceneView->setMode((WAI::ModeOrbSlam2*)wai->setMode(WAI::ModeType_ORB_SLAM2));
    auto trackedMapping = std::make_shared<AppDemoGuiTrackedMapping>("Tracked mapping", waiSceneView->getMode());
    AppDemoGui::addInfoDialog(trackedMapping);
    auto mapStorage = std::make_shared<AppDemoGuiMapStorage>("Map Storage",
                                                             waiSceneView->getMode(),
                                                             waiScene->mapNode,
                                                             waiSceneView->getExternalDir() + "slam-maps");
    AppDemoGui::addInfoDialog(mapStorage);
    auto mapTransform = std::make_shared<AppDemoGuiInfosMapNodeTransform>("Map transform",
                                                                          waiScene->mapNode,
                                                                          waiSceneView->getMode(),
                                                                          waiSceneView->getExternalDir());
    AppDemoGui::addInfoDialog(mapTransform);
    auto trackingInfos = std::make_shared<AppDemoGuiInfosTracking>("Tracking Infos",
                                                                   waiSceneView,
                                                                   waiSceneView->getMode());
    AppDemoGui::addInfoDialog(trackingInfos);
}

//-----------------------------------------------------------------------------
void onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid)
{
    s->init();

#if LIVE_VIDEO
    SLstring calibFileName = "cam_calibration_main.xml";
    SLApplication::calibVideoFile.load(WAIMapStorage::externalDir() + "config/", calibFileName, false, false);
    SLApplication::calibVideoFile.loadCalibParams();

    s->videoType(VT_MAIN);
    onLoadScenePoseEstimation(s, sv);
#else
    //SLstring calibFileName = "cam_calibration_main_huawei_p10_640_360.xml";
    SLstring calibFileName = "cam_calibration_main.xml";
    SLApplication::calibVideoFile.load(WAIMapStorage::externalDir() + "calibrations/", calibFileName, false, false);
    SLApplication::calibVideoFile.loadCalibParams();

    s->videoType(VT_FILE);
    SLCVCapture::videoFilename = "initialization_test.webm";
    SLCVCapture::videoLoops    = true;

    onLoadScenePoseEstimation(s, sv);
#endif
}
//-----------------------------------------------------------------------------
void WAISceneView::update()
{
    AppWAIScene* waiScene      = AppWAISingleton::instance()->appWaiScene;
    WAI::WAI*    wai           = AppWAISingleton::instance()->wai;
    cv::Mat      Twc           = cv::Mat(4, 4, CV_32F);
    bool         iKnowWhereIAm = wai->whereAmI(&Twc);

    //update tracking infos visualization
    updateTrackingVisualization(iKnowWhereIAm);

    if (iKnowWhereIAm)
    {
        SLMat4f om;
        om.setMatrix(Twc.at<float>(0, 0),
                     Twc.at<float>(0, 1),
                     Twc.at<float>(0, 2),
                     Twc.at<float>(0, 3),
                     Twc.at<float>(1, 0),
                     Twc.at<float>(1, 1),
                     Twc.at<float>(1, 2),
                     Twc.at<float>(1, 3),
                     Twc.at<float>(2, 0),
                     Twc.at<float>(2, 1),
                     Twc.at<float>(2, 2),
                     Twc.at<float>(2, 3),
                     Twc.at<float>(3, 0),
                     Twc.at<float>(3, 1),
                     Twc.at<float>(3, 2),
                     Twc.at<float>(3, 3));
        om.rotate(180, 1, 0, 0);

        waiScene->cameraNode->om(om);
    }
}
//-----------------------------------------------------------------------------
void WAISceneView::updateTrackingVisualization(const bool iKnowWhereIAm)
{
    if (_mode->isMarkerCorrected() && iKnowWhereIAm)
    {
        cv::Mat mapTransform = _mode->getMarkerCorrectionTransformation();
        SLMat4f om;
        om.setMatrix(mapTransform.at<float>(0, 0),
                     mapTransform.at<float>(0, 1),
                     mapTransform.at<float>(0, 2),
                     mapTransform.at<float>(0, 3),
                     mapTransform.at<float>(1, 0),
                     mapTransform.at<float>(1, 1),
                     mapTransform.at<float>(1, 2),
                     mapTransform.at<float>(1, 3),
                     mapTransform.at<float>(2, 0),
                     mapTransform.at<float>(2, 1),
                     mapTransform.at<float>(2, 2),
                     mapTransform.at<float>(2, 3),
                     mapTransform.at<float>(3, 0),
                     mapTransform.at<float>(3, 1),
                     mapTransform.at<float>(3, 2),
                     mapTransform.at<float>(3, 3));

        AppWAIScene* waiScene = AppWAISingleton::instance()->appWaiScene;
        waiScene->mapNode->om(om);

        _mapNodeTransformed = true;
    }

    AppWAIScene* waiScene = AppWAISingleton::instance()->appWaiScene;
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
                        waiScene->mapPC,
                        waiScene->mappointsMesh,
                        waiScene->redMat);
    }
    else if (waiScene->mappointsMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        waiScene->mapPC->deleteMesh(waiScene->mappointsMesh);
    }

    //update visualization of local map points:
    //only update them with a valid pose from WAI
    if (_showLocalMapPC && iKnowWhereIAm)
    {
        renderMapPoints("LocalMapPoints",
                        _mode->getLocalMapPoints(),
                        waiScene->mapLocalPC,
                        waiScene->mappointsLocalMesh,
                        waiScene->blueMat);
    }
    else if (waiScene->mappointsLocalMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        waiScene->mapLocalPC->deleteMesh(waiScene->mappointsLocalMesh);
    }

    //update visualization of matched map points
    //only update them with a valid pose from WAI
    if (_showMatchesPC && iKnowWhereIAm)
    {
        renderMapPoints("MatchedMapPoints",
                        _mode->getMatchedMapPoints(),
                        waiScene->mapMatchedPC,
                        waiScene->mappointsMatchedMesh,
                        waiScene->greenMat);
    }
    else if (waiScene->mappointsMatchedMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        waiScene->mapMatchedPC->deleteMesh(waiScene->mappointsMatchedMesh);
    }

    //update keyframe visualization
    waiScene->keyFrameNode->deleteChildren();
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
    if (AppWAISingleton::instance()->videoWriter.isOpened())
    {
        AppWAISingleton::instance()->videoWriter.write(*cameraData->imageRGB);
    }

    WAI::WAI* wai = AppWAISingleton::instance()->wai;
    wai->updateSensor(WAI::SensorType_Camera, cameraData);

    if (AppWAISingleton::instance()->videoWriterInfo.isOpened())
    {
        AppWAISingleton::instance()->videoWriterInfo.write(*cameraData->imageRGB);
    }
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
#if 0
        SLGLTexture* texture = new SLGLTexture(kf->getTexturePath());
        _kfTextures.push_back(texture);
        cam->background().texture(texture);
#endif
        }

        cv::Mat Twc = kf->getObjectMatrix();

        SLMat4f om;
        om.setMatrix(Twc.at<float>(0, 0),
                     Twc.at<float>(0, 1),
                     Twc.at<float>(0, 2),
                     Twc.at<float>(0, 3),
                     Twc.at<float>(1, 0),
                     Twc.at<float>(1, 1),
                     Twc.at<float>(1, 2),
                     Twc.at<float>(1, 3),
                     Twc.at<float>(2, 0),
                     Twc.at<float>(2, 1),
                     Twc.at<float>(2, 2),
                     Twc.at<float>(2, 3),
                     Twc.at<float>(3, 0),
                     Twc.at<float>(3, 1),
                     Twc.at<float>(3, 2),
                     Twc.at<float>(3, 3));
        om.rotate(180, 1, 0, 0);

        cam->om(om);

        //calculate vertical field of view
        SLfloat fy     = (SLfloat)kf->fy;
        SLfloat cy     = (SLfloat)kf->cy;
        SLfloat fovDeg = 2 * (SLfloat)atan2(cy, fy) * SL_RAD2DEG;
        cam->fov(fovDeg);
        cam->focalDist(0.11f);
        cam->clipNear(0.1f);
        cam->clipFar(1000.0f);

        AppWAIScene* waiScene = AppWAISingleton::instance()->appWaiScene;
        waiScene->keyFrameNode->addChild(cam);
    }
}
//-----------------------------------------------------------------------------
void WAISceneView::renderGraphs()
{
    AppWAIScene*              waiScene = AppWAISingleton::instance()->appWaiScene;
    std::vector<WAIKeyFrame*> kfs      = _mode->getKeyFrames();

    SLVVec3f covisGraphPts;
    SLVVec3f spanningTreePts;
    SLVVec3f loopEdgesPts;
    for (auto* kf : kfs)
    {
        cv::Mat Ow = kf->GetCameraCenter();

        //covisibility graph
        const std::vector<WAIKeyFrame*> vCovKFs = kf->GetBestCovisibilityKeyFrames(_minNumOfCovisibles);

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

    if (waiScene->covisibilityGraphMesh)
        waiScene->covisibilityGraph->deleteMesh(waiScene->covisibilityGraphMesh);

    if (covisGraphPts.size() && _showCovisibilityGraph)
    {
        waiScene->covisibilityGraphMesh = new SLPolyline(covisGraphPts, false, "CovisibilityGraph", waiScene->covisibilityGraphMat);
        waiScene->covisibilityGraph->addMesh(waiScene->covisibilityGraphMesh);
        waiScene->covisibilityGraph->updateAABBRec();
    }

    if (waiScene->spanningTreeMesh)
        waiScene->spanningTree->deleteMesh(waiScene->spanningTreeMesh);

    if (spanningTreePts.size() && _showSpanningTree)
    {
        waiScene->spanningTreeMesh = new SLPolyline(spanningTreePts, false, "SpanningTree", waiScene->spanningTreeMat);
        waiScene->spanningTree->addMesh(waiScene->spanningTreeMesh);
        waiScene->spanningTree->updateAABBRec();
    }

    if (waiScene->loopEdgesMesh)
        waiScene->loopEdges->deleteMesh(waiScene->loopEdgesMesh);

    if (loopEdgesPts.size() && _showLoopEdges)
    {
        waiScene->loopEdgesMesh = new SLPolyline(loopEdgesPts, false, "LoopEdges", waiScene->loopEdgesMat);
        waiScene->loopEdges->addMesh(waiScene->loopEdgesMesh);
        waiScene->loopEdges->updateAABBRec();
    }
}
