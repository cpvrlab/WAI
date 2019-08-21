#include <atomic>
#include <SLApplication.h>
#include <SLInterface.h>
#include <SLCVTrackedChessboard.h>
#include <SLKeyframeCamera.h>
#include <SLCVCapture.h>
#include <Utils.h>

#include <WAIMapStorage.h>

#include <WAICalibration.h>
#include <AppWAIScene.h>
#include <AppDemoGui.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiTrackedMapping.h>
#include <AppDemoGuiMapStorage.h>
#include <AppDemoGuiVideoStorage.h>
#include <AppDemoGuiInfosMapNodeTransform.h>
#include <AppDemoGuiInfosTracking.h>
#include <AppWAI.h>

int   WAIApp::minNumOfCovisibles = 50;
float WAIApp::meanReprojectionError;
bool  WAIApp::showKeyPoints         = true;
bool  WAIApp::showKeyPointsMatched  = true;
bool  WAIApp::showMapPC             = true;
bool  WAIApp::showLocalMapPC        = true;
bool  WAIApp::showMatchesPC         = true;
bool  WAIApp::showKeyFrames         = true;
bool  WAIApp::renderKfBackground    = true;
bool  WAIApp::allowKfsAsActiveCam   = true;
bool  WAIApp::showCovisibilityGraph = true;
bool  WAIApp::showSpanningTree      = true;
bool  WAIApp::showLoopEdges         = true;

AppWAIScene*       WAIApp::waiScene;
std::string        WAIApp::rootPath;
WAI::WAI*          WAIApp::wai;
WAICalibration*    WAIApp::wc;
int                WAIApp::scrWidth;
int                WAIApp::scrHeight;
cv::VideoWriter*   WAIApp::videoWriter;
cv::VideoWriter*   WAIApp::videoWriterInfo;
WAI::ModeOrbSlam2* WAIApp::mode        = nullptr;
std::string        WAIApp::externalDir = "";
bool               WAIApp::loaded      = false;

int WAIApp::load(int width, int height, float scr2fbX, float scr2fbY, int dpi,
                  std::string extDir, std::string dataRoot, std::string slRoot)
{
    externalDir = extDir;
    rootPath = dataRoot;
    WAIMapStorage::init(externalDir);

    wai             = new WAI::WAI(dataRoot);
    wc              = new WAICalibration();
    waiScene        = new AppWAIScene();
    videoWriter     = new cv::VideoWriter();
    videoWriterInfo = new cv::VideoWriter();

    wc->changeImageSize(width, height);
    wc->loadFromFile(rootPath + "/calibrations/cam_calibration_main.xml");
    WAI::CameraCalibration calibration = wc->getCameraCalibration();
    wai->activateSensor(WAI::SensorType_Camera, &calibration);

    SLVstring empty;
    slCreateAppAndScene(empty,
                        slRoot + "/shaders/",
                        slRoot + "/models/",
                        slRoot + "/images/textures/",
                        slRoot + "/videos/",
                        slRoot + "/images/fonts/",
                        slRoot + "/calibrations/",
                        extDir,
                        "AppDemoGLFW",
                        (void*)WAIApp::onLoadWAISceneView);

    // This load the GUI configs that are locally stored
    AppDemoGui::loadConfig(dpi);

    int svIndex = slCreateSceneView((int)(width * scr2fbX),
                                (int)(height * scr2fbY),
                                dpi,
                                (SLSceneID)0,
                                nullptr,
                                nullptr,
                                nullptr,
                                (void*)AppDemoGui::build);

    loaded = true;
    return svIndex;
}

//-----------------------------------------------------------------------------
void WAIApp::onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid)
{
    s->init();
    s->videoType(VT_MAIN);
    waiScene->rebuild();

    // Set scene name and info string
    s->name("Track Keyframe based Features");
    s->info("Example for loading an existing pose graph with map points.");

    // Save no energy
    sv->doWaitOnIdle(false); //for constant video feed
    sv->camera(waiScene->cameraNode);

    waiScene->cameraNode->background().texture(s->videoTexture());
    waiScene->cameraNode->fov(wc->calcCameraFOV());

    s->root3D(waiScene->rootNode);

    sv->onInitialize();
    s->onAfterLoad();

    mode = ((WAI::ModeOrbSlam2*)wai->setMode(WAI::ModeType_ORB_SLAM2));


    auto trackedMapping = std::make_shared<AppDemoGuiTrackedMapping>("Tracked mapping", mode);
    AppDemoGui::addInfoDialog(trackedMapping);

    auto mapStorage = std::make_shared<AppDemoGuiMapStorage>("Map Storage",
                                                             mode,
                                                             waiScene->mapNode,
                                                             externalDir + "slam-maps");
    AppDemoGui::addInfoDialog(mapStorage);


    auto mapTransform = std::make_shared<AppDemoGuiInfosMapNodeTransform>("Map transform",
                                                                          waiScene->mapNode,
                                                                          mode,
                                                                          externalDir);
    AppDemoGui::addInfoDialog(mapTransform);


    auto trackingInfos = std::make_shared<AppDemoGuiInfosTracking>("Tracking Infos",
                                                                   mode);
    AppDemoGui::addInfoDialog(trackingInfos);


    auto videoStorageGUI = std::make_shared<AppDemoGuiVideoStorage>("VideoStorage", rootPath + "/data/videos/",
                                                                    videoWriter, videoWriterInfo);
    AppDemoGui::addInfoDialog(videoStorageGUI);
}

//-----------------------------------------------------------------------------
void WAIApp::update()
{
    if(!loaded)
        return;

    cv::Mat      pose          = cv::Mat(4, 4, CV_32F);
    bool         iKnowWhereIAm = wai->whereAmI(&pose);

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

        waiScene->cameraNode->om(slPose);
    }
}
//-----------------------------------------------------------------------------
void WAIApp::updateTrackingVisualization(const bool iKnowWhereIAm)
{
    //update keypoints visualization (2d image points):
    //TODO: 2d visualization is still done in mode... do we want to keep it there?
    mode->showKeyPoints(showKeyPoints);
    mode->showKeyPointsMatched(showKeyPointsMatched);

    //update map point visualization:
    //if we still want to visualize the point cloud
    if (showMapPC)
    {
        //get new points and add them
        renderMapPoints("MapPoints",
                        mode->getMapPoints(),
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
    if (showLocalMapPC && iKnowWhereIAm)
    {
        renderMapPoints("LocalMapPoints",
                        mode->getLocalMapPoints(),
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
    if (showMatchesPC && iKnowWhereIAm)
    {
        renderMapPoints("MatchedMapPoints",
                        mode->getMatchedMapPoints(),
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
    if (showKeyFrames)
    {
        renderKeyframes();
    }

    //update pose graph visualization
    renderGraphs();
}
//-----------------------------------------------------------------------------
void WAIApp::updateCamera(WAI::CameraData* cameraData)
{
    if(!loaded)
        return;

    if (videoWriter->isOpened()) {
        videoWriter->write(*cameraData->imageRGB);
    }

    wai->updateSensor(WAI::SensorType_Camera, cameraData);

    if (videoWriterInfo->isOpened()) {
        videoWriterInfo->write(*cameraData->imageRGB);
    }
}
//-----------------------------------------------------------------------------
void WAIApp::updateMinNumOfCovisibles(int n)
{
    minNumOfCovisibles = n;
}
//-----------------------------------------------------------------------------
void WAIApp::renderMapPoints(std::string                      name,
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
void WAIApp::renderKeyframes()
{
    std::vector<WAIKeyFrame*> keyframes = mode->getKeyFrames();

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

        waiScene->keyFrameNode->addChild(cam);
    }
}
//-----------------------------------------------------------------------------
void WAIApp::renderGraphs()
{
    std::vector<WAIKeyFrame*> kfs      = mode->getKeyFrames();

    SLVVec3f covisGraphPts;
    SLVVec3f spanningTreePts;
    SLVVec3f loopEdgesPts;
    for (auto* kf : kfs)
    {
        cv::Mat Ow = kf->GetCameraCenter();

        //covisibility graph
        const std::vector<WAIKeyFrame*> vCovKFs = kf->GetBestCovisibilityKeyFrames(minNumOfCovisibles);

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

    if (covisGraphPts.size() && showCovisibilityGraph)
    {
        waiScene->covisibilityGraphMesh = new SLPolyline(covisGraphPts, false, "CovisibilityGraph", waiScene->covisibilityGraphMat);
        waiScene->covisibilityGraph->addMesh(waiScene->covisibilityGraphMesh);
        waiScene->covisibilityGraph->updateAABBRec();
    }

    if (waiScene->spanningTreeMesh)
        waiScene->spanningTree->deleteMesh(waiScene->spanningTreeMesh);

    if (spanningTreePts.size() && showSpanningTree)
    {
        waiScene->spanningTreeMesh = new SLPolyline(spanningTreePts, false, "SpanningTree", waiScene->spanningTreeMat);
        waiScene->spanningTree->addMesh(waiScene->spanningTreeMesh);
        waiScene->spanningTree->updateAABBRec();
    }

    if (waiScene->loopEdgesMesh)
        waiScene->loopEdges->deleteMesh(waiScene->loopEdgesMesh);

    if (loopEdgesPts.size() && showLoopEdges)
    {
        waiScene->loopEdgesMesh = new SLPolyline(loopEdgesPts, false, "LoopEdges", waiScene->loopEdgesMat);
        waiScene->loopEdges->addMesh(waiScene->loopEdgesMesh);
        waiScene->loopEdges->updateAABBRec();
    }
}
