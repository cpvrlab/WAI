#include <atomic>
#include <SLApplication.h>
#include <SLInterface.h>
#include <SLKeyframeCamera.h>
#include <CVCapture.h>
#include <Utils.h>

#include <WAIMapStorage.h>

#include <WAICalibration.h>
#include <AppWAIScene.h>
#include <AppDemoGui.h>
#include <AppDemoGuiMenu.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiTrackedMapping.h>
#include <AppDemoGuiAbout.h>
#include <AppDemoGuiStatsTiming.h>
#include <AppDemoGuiStatsDebugTiming.h>
#include <AppDemoGuiStatsVideo.h>
#include <AppDemoGuiInfosScene.h>
#include <AppDemoGuiTransform.h>
#include <AppDemoGuiInfosFrameworks.h>
#include <AppDemoGuiInfosSensors.h>
#include <AppDemoGuiUIPrefs.h>
#include <AppDemoGuiProperties.h>
#include <AppDemoGuiMapStorage.h>
#include <AppDemoGuiVideoStorage.h>
#include <AppDemoGuiInfosMapNodeTransform.h>
#include <AppDemoGuiInfosTracking.h>
#include <AppWAI.h>
#include <AppDirectories.h>

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

AppDemoGuiAbout*   WAIApp::aboutDial = nullptr;
GUIPreferences     WAIApp::uiPrefs;
SLGLTexture*       WAIApp::cpvrLogo = nullptr;
SLGLTexture*       WAIApp::videoImage = nullptr;
AppWAIDirectories* WAIApp::dirs = nullptr;
AppWAIScene*       WAIApp::waiScene = nullptr;
WAI::WAI*          WAIApp::wai = nullptr;
WAICalibration*    WAIApp::wc;
int                WAIApp::scrWidth;
int                WAIApp::scrHeight;
cv::VideoWriter*   WAIApp::videoWriter = nullptr;
cv::VideoWriter*   WAIApp::videoWriterInfo = nullptr;
WAI::ModeOrbSlam2* WAIApp::mode        = nullptr;
bool               WAIApp::loaded      = false;

int WAIApp::load(int width, int height, float scr2fbX, float scr2fbY, int dpi, AppWAIDirectories* directories)
{
    dirs = directories;
    WAIMapStorage::init(dirs->writableDir);

    wai             = new WAI::WAI(dirs->waiDataRoot);
    wc              = new WAICalibration();
    waiScene        = new AppWAIScene();
    videoWriter     = new cv::VideoWriter();
    videoWriterInfo = new cv::VideoWriter();

    wc->changeImageSize(width, height);
    wc->loadFromFile(dirs->waiDataRoot + "/calibrations/cam_calibration_main.xml");
    WAI::CameraCalibration calibration = wc->getCameraCalibration();
    wai->activateSensor(WAI::SensorType_Camera, &calibration);

    SLVstring empty;
    empty.push_back("WAI APP");
    slCreateAppAndScene(empty,
                        dirs->slDataRoot + "/shaders/",
                        dirs->slDataRoot + "/models/",
                        dirs->slDataRoot + "/images/textures/",
                        dirs->slDataRoot + "/images/fonts/",
                        dirs->writableDir,
                        "AppDemoGLFW",
                        (void*)WAIApp::onLoadWAISceneView);

    // This load the GUI configs that are locally stored
    uiPrefs.load(dpi);
    setupGUI();

    int svIndex = slCreateSceneView((int)(width * scr2fbX),
                                    (int)(height * scr2fbY),
                                    dpi,
                                    (SLSceneID)0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    (void*)buildGUI);

    loaded = true;
    return svIndex;
}

void WAIApp::setupGUI()
{
    aboutDial = new AppDemoGuiAbout("about", cpvrLogo, &uiPrefs.showAbout);
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsTiming("timing", &uiPrefs.showStatsTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsDebugTiming("debug timing", &uiPrefs.showStatsDebugTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsVideo("video", wc, &uiPrefs.showStatsVideo));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosScene("scene", &uiPrefs.showInfosScene));
    AppDemoGui::addInfoDialog(new AppDemoGuiTransform("transform", &uiPrefs.showTransform));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosFrameworks("frameworks", &uiPrefs.showInfosFrameworks));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosSensors("sensors", &uiPrefs.showInfosSensors));
    AppDemoGui::addInfoDialog(new AppDemoGuiUIPrefs("prefs", &uiPrefs, &uiPrefs.showUIPrefs));
    AppDemoGui::addInfoDialog(new AppDemoGuiProperties("properties", &uiPrefs.showProperties));
}

void WAIApp::buildGUI(SLScene* s, SLSceneView* sv)
{
    if (uiPrefs.showAbout)
    {
        aboutDial->buildInfos(s, sv);
    }
    else
    {
        AppDemoGui::buildInfosDialogs(s, sv);
        AppDemoGuiMenu::build(&uiPrefs, s, sv);
    }
}

void WAIApp::openTest(std::string path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    cv::FileNode n = fs["Links"];
    std::string cal_path = (std::string)n["Calibration"];
    std::string vid_path = (std::string)n["Videos"];
    std::string map_path = (std::string)n["Maps"];
}

void WAIApp::refreshTexture(cv::Mat *image)
{
    videoImage->copyVideoImage(image->cols, image->rows,
                               CVCapture::instance()->format,
                               image->data,
                               image->isContinuous(),
                               true);
}

//-----------------------------------------------------------------------------
void WAIApp::onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid)
{
    s->init();
    waiScene->rebuild();

    // Set scene name and info string
    s->name("Track Keyframe based Features");
    s->info("Example for loading an existing pose graph with map points.");

    // Save no energy
    sv->doWaitOnIdle(false); //for constant video feed
    sv->camera(waiScene->cameraNode);

    if (videoImage == nullptr)
    {
        videoImage = new SLGLTexture("LiveVideoError.png", GL_LINEAR, GL_LINEAR);
    }
    waiScene->cameraNode->background().texture(videoImage);
    waiScene->cameraNode->fov(wc->calcCameraFOV());

    s->root3D(waiScene->rootNode);

    sv->onInitialize();

    mode = ((WAI::ModeOrbSlam2*)wai->setMode(WAI::ModeType_ORB_SLAM2));
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

    slUpdateScene();
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
    if(!loaded || videoImage == nullptr)
        return;

    if (videoWriter->isOpened()) {
        videoWriter->write(*cameraData->imageRGB);
    }

    wai->updateSensor(WAI::SensorType_Camera, cameraData);
    refreshTexture(cameraData->imageRGB);

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
