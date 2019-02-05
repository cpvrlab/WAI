#include <SLApplication.h>
#include <SLBox.h>
#include <SLLightSpot.h>
#include <SLCoordAxis.h>
#include <SLPoints.h>

#include <SLCVCamera.h>

#include <WAIMapStorage.h>

#include <AppDemoGui.h>
#include <AppDemoGuiTrackedMapping.h>
#include <AppDemoGuiMapStorage.h>
#include <AppDemoGuiInfosMapNodeTransform.h>
#include <AppDemoGuiInfosTracking.h>
#include <AppWAISceneView.h>

WAISceneView::WAISceneView(SLCVCalibration* calib, std::string externalDir, std::string dataRoot)
  : _wai(dataRoot),
    _mapPC(new SLNode("MapPC")),
    _mapMatchedPC(new SLNode("MapMatchedPC")),
    _mapLocalPC(new SLNode("MapLocalPC")),
    _keyFrameNode(new SLNode("KeyFrames")),
    _covisibilityGraph(new SLNode("CovisibilityGraph")),
    _spanningTree(new SLNode("SpanningTree")),
    _loopEdges(new SLNode("LoopEdges")),
    _redMat(new SLMaterial(SLCol4f::RED, "Red")),
    _greenMat(new SLMaterial(SLCol4f::GREEN, "Green")),
    _blueMat(new SLMaterial(SLCol4f::BLUE, "Blue")),
    _covisibilityGraphMat(new SLMaterial("YellowLines", SLCol4f::YELLOW)),
    _spanningTreeMat(new SLMaterial("GreenLines", SLCol4f::GREEN)),
    _loopEdgesMat(new SLMaterial("RedLines", SLCol4f::RED)),
    _externalDir(externalDir)
{
    WAIMapStorage::init(externalDir);
    WAI::CameraCalibration calibration = {calib->fx(),
                                          calib->fy(),
                                          calib->cx(),
                                          calib->cy(),
                                          calib->k1(),
                                          calib->k2(),
                                          calib->p1(),
                                          calib->p2()};
    _wai.activateSensor(WAI::SensorType_Camera, &calibration);
    _mode = (WAI::ModeOrbSlam2*)_wai.setMode(WAI::ModeType_ORB_SLAM2);
}
//-----------------------------------------------------------------------------
void WAISceneView::setMapNode(SLNode* mapNode)
{
    _mapNode = mapNode;
    _mapNode->addChild(_mapPC);
    _mapNode->addChild(_mapMatchedPC);
    _mapNode->addChild(_mapLocalPC);
    _mapNode->addChild(_keyFrameNode);
    _mapNode->addChild(_covisibilityGraph);
    _mapNode->addChild(_spanningTree);
    _mapNode->addChild(_loopEdges);
}
//-----------------------------------------------------------------------------
void onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid)
{
    WAISceneView* waiSceneView = (WAISceneView*)sv;
    // Set scene name and info string
    s->name("Track Keyframe based Features");
    s->info("Example for loading an existing pose graph with map points.");

    s->videoType(VT_MAIN);

    //make some light
    SLLightSpot* light1 = new SLLightSpot(1, 1, 1, 0.3f);
    light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
    light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
    light1->specular(SLCol4f(1, 1, 1));
    light1->attenuation(1, 0, 0);

    //always equal for tracking
    //setup tracking camera
    SLCamera* cameraNode = new SLCamera("Camera 1");
    cameraNode->translation(0, 0, 0.1f);
    cameraNode->lookAt(0, 0, 0);
    //for tracking we have to use the field of view from calibration
    cameraNode->fov(SLApplication::activeCalib->cameraFovDeg());
    cameraNode->clipNear(0.001f);
    cameraNode->clipFar(1000000.0f); // Increase to infinity?
    cameraNode->setInitialState();
    cameraNode->background().texture(s->videoTexture());

    // Save no energy
    sv->doWaitOnIdle(false); //for constant video feed
    sv->camera(cameraNode);
    waiSceneView->setCameraNode(cameraNode);

    //add yellow box and axis for augmentation
    SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
    SLfloat     l = 0.593f, b = 0.466f, h = 0.257f;
    SLBox*      box1     = new SLBox(0.0f, 0.0f, 0.0f, l, h, b, "Box 1", yellow);
    SLNode*     boxNode  = new SLNode(box1, "boxNode");
    SLNode*     axisNode = new SLNode(new SLCoordAxis(), "axis node");
    boxNode->addChild(axisNode);
    //boxNode->translation(0.0f, 0.0f, -2.0f);

    SLNode* mapNode = new SLNode("map");
    waiSceneView->setMapNode(mapNode);

    mapNode->rotate(180, 1, 0, 0);
    mapNode->addChild(cameraNode);

    //setup scene
    SLNode* scene = new SLNode("scene");
    scene->addChild(light1);
    scene->addChild(boxNode);
    scene->addChild(mapNode);

    s->root3D(scene);

    sv->onInitialize();
    s->onAfterLoad();

    auto trackedMapping = std::make_shared<AppDemoGuiTrackedMapping>("Tracked mapping", waiSceneView->getMode());
    AppDemoGui::addInfoDialog(trackedMapping);
    auto mapStorage = std::make_shared<AppDemoGuiMapStorage>("Map Storage",
                                                             waiSceneView->getMode(),
                                                             mapNode,
                                                             waiSceneView->getExternalDir() + "slam-maps");
    AppDemoGui::addInfoDialog(mapStorage);
    auto mapTransform = std::make_shared<AppDemoGuiInfosMapNodeTransform>("Map transform",
                                                                          mapNode,
                                                                          waiSceneView->getMode(),
                                                                          waiSceneView->getExternalDir());
    AppDemoGui::addInfoDialog(mapTransform);
    auto trackingInfos = std::make_shared<AppDemoGuiInfosTracking>("Tracking Infos",
                                                                   waiSceneView,
                                                                   waiSceneView->getMode());
    AppDemoGui::addInfoDialog(trackingInfos);
}
//-----------------------------------------------------------------------------
void WAISceneView::update()
{
    cv::Mat pose          = cv::Mat(4, 4, CV_32F);
    bool    iKnowWhereIAm = _wai.whereAmI(&pose);

    if (iKnowWhereIAm)
    {
        // update map node
        if (_mappointsMesh)
        {
            _mapPC->deleteMesh(_mappointsMesh);
        }
        if (_showKeyPoints)
        {
            renderMapPoints("MapPoints",
                            _mode->getMapPoints(),
                            _mapPC,
                            _mappointsMesh,
                            _redMat);
        }

        if (_mappointsMatchedMesh)
        {
            _mapMatchedPC->deleteMesh(_mappointsMatchedMesh);
        }
        if (_showKeyPointsMatched)
        {
            renderMapPoints("MatchedMapPoints",
                            _mode->getMatchedMapPoints(),
                            _mapMatchedPC,
                            _mappointsMatchedMesh,
                            _greenMat);
        }

        if (_mappointsLocalMesh)
        {
            _mapLocalPC->deleteMesh(_mappointsLocalMesh);
        }
        if (_showLocalMapPC)
        {
            renderMapPoints("LocalMapPoints",
                            _mode->getLocalMapPoints(),
                            _mapLocalPC,
                            _mappointsLocalMesh,
                            _blueMat);
        }

        _keyFrameNode->deleteChildren();
        if (_showKeyFrames)
        {
            renderKeyFrames();
        }

        renderGraphs();

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

        _cameraNode->om(slPose);
    }
}
//-----------------------------------------------------------------------------
void WAISceneView::updateCamera(WAI::CameraData* cameraData)
{
    _wai.updateSensor(WAI::SensorType_Camera, cameraData);
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
void WAISceneView::renderKeyFrames()
{
    std::vector<WAIKeyFrame*> keyframes = _mode->getKeyFrames();

    // TODO(jan): delete keyframe textures
    for (WAIKeyFrame* kf : keyframes)
    {
        SLCVCamera* cam = new SLCVCamera("KeyFrame " + std::to_string(kf->mnId));
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
        cam->focalDist(0.11);
        cam->clipNear(0.1);
        cam->clipFar(1000.0);
        _keyFrameNode->addChild(cam);
    }
}
//-----------------------------------------------------------------------------
void WAISceneView::renderGraphs()
{
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

    if (_covisibilityGraphMesh)
        _covisibilityGraph->deleteMesh(_covisibilityGraphMesh);

    if (covisGraphPts.size())
    {
        _covisibilityGraphMesh = new SLPolyline(covisGraphPts, false, "CovisibilityGraph", _covisibilityGraphMat);
        _covisibilityGraph->addMesh(_covisibilityGraphMesh);
        _covisibilityGraph->updateAABBRec();
    }

    if (_spanningTreeMesh)
        _spanningTree->deleteMesh(_spanningTreeMesh);

    if (spanningTreePts.size())
    {
        _spanningTreeMesh = new SLPolyline(spanningTreePts, false, "SpanningTree", _spanningTreeMat);
        _spanningTree->addMesh(_spanningTreeMesh);
        _spanningTree->updateAABBRec();
    }

    if (_loopEdgesMesh)
        _loopEdges->deleteMesh(_loopEdgesMesh);

    if (loopEdgesPts.size())
    {
        _loopEdgesMesh = new SLPolyline(loopEdgesPts, false, "LoopEdges", _loopEdgesMat);
        _loopEdges->addMesh(_loopEdgesMesh);
        _loopEdges->updateAABBRec();
    }
}
