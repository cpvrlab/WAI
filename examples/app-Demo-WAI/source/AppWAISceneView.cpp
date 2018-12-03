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
#include <AppWAISceneView.h>

WAISceneView::WAISceneView(SLCVCalibration* calib, std::string externalDir)
  : _externalDir(externalDir)
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
    cameraNode->translation(0, 0, 0.1);
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

    //add yellow box and axis for augmentation
    SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
    SLfloat     l = 0.593f, b = 0.466f, h = 0.257f;
    SLBox*      box1     = new SLBox(0.0f, 0.0f, 0.0f, l, h, b, "Box 1", yellow);
    SLNode*     boxNode  = new SLNode(box1, "boxNode");
    SLNode*     axisNode = new SLNode(new SLCoordAxis(), "axis node");
    boxNode->addChild(axisNode);
    boxNode->translation(0.0f, 0.0f, -2.0f);

    SLNode* keyFrameNode = new SLNode("KeyFrames");

    SLNode* mapNode = new SLNode("map");
    mapNode->rotate(180, 1, 0, 0);
    mapNode->addChild(cameraNode);
    mapNode->addChild(keyFrameNode);

    //setup scene
    SLNode* scene = new SLNode("scene");
    scene->addChild(light1);
    scene->addChild(boxNode);
    scene->addChild(mapNode);

    s->root3D(scene);

    waiSceneView->setCameraNode(cameraNode);
    waiSceneView->setMapNode(mapNode);
    waiSceneView->setKeyFrameNode(keyFrameNode);

    sv->onInitialize();
    s->onAfterLoad();

    auto trackingInfos = std::make_shared<AppDemoGuiTrackedMapping>("Tracked mapping", waiSceneView->getMode());
    AppDemoGui::addInfoDialog(trackingInfos);
    auto mapStorage = std::make_shared<AppDemoGuiMapStorage>("Map Storage",
                                                             waiSceneView->getMode(),
                                                             mapNode,
                                                             waiSceneView->getExternalDir());
    AppDemoGui::addInfoDialog(mapStorage);
}
//-----------------------------------------------------------------------------
void WAISceneView::update()
{
    cv::Mat pose          = cv::Mat(4, 4, CV_32F);
    bool    iKnowWhereIAm = _wai.whereAmI(&pose);

    if (iKnowWhereIAm)
    {
        // update map node
        std::vector<WAIMapPoint*> mapPoints = _mode->getMapPoints();

        SLVVec3f points, normals;
        for (WAIMapPoint* mapPoint : mapPoints)
        {
            SLVec3f worldPos = SLVec3f(mapPoint->worldPosVec().x,
                                       mapPoint->worldPosVec().y,
                                       mapPoint->worldPosVec().z);
            SLVec3f normal   = SLVec3f(mapPoint->normalVec().x,
                                     mapPoint->normalVec().y,
                                     mapPoint->normalVec().z);
            points.push_back(worldPos);
            normals.push_back(normal);
        }

        if (_mappointsMesh)
        {
            _mapNode->deleteMesh(_mappointsMesh);
        }
        _mappointsMesh = new SLPoints(points, normals, "Map Points", _redMat);
        _mapNode->addMesh(_mappointsMesh);
        _mapNode->updateAABBRec();

        std::vector<WAIMapPoint*> mapPointsMatched = _mode->getMatchedMapPoints();

        points.clear();
        normals.clear();
        for (WAIMapPoint* mapPoint : mapPointsMatched)
        {
            SLVec3f worldPos = SLVec3f(mapPoint->worldPosVec().x,
                                       mapPoint->worldPosVec().y,
                                       mapPoint->worldPosVec().z);
            SLVec3f normal   = SLVec3f(mapPoint->normalVec().x,
                                     mapPoint->normalVec().y,
                                     mapPoint->normalVec().z);
            points.push_back(worldPos);
            normals.push_back(normal);
        }

        if (_mappointsMatchedMesh)
        {
            _mapNode->deleteMesh(_mappointsMatchedMesh);
        }
        _mappointsMatchedMesh = new SLPoints(points, normals, "Map Points Matched", _greenMat);
        _mapNode->addMesh(_mappointsMatchedMesh);
        _mapNode->updateAABBRec();

        std::vector<WAIMapPoint*> mapPointsLocal = _mode->getLocalMapPoints();

        points.clear();
        normals.clear();
        for (WAIMapPoint* mapPoint : mapPointsLocal)
        {
            SLVec3f worldPos = SLVec3f(mapPoint->worldPosVec().x,
                                       mapPoint->worldPosVec().y,
                                       mapPoint->worldPosVec().z);
            SLVec3f normal   = SLVec3f(mapPoint->normalVec().x,
                                     mapPoint->normalVec().y,
                                     mapPoint->normalVec().z);
            points.push_back(worldPos);
            normals.push_back(normal);
        }

        if (_mappointsLocalMesh)
        {
            _mapNode->deleteMesh(_mappointsLocalMesh);
        }
        _mappointsLocalMesh = new SLPoints(points, normals, "Map Points Local", _blueMat);
        _mapNode->addMesh(_mappointsLocalMesh);
        _mapNode->updateAABBRec();

        std::vector<WAIKeyFrame*> keyframes = _mode->getKeyFrames();

        _keyFrameNode->deleteChildren();
        // TODO(jan): delete keyframe textures
        for (WAIKeyFrame* kf : keyframes)
        {
            // TODO(jan): maybe adjust the name per camera
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
