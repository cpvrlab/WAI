#include <SLApplication.h>
#include <SLBox.h>
#include <SLLightSpot.h>

#include <AppWAISceneView.h>

//-----------------------------------------------------------------------------
void onLoad(SLScene* s, SLSceneView* sv, SLSceneID sid)
{
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
    SLCamera* trackingCam = new SLCamera("Camera 1");
    trackingCam->translation(0, 0, 0.1);
    trackingCam->lookAt(0, 0, 0);
    //for tracking we have to use the field of view from calibration
    trackingCam->fov(SLApplication::activeCalib->cameraFovDeg());
    trackingCam->clipNear(0.001f);
    trackingCam->clipFar(1000000.0f); // Increase to infinity?
    trackingCam->setInitialState();
    trackingCam->background().texture(s->videoTexture());

    //the map node contains the visual representation of the slam map
    SLCVMapNode* mapNode = new SLCVMapNode("map");

    // Save no energy
    sv->doWaitOnIdle(false); //for constant video feed
    sv->camera(trackingCam);

    //SLCVOrbTracking* orbT = raulMurTracker->orbTracking();
    //setup scene specific gui dialoges
    auto trackingInfos = std::make_shared<SLImGuiInfosTracking>("Tracking infos", tm, mapNode);
    AppDemoGui::addInfoDialog(trackingInfos);
    auto mapNodeTransform = std::make_shared<SLImGuiInfosMapNodeTransform>("Map node transform", mapNode, tm);
    AppDemoGui::addInfoDialog(mapNodeTransform);
    auto mapStorage = std::make_shared<SLImGuiMapStorage>("Map storage", tm);
    AppDemoGui::addInfoDialog(mapStorage);

    //add yellow box and axis for augmentation
    SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
    SLfloat     l = 0.593f, b = 0.466f, h = 0.257f;
    SLBox*      box1     = new SLBox(0.0f, 0.0f, 0.0f, l, h, b, "Box 1", yellow);
    SLNode*     boxNode  = new SLNode(box1, "boxNode");
    SLNode*     axisNode = new SLNode(new SLCoordAxis(), "axis node");
    boxNode->addChild(axisNode);

    //setup scene
    SLNode* scene = new SLNode("scene");
    scene->addChild(light1);
    scene->addChild(boxNode);
    scene->addChild(mapNode);

    s->root3D(scene);

    ((WAISceneView*)sv)->setMapNode(mapNode);
}
//-----------------------------------------------------------------------------
