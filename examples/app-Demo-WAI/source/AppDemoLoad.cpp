//#############################################################################
//  File:      AppDemoSceneLoad.cpp
//  Author:    Marcus Hudritsch
//  Date:      Februar 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLAssimpImporter.h>
#include <SLScene.h>
#include <SLSceneView.h>

#include <SLBox.h>
#include <SLCVCapture.h>
#include <SLCVMapNode.h>
#include <SLCVTrackedAruco.h>
#include <SLCVTrackedChessboard.h>
#include <SLCVTrackedFaces.h>
#include <SLCVTrackedFeatures.h>
#include <SLCVTrackedMapping.h>
#include <SLCVTrackedRaulMur.h>
#include <SLCVTrackedRaulMurAsync.h>
#include <SLCone.h>
#include <SLCoordAxis.h>
#include <SLCylinder.h>
#include <SLDisk.h>
#include <SLGrid.h>
#include <SLLens.h>
#include <SLLightDirect.h>
#include <SLLightRect.h>
#include <SLLightSpot.h>
#include <SLPoints.h>
#include <SLPolygon.h>
#include <SLRectangle.h>
#include <SLSkybox.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SLTransferFunction.h>

#include <SLCVKeyFrameDB.h>
#include <SLCVMap.h>
#include <SLCVMapIO.h>
#include <SLCVMapPoint.h>
#include <SLCVMapStorage.h>
#include <SLCVOrbVocabulary.h>
#include <SLImGuiInfosCameraMovement.h>
#include <SLImGuiInfosChristoffelTower.h>
#include <SLImGuiInfosMapNodeTransform.h>
#include <SLImGuiInfosMemoryStats.h>
#include <SLImGuiInfosTracking.h>
#include <SLImGuiMapStorage.h>

#include <AppDemoGui.h>
#include <SLCVMapStorage.h>
#include <SLImGuiTrackedMapping.h>

//-----------------------------------------------------------------------------
//! appDemoLoadScene builds a scene from source code.
/*! appDemoLoadScene builds a scene from source code. Such a function must be
passed as a void*-pointer to slCreateScene. It will be called from within
slCreateSceneView as soon as the view is initialized. You could separate
different scene by a different sceneID.<br>
The purpose is to assemble a scene by creating scenegraph objects with nodes
(SLNode) and meshes (SLMesh). See the scene with SID_Minimal for a minimal
example of the different steps.
*/
void appDemoLoadScene(SLScene* s, SLSceneView* sv, SceneID sceneID)
{
    //SLApplication::sceneID = sceneID;

    // remove scene specific uis
    AppDemoGui::clearInfoDialogs();
    // Initialize all preloaded stuff from SLScene
    s->init();

    switch (sceneID)
    {
        case Scene_Empty:
        {
            s->name("No Scene loaded.");
            s->info("No Scene loaded.");
            s->root3D(nullptr);
            sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));
            sv->camera(nullptr);
            sv->doWaitOnIdle(true);
        }
        break;

        case Scene_WAI:
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
        }
        break;

        case Scene_None:
        case Scene_Minimal:
        default:
        {
            // Set scene name and info string
            s->name("Minimal Scene Test");
            s->info("Minimal texture mapping example with one light source.");

            // Create textures and materials
            SLGLTexture* texC = new SLGLTexture("earth1024_C.jpg");
            SLMaterial*  m1   = new SLMaterial("m1", texC);

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");

            // Create a light source node
            SLLightSpot* light1 = new SLLightSpot(0.3f);
            light1->translation(0, 0, 5);
            light1->lookAt(0, 0, 0);
            light1->name("light node");
            scene->addChild(light1);

            // Create meshes and nodes
            SLMesh* rectMesh = new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "rectangle mesh", m1);
            SLNode* rectNode = new SLNode(rectMesh, "rectangle node");
            scene->addChild(rectNode);

            SLNode* axisNode = new SLNode(new SLCoordAxis(), "axis node");
            scene->addChild(axisNode);

            // Set background color and the root scene node
            sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));

            // pass the scene group as root node
            s->root3D(scene);

            // Save energy
            sv->doWaitOnIdle(true);
        }
        break;
    }

    ////////////////////////////////////////////////////////////////////////////
    // call onInitialize on all scene views to init the scenegraph and stats
    for (auto sv : s->sceneViews())
    {
        if (sv != nullptr)
        {
            sv->onInitialize();
        }
    }

    s->onAfterLoad();
}
