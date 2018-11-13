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

#ifdef SL_MEMLEAKDETECT // set in SL.h for debug config only
#include <debug_new.h>  // memory leak detector
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
// Foreward declarations for helper functions used only in this file
SLNode *SphereGroup(SLint, SLfloat, SLfloat, SLfloat, SLfloat, SLuint,
                    SLMaterial *, SLMaterial *);
SLNode *BuildFigureGroup(SLMaterial *mat, SLbool withAnimation = false);
SLNode *LoadBernModel();
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
void appDemoLoadScene(SLScene *s, SLSceneView *sv, SLSceneID sceneID) {
  SLApplication::sceneID = sceneID;

  // remove scene specific uis
  AppDemoGui::clearInfoDialogs();
  // Initialize all preloaded stuff from SLScene
  s->init();

  switch (SLApplication::sceneID) {
  case SID_Empty: {
    s->name("No Scene loaded.");
    s->info("No Scene loaded.");
    s->root3D(nullptr);
    sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                               SLCol4f(0.2f, 0.2f, 0.2f));
    sv->camera(nullptr);
    sv->doWaitOnIdle(true);
  } break;

  case SID_Minimal:
  default: {
    // Set scene name and info string
    s->name("Minimal Scene Test");
    s->info("Minimal texture mapping example with one light source.");

    // Create textures and materials
    SLGLTexture *texC = new SLGLTexture("earth1024_C.jpg");
    SLMaterial *m1 = new SLMaterial("m1", texC);

    // Create a scene group node
    SLNode *scene = new SLNode("scene node");

    // Create a light source node
    SLLightSpot *light1 = new SLLightSpot(0.3f);
    light1->translation(0, 0, 5);
    light1->lookAt(0, 0, 0);
    light1->name("light node");
    scene->addChild(light1);

    // Create meshes and nodes
    SLMesh *rectMesh = new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1,
                                       "rectangle mesh", m1);
    SLNode *rectNode = new SLNode(rectMesh, "rectangle node");
    scene->addChild(rectNode);

    SLNode *axisNode = new SLNode(new SLCoordAxis(), "axis node");
    scene->addChild(axisNode);

    // Set background color and the root scene node
    sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                               SLCol4f(0.2f, 0.2f, 0.2f));

    // pass the scene group as root node
    s->root3D(scene);

    // Save energy
    sv->doWaitOnIdle(true);
  } break;
  }

  ////////////////////////////////////////////////////////////////////////////
  // call onInitialize on all scene views to init the scenegraph and stats
  for (auto sv : s->sceneViews()) {
    if (sv != nullptr) {
      sv->onInitialize();
    }
  }

  s->onAfterLoad();
}
