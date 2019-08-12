#ifndef APP_WAI_SCENE
#define APP_WAI_SCENE

#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>

#include <SLCVCalibration.h>

#include <WAI.h>

class AppWAIScene
{
    public:
    AppWAIScene();
    SLCamera* cameraNode        = nullptr;

    SLNode*   rootNode          = nullptr;
    SLNode*   mapNode           = nullptr;
    SLNode*   mapPC             = nullptr;
    SLNode*   mapMatchedPC      = nullptr;
    SLNode*   mapLocalPC        = nullptr;
    SLNode*   keyFrameNode      = nullptr;
    SLNode*   covisibilityGraph = nullptr;
    SLNode*   spanningTree      = nullptr;
    SLNode*   loopEdges         = nullptr;

    SLMaterial* redMat               = nullptr;
    SLMaterial* greenMat             = nullptr;
    SLMaterial* blueMat              = nullptr;
    SLMaterial* covisibilityGraphMat = nullptr;
    SLMaterial* spanningTreeMat      = nullptr;
    SLMaterial* loopEdgesMat         = nullptr;

    SLPoints*   mappointsMesh         = nullptr;
    SLPoints*   mappointsMatchedMesh  = nullptr;
    SLPoints*   mappointsLocalMesh    = nullptr;
    SLPolyline* covisibilityGraphMesh = nullptr;
    SLPolyline* spanningTreeMesh      = nullptr;
    SLPolyline* loopEdgesMesh         = nullptr;

    void rebuild();
};

#endif
