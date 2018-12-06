//#############################################################################
//  File:      WAISceneView.h
//  Purpose:   Node transform test application that demonstrates all transform
//             possibilities of SLNode
//  Author:    Marc Wacker
//  Date:      July 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_WAI_SCENE_VIEW
#define APP_WAI_SCENE_VIEW

#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>

#include <SLCVCalibration.h>

#include <WAI.h>

void onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid);

//-----------------------------------------------------------------------------
enum TransformMode
{
    TranslationMode,
    RotationMode,
    RotationAroundMode,
    LookAtMode
};
//-----------------------------------------------------------------------------
/*!
SLSceneView derived class for a node transform test application that
demonstrates all transform possibilities in SLNode
*/
class WAISceneView : public SLSceneView
{
    public:
    WAISceneView(SLCVCalibration* calib, std::string externalDir, std::string dataRoot);
    void update();
    void updateCamera(WAI::CameraData* cameraData);
    void updateMinNumOfCovisibles(int n);

    void setCameraNode(SLCamera* cameraNode) { _cameraNode = cameraNode; }
    void setMapNode(SLNode* mapNode);

    WAI::ModeOrbSlam2* getMode() { return _mode; }
    std::string        getExternalDir() { return _externalDir; }

    int getMinNumOfCovisibles() { return _minNumOfCovisibles; }

    bool showKeyPoints() { return _showKeyPoints; }
    void showKeyPoints(bool showKeyPoints) { _showKeyPoints = showKeyPoints; }
    bool showKeyPointsMatched() { return _showKeyPointsMatched; }
    void showKeyPointsMatched(bool showKeyPointsMatched) { _showKeyPointsMatched = showKeyPointsMatched; }
    bool showMapPC() { return _showMapPC; }
    void showMapPC(bool showMapPC) { _showMapPC = showMapPC; }
    bool showLocalMapPC() { return _showLocalMapPC; }
    void showLocalMapPC(bool showLocalMapPC) { _showLocalMapPC = showLocalMapPC; }
    bool showMatchesPC() { return _showMatchesPC; }
    void showMatchesPC(bool showMatchesPC) { _showMatchesPC = showMatchesPC; }
    bool showKeyFrames() { return _showKeyFrames; }
    void showKeyFrames(bool showKeyFrames) { _showKeyFrames = showKeyFrames; }
    bool renderKfBackground() { return _renderKfBackground; }
    void renderKfBackground(bool renderKfBackground) { _renderKfBackground = renderKfBackground; }
    bool allowKfsAsActiveCam() { return _allowKfsAsActiveCam; }
    void allowKfsAsActiveCam(bool allowKfsAsActiveCam) { _allowKfsAsActiveCam = allowKfsAsActiveCam; }
    bool showCovisibilityGraph() { return _showCovisibilityGraph; }
    void showCovisibilityGraph(bool showCovisibilityGraph) { _showCovisibilityGraph = showCovisibilityGraph; }
    bool showSpanningTree() { return _showSpanningTree; }
    void showSpanningTree(bool showSpanningTree) { _showSpanningTree = showSpanningTree; }
    bool showLoopEdges() { return _showLoopEdges; }
    void showLoopEdges(bool showLoopEdges) { _showLoopEdges = showLoopEdges; }

    private:
    WAI::WAI           _wai;
    WAI::ModeOrbSlam2* _mode;
    SLCamera*          _cameraNode        = nullptr;
    SLNode*            _mapNode           = nullptr;
    SLNode*            _keyFrameNode      = nullptr;
    SLNode*            _covisibilityGraph = nullptr;
    SLNode*            _spanningTree      = nullptr;
    SLNode*            _loopEdges         = nullptr;

    SLMaterial* _redMat               = nullptr;
    SLMaterial* _greenMat             = nullptr;
    SLMaterial* _blueMat              = nullptr;
    SLMaterial* _covisibilityGraphMat = nullptr;
    SLMaterial* _spanningTreeMat      = nullptr;
    SLMaterial* _loopEdgesMat         = nullptr;

    SLPoints*   _mappointsMesh         = nullptr;
    SLPoints*   _mappointsMatchedMesh  = nullptr;
    SLPoints*   _mappointsLocalMesh    = nullptr;
    SLPolyline* _covisibilityGraphMesh = nullptr;
    SLPolyline* _spanningTreeMesh      = nullptr;
    SLPolyline* _loopEdgesMesh         = nullptr;

    std::string _externalDir;

    void renderMapPoints();
    void renderMatchedMapPoints();
    void renderLocalMapPoints();
    void renderKeyFrames();
    void renderGraphs();

    //! minimum number of covisibles for covisibility graph visualization
    int   _minNumOfCovisibles = 50;
    float _meanReprojectionError;

    bool _showKeyPoints         = true;
    bool _showKeyPointsMatched  = true;
    bool _showMapPC             = true;
    bool _showLocalMapPC        = true;
    bool _showMatchesPC         = true;
    bool _showKeyFrames         = true;
    bool _renderKfBackground    = true;
    bool _allowKfsAsActiveCam   = true;
    bool _showCovisibilityGraph = true;
    bool _showSpanningTree      = true;
    bool _showLoopEdges         = true;
};
//-----------------------------------------------------------------------------

#endif
