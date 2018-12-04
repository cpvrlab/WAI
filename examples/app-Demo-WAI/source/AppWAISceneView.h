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

#include <SLSceneView.h>
#include <SLPoints.h>

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

    void setCameraNode(SLCamera* cameraNode) { _cameraNode = cameraNode; }
    void setMapNode(SLNode* mapNode) { _mapNode = mapNode; }
    void setKeyFrameNode(SLNode* keyFrameNode) { _keyFrameNode = keyFrameNode; }

    WAI::ModeOrbSlam2* getMode() { return _mode; }
    std::string        getExternalDir() { return _externalDir; }

    private:
    WAI::WAI           _wai;
    WAI::ModeOrbSlam2* _mode;
    SLCamera*          _cameraNode   = nullptr;
    SLNode*            _mapNode      = nullptr;
    SLNode*            _keyFrameNode = nullptr;

    SLMaterial* _redMat   = new SLMaterial(SLCol4f::RED, "Red");
    SLMaterial* _greenMat = new SLMaterial(SLCol4f::GREEN, "Green");
    SLMaterial* _blueMat  = new SLMaterial(SLCol4f::BLUE, "Blue");

    SLPoints* _mappointsMesh        = nullptr;
    SLPoints* _mappointsMatchedMesh = nullptr;
    SLPoints* _mappointsLocalMesh   = nullptr;

    std::string _externalDir;
};
//-----------------------------------------------------------------------------
