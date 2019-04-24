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

#ifndef APP_DEMO_CALIBRATION_SCENE_VIEW
#define APP_DEMO_CALIBRATION_SCENE_VIEW

#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>

#include <SLCVCalibration.h>

#include <WAI.h>

#define DATA_ORIENTED 0
#define LIVE_VIDEO 1

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
class CalibrationSceneView : public SLSceneView
{
    public:
    CalibrationSceneView(WAI::WAI wai, SLCamera* cameraNode, SLSceneView * sv);
    void update();
    void updateCamera(WAI::CameraData* cameraData);

    void setCameraNode(SLCamera* cameraNode) { _cameraNode = cameraNode; }
    void setMapNode(SLNode* mapNode);

    private:
    SLCVCalibration* _calib       = nullptr;
    SLCamera* _cameraNode         = nullptr;
};
//-----------------------------------------------------------------------------

#endif
