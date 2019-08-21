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

#include "AppWAIScene.h"
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>

#include <SLCVCalibration.h>
#include <WAIAutoCalibration.h>

#include <WAI.h>

#define LIVE_VIDEO 1

//-----------------------------------------------------------------------------
class WAIApp
{
    public:
    static int load(int width, int height, float scr2fbX, float scr2fbY, int dpi,
                  std::string extDir, std::string dataRoot, std::string slRoot);

    static void onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid);
    static void update();
    static void updateCamera(WAI::CameraData* cameraData);
    static void updateMinNumOfCovisibles(int n);

    static void updateTrackingVisualization(const bool iKnowWhereIAm);

    static void renderMapPoints(std::string                      name,
                         const std::vector<WAIMapPoint*>& pts,
                         SLNode*&                         node,
                         SLPoints*&                       mesh,
                         SLMaterial*&                     material);

    static void renderKeyframes();
    static void renderGraphs();

    //! minimum number of covisibles for covisibility graph visualization
    static std::string        rootPath;
    static std::string        externalDir;
    static WAI::WAI*          wai;
    static WAICalibration*    wc;
    static int                scrWidth;
    static int                scrHeight;
    static cv::VideoWriter*   videoWriter;
    static cv::VideoWriter*   videoWriterInfo;
    static WAI::ModeOrbSlam2* mode;
    static AppWAIScene*       waiScene;
    static bool               loaded;

    static int   minNumOfCovisibles;
    static float meanReprojectionError;
    static bool  showKeyPoints;
    static bool  showKeyPointsMatched;
    static bool  showMapPC;
    static bool  showLocalMapPC;
    static bool  showMatchesPC;
    static bool  showKeyFrames;
    static bool  renderKfBackground;
    static bool  allowKfsAsActiveCam;
    static bool  showCovisibilityGraph;
    static bool  showSpanningTree;
    static bool  showLoopEdges;
};
//-----------------------------------------------------------------------------

#endif
