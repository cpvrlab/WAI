//#############################################################################
//  File:      AppDemoGuiInfosTracking.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <imgui.h>
#include <imgui_internal.h>

#include <AppDemoGuiInfosTracking.h>

//-----------------------------------------------------------------------------
AppDemoGuiInfosTracking::AppDemoGuiInfosTracking(std::string        name,
                                                 WAISceneView*      sceneView,
                                                 WAI::ModeOrbSlam2* mode)
  : AppDemoGuiInfosDialog(name),
    _sceneView(sceneView),
    _mode(mode)
{
    _minNumCovisibleMapPts = _sceneView->getMinNumOfCovisibles();
}
//-----------------------------------------------------------------------------
void AppDemoGuiInfosTracking::buildInfos()
{
    //-------------------------------------------------------------------------
    //numbers
    //add tracking state
    ImGui::Text("Tracking State : %s ", _mode->getPrintableState().c_str());
    //tracking type
    ImGui::Text("Tracking Type : %s ", _mode->getPrintableType().c_str());
    //mean reprojection error
    ImGui::Text("Mean Reproj. Error : %f ", _mode->getMeanReprojectionError());
    //add number of matches map points in current frame
    ImGui::Text("Num Map Matches : %d ", _mode->getNMapMatches());
    //L2 norm of the difference between the last and the current camera pose
    ImGui::Text("Pose Difference : %f ", _mode->poseDifference());
    ImGui::Separator();

    bool b;
    //-------------------------------------------------------------------------
    //keypoints infos
    if (ImGui::CollapsingHeader("KeyPoints"))
    {
        //show 2D key points in video image
        b = _sceneView->showKeyPoints();
        if (ImGui::Checkbox("KeyPts", &b))
        {
            _sceneView->showKeyPoints(b);
        }

        //show matched 2D key points in video image
        b = _sceneView->showKeyPointsMatched();
        if (ImGui::Checkbox("KeyPts Matched", &b))
        {
            _sceneView->showKeyPointsMatched(b);
        }
    }
    //-------------------------------------------------------------------------
    //mappoints infos
    if (ImGui::CollapsingHeader("MapPoints"))
    {
        //number of map points
        ImGui::Text("Count : %d ", _mode->getMapPointCount());
        //show and update all mappoints
        b = _sceneView->showMapPC();
        ImGui::Checkbox("Show Map Pts", &b);
        _sceneView->showMapPC(b);

        //show and update matches to mappoints
        b = _sceneView->showMatchesPC();
        if (ImGui::Checkbox("Show Matches to Map Pts", &b))
        {
            _sceneView->showMatchesPC(b);
        }
        //show and update local map points
        b = _sceneView->showLocalMapPC();
        if (ImGui::Checkbox("Show Local Map Pts", &b))
        {
            _sceneView->showLocalMapPC(b);
        }
    }
    //-------------------------------------------------------------------------
    //keyframe infos
    if (ImGui::CollapsingHeader("KeyFrames"))
    {
        //add number of keyframes
        ImGui::Text("Number of Keyframes : %d ", _mode->getKeyFrameCount());
        //show keyframe scene objects
        //show and update all mappoints
        b = _sceneView->showKeyFrames();
        ImGui::Checkbox("Show", &b);
        _sceneView->showKeyFrames(b);

        //if backgound rendering is active kf images will be rendered on
        //near clipping plane if kf is not the active camera
        b = _sceneView->renderKfBackground();
        ImGui::Checkbox("Show Image", &b);
        _sceneView->renderKfBackground(b);

        //allow SLCVCameras as active camera so that we can look through it
        b = _sceneView->allowKfsAsActiveCam();
        ImGui::Checkbox("Allow as Active Cam", &b);
        _sceneView->allowKfsAsActiveCam(b);
    }

    //-------------------------------------------------------------------------
    //keyframe infos
    if (ImGui::CollapsingHeader("Graph"))
    {
        //covisibility graph
        b = _sceneView->showCovisibilityGraph();
        ImGui::Checkbox("Show Covisibility (100 common KPts)", &b);
        _sceneView->showCovisibilityGraph(b);
        if (b)
        {
            //Definition of minimum number of covisible map points
            if (ImGui::InputInt("Min. covis. map pts", &_minNumCovisibleMapPts, 10, 0))
            {
                _sceneView->updateMinNumOfCovisibles(_minNumCovisibleMapPts);
            }
        }
        //spanning tree
        b = _sceneView->showSpanningTree();
        ImGui::Checkbox("Show spanning tree", &b);
        _sceneView->showSpanningTree(b);
        //loop edges
        b = _sceneView->showLoopEdges();
        ImGui::Checkbox("Show loop edges", &b);
        _sceneView->showLoopEdges(b);
    }
}
