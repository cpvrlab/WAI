//#############################################################################
//  File:      AppDemoGuiMapStorage.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <imgui.h>
#include <imgui_internal.h>

#include <AppDemoGuiMapStorage.h>

//-----------------------------------------------------------------------------
AppDemoGuiMapStorage::AppDemoGuiMapStorage(const string&      name,
                                           WAI::ModeOrbSlam2* tracking,
                                           SLNode*            node,
                                           std::string        externalDir)
  : AppDemoGuiInfosDialog(name),
    _tracking(tracking),
    _node(node),
    _externalDir(externalDir)
{
    wai_assert(tracking);
    _map  = tracking->getMap();
    _kfDB = tracking->getKfDB();
}
//-----------------------------------------------------------------------------
void AppDemoGuiMapStorage::buildInfos()
{
    if (!_map /* || !_mapNode*/)
        return;

    if (ImGui::Button("Save map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        SLMat4f om           = _node->om();
        cv::Mat cvOm         = cv::Mat(4, 4, CV_32F);
        cvOm.at<float>(0, 0) = om.m(0);
        cvOm.at<float>(0, 1) = -om.m(1);
        cvOm.at<float>(0, 2) = -om.m(2);
        cvOm.at<float>(0, 3) = om.m(12);
        cvOm.at<float>(1, 0) = om.m(4);
        cvOm.at<float>(1, 1) = -om.m(5);
        cvOm.at<float>(1, 2) = -om.m(6);
        cvOm.at<float>(1, 3) = -om.m(13);
        cvOm.at<float>(2, 0) = om.m(8);
        cvOm.at<float>(2, 1) = -om.m(9);
        cvOm.at<float>(2, 2) = -om.m(10);
        cvOm.at<float>(2, 3) = -om.m(14);
        cvOm.at<float>(3, 3) = 1.0f;
        WAIMapStorage::saveMap(WAIMapStorage::getCurrentId(), _tracking, true, cvOm, _externalDir);
        //update key frames, because there may be new textures in file system
        //_mapNode->updateKeyFrames(_map->GetAllKeyFrames());
    }

    ImGui::Separator();
    if (ImGui::Button("New map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        //increase current id and maximum id in MapStorage
        WAIMapStorage::newMap();
        //clear current field in combobox, until this new map is saved
        WAIMapStorage::currItem = nullptr;
        WAIMapStorage::currN    = -1;
    }

    ImGui::Separator();
    {
        if (ImGui::BeginCombo("Current", WAIMapStorage::currItem)) // The second parameter is the label previewed before opening the combo.
        {
            for (int n = 0; n < WAIMapStorage::existingMapNames.size(); n++)
            {
                bool isSelected = (WAIMapStorage::currItem == WAIMapStorage::existingMapNames[n].c_str()); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(WAIMapStorage::existingMapNames[n].c_str(), isSelected))
                {
                    WAIMapStorage::currItem = WAIMapStorage::existingMapNames[n].c_str();
                    WAIMapStorage::currN    = n;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
    }

    if (ImGui::Button("Load map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        if (WAIMapStorage::currItem)
        {
            //load selected map
            cv::Mat cvOm            = cv::Mat(4, 4, CV_32F);
            string  selectedMapName = WAIMapStorage::existingMapNames[WAIMapStorage::currN];
            if (WAIMapStorage::loadMap(selectedMapName, _tracking, WAIOrbVocabulary::get(), true, cvOm))
            {
                SLMat4f om;
                om.setMatrix(cvOm.at<float>(0, 0),
                             -cvOm.at<float>(0, 1),
                             -cvOm.at<float>(0, 2),
                             cvOm.at<float>(0, 3),
                             cvOm.at<float>(1, 0),
                             -cvOm.at<float>(1, 1),
                             -cvOm.at<float>(1, 2),
                             cvOm.at<float>(1, 3),
                             cvOm.at<float>(2, 0),
                             -cvOm.at<float>(2, 1),
                             -cvOm.at<float>(2, 2),
                             cvOm.at<float>(2, 3),
                             cvOm.at<float>(3, 0),
                             -cvOm.at<float>(3, 1),
                             -cvOm.at<float>(3, 2),
                             cvOm.at<float>(3, 3));
                _node->om(om);
                ImGui::Text("Info: map loading successful!");
            }
            else
            {
                ImGui::Text("Info: map loading failed!");
            }
        }
    }
}
