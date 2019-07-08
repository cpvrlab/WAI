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

#include <Utils.h>
#include <AppDemoGuiVideoStorage.h>
#include <AppWAISingleton.h>
#include "AppWAISingleton.h"

//-----------------------------------------------------------------------------
AppDemoGuiVideoStorage::AppDemoGuiVideoStorage(const std::string& name, std::string videoDir)
  : AppDemoGuiInfosDialog(name),
    _videoPrefix("video-"),
    _nextId(0)
{
    _videoDir = Utils::unifySlashes(videoDir);
    //_videoDir = videoDir;
    _currentItem = "";

    _existingVideoNames.clear();
    vector<pair<int, string>> existingVideoNamesSorted;

    //check if visual odometry maps directory exists
    if (!Utils::dirExists(_videoDir))
    {
        Utils::makeDir(_videoDir);
    }
    else
    {
        //parse content: we search for directories in mapsDir
        std::vector<std::string> content = Utils::getFileNamesInDir(_videoDir);
        for (auto path : content)
        {
            std::string name = Utils::getFileName(path);
            //find json files that contain mapPrefix and estimate highest used id
            if (Utils::containsString(name, _videoPrefix))
            {
                //estimate highest used id
                std::vector<std::string> splitted;
                Utils::splitString(name, '-', splitted);
                if (splitted.size())
                {
                    int id = atoi(splitted.back().c_str());
                    existingVideoNamesSorted.push_back(make_pair(id, name));
                    if (id >= _nextId)
                    {
                        _nextId = id + 1;
                    }
                }
            }
        }
    }

    //sort existingMapNames
    std::sort(existingVideoNamesSorted.begin(), existingVideoNamesSorted.end(), [](const pair<int, string>& left, const pair<int, string>& right) { return left.first < right.first; });
    for (auto it = existingVideoNamesSorted.begin(); it != existingVideoNamesSorted.end(); ++it)
        _existingVideoNames.push_back(it->second);
}
//-----------------------------------------------------------------------------
void AppDemoGuiVideoStorage::saveVideo(std::string filename)
{
    std::string infoDir = _videoDir + "info/";
    std::string infoPath = infoDir + filename;
    std::string path = _videoDir + filename;

    if (!Utils::dirExists(_videoDir))
    {
        Utils::makeDir(_videoDir);
    }
    else
    {
        if (Utils::fileExists(path))
        {
            Utils::deleteFile(path);
        }
    }

    if (!Utils::dirExists(infoDir))
    {
        Utils::makeDir(infoDir);
    }
    else
    {
        if (Utils::fileExists(infoPath))
        {
            Utils::deleteFile(infoPath);
        }
    }

    if (AppWAISingleton::instance()->videoWriter.isOpened())
    {
        AppWAISingleton::instance()->videoWriter.release();
    }
    if (AppWAISingleton::instance()->videoWriterInfo.isOpened())
    {
        AppWAISingleton::instance()->videoWriterInfo.release();
    }

    cv::Size size  = cv::Size(SLCVCapture::lastFrame.cols, SLCVCapture::lastFrame.rows);

    bool ret = AppWAISingleton::instance()->videoWriter.open(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);

    ret = AppWAISingleton::instance()->videoWriterInfo.open(infoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);
}
//-----------------------------------------------------------------------------
void AppDemoGuiVideoStorage::buildInfos()
{
    if (ImGui::Button("Start recording", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        std::string filename;
        if (_currentItem.empty())
        {
            filename = _videoPrefix + std::to_string(_nextId) + ".avi";
        }
        else
        {
            filename = _currentItem;
        }
        saveVideo(filename);
    }

    ImGui::Separator();

    if (ImGui::Button("Stop recording", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        AppWAISingleton::instance()->videoWriter.release();
    }

    ImGui::Separator();
    if (ImGui::Button("New video", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        _currentItem = "";
        _nextId = _nextId + 1;
        AppWAISingleton::instance()->videoWriter.release();
    }

    ImGui::Separator();
    {
        if (ImGui::BeginCombo("Current", _currentItem.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (int n = 0; n < _existingVideoNames.size(); n++)
            {
                bool isSelected = (_currentItem == _existingVideoNames[n]); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_existingVideoNames[n].c_str(), isSelected))
                {
                    _currentItem = _existingVideoNames[n];
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
    }
}
