//#############################################################################
//  File:      AppDemoGuiMapStorage.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_MAPSTORAGE_H
#define SL_IMGUI_MAPSTORAGE_H

#include <AppDemoGuiInfosDialog.h>

#include <opencv2/core.hpp>

#include <WAIMap.h>
#include <WAIMapStorage.h>
#include <WAIOrbVocabulary.h>
#include <WAIModeOrbSlam2.h>

#include <SLMat4.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
class AppDemoGuiMapStorage : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiMapStorage(const std::string& name, WAI::ModeOrbSlam2* tracking, SLNode* mapNode, std::string mapDir);

    void buildInfos() override;

    private:
    WAIMap*                  _map;
    WAIKeyFrameDB*           _kfDB;
    WAI::ModeOrbSlam2*       _tracking;
    SLNode*                  _mapNode;
    std::string              _mapDir;
    std::string              _mapPrefix;
    std::vector<std::string> _existingMapNames;
    int                      _nextId;
    bool                     _saveAndLoadImages = true;
};

#endif //SL_IMGUI_MAPSTORAGE_H
