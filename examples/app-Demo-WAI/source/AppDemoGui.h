//#############################################################################
//  File:      AppDemoGui.h
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGUIDEMO_H
#define SLGUIDEMO_H

#include <SL.h>
#include <string>
#include <map>
#include <memory>
#include <AppDemoGuiInfosDialog.h>

class SLScene;
class SLSceneView;
class SLNode;
class SLGLTexture;
class AppDemoGuiInfosDialog;

//-----------------------------------------------------------------------------
enum SceneID
{
    Scene_None,
    Scene_Empty,
    Scene_Minimal,
    Scene_WAI
};
//-----------------------------------------------------------------------------
//! ImGui UI class for the UI of the demo applications
/* The UI is completely built within this class by calling build function
AppDemoGui::build. This build function is passed in the slCreateSceneView and
it is called in SLSceneView::onPaint in every frame.<br>
The entire UI is configured and built on every frame. That is why it is called
"Im" for immediate. See also the SLGLImGui class to see how it minimaly
integrated in the SLProject.<br>
*/
class AppDemoGui
{
    public:
    static void build(SLScene* s, SLSceneView* sv);

    //!< Checks, if a dialog with this name already exists, and adds it if not
    static void addInfoDialog(const std::shared_ptr<AppDemoGuiInfosDialog>& dialog);
    static void clearInfoDialogs();
    static void buildMenuBar(SLScene* s, SLSceneView* sv);
    static void buildSceneGraph(SLScene* s);
    static void addSceneGraphNode(SLScene* s, SLNode* node);
    static void buildProperties(SLScene* s);
    static void buildStatsDebugTiming(SLScene* s, SLSceneView* sv);
    static void buildInfosDialogs();

    static void loadConfig(SLint dotsPerInch);
    static void saveConfig();

    static SLGLTexture* cpvrLogo;             //!< cpvr logo texture image
    static SLstring     configTime;           //!< Time of stored configuration
    static SLstring     infoAbout;            //!< About info string
    static SLstring     infoCredits;          //!< Credits info string
    static SLstring     infoHelp;             //!< Help info string
    static SLstring     infoCalibrate;        //!< Calibration info string
    static SLbool       showAbout;            //!< Flag if about info should be shown
    static SLbool       showHelp;             //!< Flag if help info should be shown
    static SLbool       showHelpCalibration;  //!< Flag if calibration info should be shown
    static SLbool       showCredits;          //!< Flag if credits info should be shown
    static SLbool       showStatsTiming;      //!< Flag if timing info should be shown
    static SLbool       showStatsScene;       //!< Flag if scene info should be shown
    static SLbool       showStatsVideo;       //!< Flag if video info should be shown
    static SLbool       showInfosSensors;     //!< Flag if device sensors info should be shown
    static SLbool       showInfosFrameworks;  //!< Flag if frameworks info should be shown
    static SLbool       showInfosScene;       //!< Flag if scene info should be shown
    static SLbool       showSceneGraph;       //!< Flag if scene graph should be shown
    static SLbool       showProperties;       //!< Flag if properties should be shown
    static SLbool       showChristoffel;      //!< Flag if Christoffel infos should be shown
    static SLbool       showStatsDebugTiming; //!< Flag if tracking info should be shown
    static SLbool       showUIPrefs;          //!< Flag if UI preferences
    static SLbool       showTransform;        //!< Flag if tranform dialog should be shown
    //! Vector containing all info dialogs, that belong to special scenes
    static std::map<std::string, std::shared_ptr<AppDemoGuiInfosDialog>> _infoDialogs;
};
//-----------------------------------------------------------------------------
#endif
