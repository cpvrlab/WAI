#ifndef SL_IMGUI_TRACKEDMAPPING_H
#define SL_IMGUI_TRACKEDMAPPING_H

#include <WAIModeOrbSlam2.h>.h>

#include <AppDemoGuiInfosDialog.h>

//-----------------------------------------------------------------------------
class AppDemoGuiTrackedMapping : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiTrackedMapping(std::string name, WAI::ModeOrbSlam2* orbSlamMode);

    void buildInfos() override;

    private:
    WAI::ModeOrbSlam2* _orbSlamMode = nullptr;

    //!currently selected combobox item
    static const char* _currItem;
    //!currently selected combobox index
    static int _currN;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H
