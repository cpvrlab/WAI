#ifndef WAIAUTOCALIBRATION
#define WAIAUTOCALIBRATION
using namespace std;
#include <SLCV.h>
#include <WAICalibration.h>
#include <WAISensorCamera.h>

class AutoCalibration : public WAICalibration
{
    public:
    AutoCalibration(cv::Size imgSize);

    void reset();

    void  feed(std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> correspondances);
    float tryCalibrate();
    void  setCameraParameters(float fx, float fy, float cx, float cy, float k1 = 0, float k2 = 0, float p1 = 0, float p2 = 0);

    private:
    std::vector<std::vector<cv::Vec3f>> _points3d;
    std::vector<std::vector<cv::Vec2f>> _points2d;
};
#endif
