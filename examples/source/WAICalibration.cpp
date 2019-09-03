#include <opencv2/core/core.hpp>
#include <iostream>
#include <WAICalibration.h>
#include <WAISensorCamera.h>

using namespace std;
using namespace cv;

WAICalibration::WAICalibration()
{
    _imgSize.width  = 640;
    _imgSize.height = 480;
    reset();
}

void WAICalibration::changeImageSize(int width, int height)
{
    _imgSize.width  = width;
    _imgSize.height = height;
    reset();
}

void WAICalibration::reset()
{
    float fov = 30;
    computeMatrix(_cameraMat, fov);
    // No distortion
    _distortion = (Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

    _cameraFovDeg = fov;
    _calibrationPath = std::string("");
    _state        = CalibrationState_Guess;
}

void WAICalibration::computeMatrix(cv::Mat& mat, float fov)
{
    float cx = (float)_imgSize.width * 0.5f;
    float cy = (float)_imgSize.height * 0.5f;
    float fy = cy / tanf(fov * 0.5f * M_PI / 180.0);
    float fx = fy;
    mat      = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}


std::string WAICalibration::stateStr()
{
    if (_state == CalibrationState_Guess)
        return std::string("Guess");
    else
        return std::string("Calibrated");
}

bool WAICalibration::loadFromFile(std::string path)
{
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened())
    {
        return false;
    }

    fs["imageSizeWidth"] >> _imgSize.width;
    fs["imageSizeHeight"] >> _imgSize.height;
    fs["cameraMat"] >> _cameraMat;
    fs["distortion"] >> _distortion;

    fs.release();
    _state = CalibrationState_Calibrated;

    float fov = calcCameraFOV();

    _calibrationPath = path;

    std::cout << "calibration file " << path << " loaded.    FOV = " << fov << std::endl;
    return true;
}

WAI::CameraCalibration WAICalibration::getCameraCalibration()
{
    WAI::CameraCalibration calibration = {fx(), fy(), cx(), cy(), k1(), k2(), p1(), p2()};
    return calibration;
}

float WAICalibration::calcCameraFOV()
{
    //calculate vertical field of view
    float fy     = _cameraMat.at<double>(1, 1);
    float cy     = _cameraMat.at<double>(1, 2);
    float fovRad = 2.0 * atan2(cy, fy);
    return fovRad * 180.0 / M_PI;
}

float WAICalibration::calcCameraFOV(cv::Mat& cameraMat)
{
    //calculate vertical field of view
    float fy     = cameraMat.at<double>(1, 1);
    float cy     = cameraMat.at<double>(1, 2);
    float fovRad = 2.0 * atan2(cy, fy);
    return fovRad * 180.0 / M_PI;
}
