#include <WAIAutoCalibration.h>
#include <WAICalibration.h>
using namespace cv;

AutoCalibration::AutoCalibration(cv::Size imgSize)
{
    _imgSize = imgSize;
    reset();
}

void AutoCalibration::setCameraParameters(float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2)
{
    _cameraMat  = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    _distortion = (Mat_<double>(5, 1) << k1, k2, p1, p2, 0);

    _cameraFovDeg = calcCameraFOV();
}

void AutoCalibration::reset()
{
    WAICalibration::reset();
    _points2d.clear();
    _points3d.clear();
}

void AutoCalibration::feed(std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> correspondances)
{
    _points3d.push_back(correspondances.first);
    _points2d.push_back(correspondances.second);
}

//! Calculates the reprojection error of the calibration
static double calcReprojectionErrors(const std::vector<std::vector<cv::Vec3f>>& objectPoints,
                                     const std::vector<std::vector<cv::Vec2f>>& imagePoints,
                                     const std::vector<cv::Mat>&                rvecs,
                                     const std::vector<cv::Mat>&                tvecs,
                                     const cv::Mat&                             cameraMatrix,
                                     const cv::Mat&                             distCoeffs)
{
    SLCVVPoint2f imagePoints2;
    size_t       totalPoints = 0;
    double       totalErr    = 0, err;

    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
        cv::projectPoints(objectPoints[i],
                          rvecs[i],
                          tvecs[i],
                          cameraMatrix,
                          distCoeffs,
                          imagePoints2);

        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

//----
float AutoCalibration::tryCalibrate()
{
    std::vector<cv::Mat> rvecs, tvecs;
    double               rms = cv::calibrateCamera(_points3d,
                                     _points2d,
                                     _imgSize,
                                     _cameraMat,
                                     _distortion,
                                     rvecs,
                                     tvecs,
                                     CALIB_USE_INTRINSIC_GUESS);

    float totalAvgErr = calcReprojectionErrors(_points3d,
                                               _points2d,
                                               rvecs,
                                               tvecs,
                                               _cameraMat,
                                               _distortion);

    _state = Calibrated;

    return totalAvgErr;
}
