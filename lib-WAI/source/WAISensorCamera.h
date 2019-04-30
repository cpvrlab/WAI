#ifndef WAI_SENSOR_CAMERA_H
#define WAI_SENSOR_CAMERA_H

#include <WAIPlatform.h>
#include <WAIHelper.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <WAISensor.h>

namespace WAI
{
struct CameraFrame
{
    int   width;
    int   height;
    int   pitch;
    int   bytesPerPixel;
    void* memory;
};

struct CameraCalibration
{
    float fx, fy, cx, cy, k1, k2, p1, p2;
};

struct CameraData
{
    cv::Mat* imageGray;
    cv::Mat* imageRGB;
    i32      knownPoseProvided;
    cv::Mat* knownPose;
};

class SensorCamera : public Sensor
{
    public:
    SensorCamera(CameraCalibration* cameraCalibration);
    void              update(void* cameraData);
    cv::Mat           getImageGray() { return _imageGray; }
    cv::Mat           getImageRGB() { return _imageRGB; }
    CameraCalibration getCameraCalibration() { return _cameraCalibration; }
    cv::Mat           getCameraMatrix() { return _cameraMatrix; }
    cv::Mat           getDistortionMatrix() { return _distortionMatrix; }
    void              subscribeToUpdate(Mode* mode);

    bool32  knownPoseProvided() { return _knownPoseProvided; }
    cv::Mat knownPose() { return _knownFramePose; }

    private:
    cv::Mat _imageGray;
    cv::Mat _imageRGB;

    bool32  _knownPoseProvided;
    cv::Mat _knownFramePose;

    cv::Mat           _cameraMatrix;
    cv::Mat           _distortionMatrix;
    CameraCalibration _cameraCalibration;
    Mode*             _mode = 0;
};
}

#endif
