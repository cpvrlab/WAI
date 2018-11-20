#ifndef WAI_SENSOR_CAMERA_H

#    include <opencv2/opencv.hpp>
#    include <opencv2/aruco.hpp>
#    include <WAISensor.h>

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

class SensorCamera : public Sensor
{
    public:
    SensorCamera(CameraCalibration* cameraCalibration);
    void              update(void* imageGray);
    cv::Mat           getImageGray() { return _imageGray; }
    CameraCalibration getCameraCalibration() { return _cameraCalibration; }
    cv::Mat           getCameraMatrix() { return _cameraMatrix; }
    cv::Mat           getDistortionMatrix() { return _distortionMatrix; }
    void              subscribeToUpdate(Mode* mode);

    private:
    cv::Mat           _imageGray;
    cv::Mat           _cameraMatrix;
    cv::Mat           _distortionMatrix;
    CameraCalibration _cameraCalibration;
    Mode*             _mode = 0;
};
}

#    define WAI_SENSOR_CAMERA_H
#endif
