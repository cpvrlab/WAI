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

class SensorCamera : public Sensor
{
    public:
    struct CameraCalibration
    {
        float fx, fy, cx, cy, k1, k2, p1, p2;
    };

    SensorCamera(CameraCalibration* cameraCalibration);
    void              update(void* imageGray);
    cv::Mat           getImageGray() { return _imageGray; }
    CameraCalibration getCameraCalibration() { return _cameraCalibration; }

    private:
    cv::Mat           _imageGray;         // TODO(jan): this will be loaded from a sensor class
    CameraCalibration _cameraCalibration; // TODO(jan): move to sensor class
};
}

#    define WAI_SENSOR_CAMERA_H
#endif
