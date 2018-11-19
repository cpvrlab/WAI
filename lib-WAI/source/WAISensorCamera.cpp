#include <WAISensorCamera.h>

WAI::SensorCamera::SensorCamera(CameraCalibration* cameraCalibration)
{
    _cameraCalibration = *cameraCalibration;
}

void WAI::SensorCamera::update(void* imageGray)
{
    _imageGray = ((cv::Mat*)imageGray)->clone();
}
