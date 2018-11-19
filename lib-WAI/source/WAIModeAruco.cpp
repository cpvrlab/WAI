#include <WAIModeAruco.h>

// TODO(jan): possibility to change edgeLength
WAI::ModeAruco::ModeAruco(SensorCamera* camera) : _camera(camera), _edgeLength(0.06f)
{
    // TODO(jan): possibility to choose aruco params
    _arucoParams                                        = cv::aruco::DetectorParameters::create();
    _arucoParams->adaptiveThreshWinSizeMin              = 4;
    _arucoParams->adaptiveThreshWinSizeMax              = 7;
    _arucoParams->adaptiveThreshWinSizeStep             = 1;
    _arucoParams->adaptiveThreshConstant                = 7;
    _arucoParams->minMarkerPerimeterRate                = 0.03;
    _arucoParams->maxMarkerPerimeterRate                = 4.0;
    _arucoParams->polygonalApproxAccuracyRate           = 0.05;
    _arucoParams->minDistanceToBorder                   = 3;
    _arucoParams->cornerRefinementMethod                = 1; //cv::aruco::CornerRefineMethod
    _arucoParams->cornerRefinementWinSize               = 5;
    _arucoParams->cornerRefinementMaxIterations         = 30;
    _arucoParams->cornerRefinementMinAccuracy           = 0.1;
    _arucoParams->markerBorderBits                      = 1;
    _arucoParams->perspectiveRemovePixelPerCell         = 8;
    _arucoParams->perspectiveRemoveIgnoredMarginPerCell = 0.13;
    _arucoParams->maxErroneousBitsInBorderRate          = 0.04;

    // TODO(jan): possibility to choose aruco dictionary
    _dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(0));
}

bool WAI::ModeAruco::getPose(M4x4* pose)
{
    bool result = false;

    std::vector<int>                      arucoIDs = std::vector<int>();
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    cv::aruco::detectMarkers(_camera->getImageGray(),
                             _dictionary,
                             corners,
                             arucoIDs,
                             _arucoParams,
                             rejected);

    if (!arucoIDs.empty())
    {
        SensorCamera::CameraCalibration cameraCalibration = _camera->getCameraCalibration();

        cv::Mat cameraMat         = cv::Mat::zeros(3, 3, CV_32F);
        cameraMat.at<float>(0, 0) = cameraCalibration.fx;
        cameraMat.at<float>(1, 1) = cameraCalibration.fy;
        cameraMat.at<float>(0, 2) = cameraCalibration.cx;
        cameraMat.at<float>(1, 2) = cameraCalibration.cy;
        cameraMat.at<float>(2, 2) = 1.0f;

        cv::Mat distortionMat         = cv::Mat::zeros(4, 1, CV_32F);
        distortionMat.at<float>(0, 0) = cameraCalibration.k1;
        distortionMat.at<float>(1, 0) = cameraCalibration.k2;
        distortionMat.at<float>(2, 0) = cameraCalibration.p1;
        distortionMat.at<float>(3, 0) = cameraCalibration.p2;

        std::vector<cv::Point3d> rVecs, tVecs;
        cv::aruco::estimatePoseSingleMarkers(corners,
                                             _edgeLength,
                                             cameraMat,
                                             distortionMat,
                                             rVecs,
                                             tVecs);

        // TODO(jan): what if we detect multiple markers?
        cv::Mat rotMat = cv::Mat::zeros(3, 3, CV_32F);
        cv::Rodrigues(cv::Mat(rVecs[0]), rotMat);

        float sign = 1.0f;
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                // Transpose rotation matrix and flip y- and z-axis
                pose->e[y][x] = sign * rotMat.at<float>(y, x);
            }

            sign = -1.0f;
        }

        pose->e[3][0] = tVecs[0].x;
        pose->e[3][1] = -1.0f * tVecs[0].x;
        pose->e[3][2] = -1.0f * tVecs[0].x;
        pose->e[3][3] = 1.0f;

        result = true;
    }

    return result;
}
