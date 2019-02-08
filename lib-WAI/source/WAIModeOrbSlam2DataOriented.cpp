#include <WAIModeOrbSlam2DataOriented.h>
#include <WAIOrbExtraction.cpp>

#define FRAME_GRID_ROWS 36 //48
#define FRAME_GRID_COLS 64

void undistortKeyPoints(const cv::Mat                   cameraMat,
                        const cv::Mat                   distortionCoefficients,
                        const std::vector<cv::KeyPoint> keyPoints,
                        const i32                       numberOfKeyPoints,
                        std::vector<cv::KeyPoint>&      undistortedKeyPoints)
{
    if (distortionCoefficients.at<r32>(0) == 0.0f)
    {
        undistortedKeyPoints = keyPoints;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(numberOfKeyPoints, 2, CV_32F);
    for (int i = 0; i < numberOfKeyPoints; i++)
    {
        mat.at<float>(i, 0) = keyPoints[i].pt.x;
        mat.at<float>(i, 1) = keyPoints[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, cameraMat, distortionCoefficients, cv::Mat(), cameraMat);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    undistortedKeyPoints.resize(numberOfKeyPoints);
    for (int i = 0; i < numberOfKeyPoints; i++)
    {
        cv::KeyPoint kp         = keyPoints[i];
        kp.pt.x                 = mat.at<float>(i, 0);
        kp.pt.y                 = mat.at<float>(i, 1);
        undistortedKeyPoints[i] = kp;
    }
}

void computeScalePyramid(const cv::Mat          image,
                         const i32              numberOfLevels,
                         const std::vector<r32> inverseScaleFactors,
                         const i32              edgeThreshold,
                         std::vector<cv::Mat>&  imagePyramid)
{
    for (i32 level = 0; level < numberOfLevels; level++)
    {
        r32      scale = inverseScaleFactors[level];
        cv::Size sz(cvRound((r32)image.cols * scale), cvRound((r32)image.rows * scale));
        cv::Size wholeSize(sz.width + edgeThreshold * 2, sz.height + edgeThreshold * 2);
        cv::Mat  temp(wholeSize, image.type()), masktemp;
        imagePyramid[level] = temp(cv::Rect(edgeThreshold, edgeThreshold, sz.width, sz.height));

        if (level)
        {
            resize(imagePyramid[level - 1],
                   imagePyramid[level],
                   sz,
                   0,
                   0,
                   CV_INTER_LINEAR);

            copyMakeBorder(imagePyramid[level],
                           temp,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image,
                           temp,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           cv::BORDER_REFLECT_101);
        }
    }
}

WAI::ModeOrbSlam2DataOriented::ModeOrbSlam2DataOriented(SensorCamera* camera)
  : _camera(camera)
{
    r32 scaleFactor        = 1.2f;
    i32 pyramidScaleLevels = 8;
    i32 numberOfFeatures   = 1000;
    i32 orbPatchSize       = 31;
    i32 orbHalfPatchSize   = 15;

    _state.status             = OrbSlamStatus_Initializing;
    _state.pyramidScaleLevels = pyramidScaleLevels;
    _state.numberOfFeatures   = numberOfFeatures;

    const int        npoints  = 512;
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(_state.pattern));

    _state.pyramidScaleFactors.resize(pyramidScaleLevels);
    _state.inversePyramidScaleFactors.resize(pyramidScaleLevels);
    _state.numberOfFeaturesPerScaleLevel.resize(pyramidScaleLevels);

    _state.pyramidScaleFactors[0]        = 1.0f;
    _state.inversePyramidScaleFactors[0] = 1.0f;

    for (i32 i = 1; i < _state.pyramidScaleLevels; i++)
    {
        _state.pyramidScaleFactors[i]        = _state.pyramidScaleFactors[i - 1] * scaleFactor;
        _state.inversePyramidScaleFactors[i] = 1.0f / _state.pyramidScaleFactors[i];
    }

    r32 numberOfFeaturesPerScaleLevel = numberOfFeatures * (1.0f - scaleFactor) / (1.0f - pow((r64)scaleFactor, (r64)pyramidScaleLevels));
    i32 sumFeatures                   = 0;
    for (i32 level = 0; level < pyramidScaleLevels - 1; level++)
    {
        _state.numberOfFeaturesPerScaleLevel[level] = cvRound(numberOfFeaturesPerScaleLevel);
        sumFeatures += _state.numberOfFeaturesPerScaleLevel[level];
        numberOfFeaturesPerScaleLevel *= scaleFactor;
    }
    _state.numberOfFeaturesPerScaleLevel[pyramidScaleLevels - 1] = std::max(numberOfFeatures - sumFeatures, 0);

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    _state.umax.resize(orbHalfPatchSize + 1);
    i32       v, v0;
    i32       vmax = cvFloor(orbHalfPatchSize * sqrt(2.f) / 2 + 1);
    i32       vmin = cvCeil(orbHalfPatchSize * sqrt(2.f) / 2);
    const r64 hp2  = orbHalfPatchSize * orbHalfPatchSize;
    for (v = 0; v <= vmax; v++)
    {
        _state.umax[v] = cvRound(sqrt(hp2 - v * v));
    }

    // Make sure we are symmetric
    for (v = orbHalfPatchSize, v0 = 0; v >= vmin; v--)
    {
        while (_state.umax[v0] == _state.umax[v0 + 1])
        {
            v0++;
        }

        _state.umax[v] = v0;
        v0++;
    }
}

void WAI::ModeOrbSlam2DataOriented::notifyUpdate()
{
    switch (_state.status)
    {
        case OrbSlamStatus_Initializing:
        {
            cv::Mat cameraMat     = _camera->getCameraMatrix();
            cv::Mat distortionMat = _camera->getDistortionMatrix();
            cv::Mat cameraFrame   = _camera->getImageGray();

            std::vector<cv::Mat> imagePyramid;

            computeScalePyramid(cameraFrame,
                                _state.pyramidScaleLevels,
                                _state.inversePyramidScaleFactors,
                                _state.edgeThreshold,
                                imagePyramid);

            std::vector<std::vector<cv::KeyPoint>> allKeyPoints;
            computeKeyPointsInOctTree(_state.pyramidScaleLevels,
                                      imagePyramid,
                                      _state.edgeThreshold,
                                      _state.numberOfFeatures,
                                      _state.numberOfFeaturesPerScaleLevel,
                                      _state.initialFastThreshold,
                                      _state.minimalFastThreshold,
                                      _state.orbOctTreePatchSize,
                                      _state.orbOctTreeHalfPatchSize,
                                      _state.pyramidScaleFactors,
                                      _state.umax,
                                      allKeyPoints);
            cv::Mat descriptors;

            i32 numberOfKeyPoints = 0;
            for (i32 level = 0; level < _state.pyramidScaleLevels; level++)
            {
                numberOfKeyPoints += (i32)allKeyPoints[level].size();
            }

            if (numberOfKeyPoints)
            {
                descriptors.create(numberOfKeyPoints, 32, CV_8U);
            }

            std::vector<cv::KeyPoint> keyPoints;
            keyPoints.reserve(numberOfKeyPoints);

            i32 offset = 0;
            for (i32 level = 0; level < _state.pyramidScaleLevels; ++level)
            {
                i32                        tOffset           = level * 3;
                std::vector<cv::KeyPoint>& keyPointsForLevel = allKeyPoints[level];
                i32                        nkeypointsLevel   = (i32)keyPointsForLevel.size();

                if (nkeypointsLevel == 0) continue;

                // preprocess the resized image
                cv::Mat workingMat = imagePyramid[level].clone();
                cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

                // Compute the descriptors
                cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);

                for (size_t i = 0; i < keyPointsForLevel.size(); i++)
                {
                    computeOrbDescriptor(keyPointsForLevel[i], cameraFrame, &_state.pattern[0], descriptors.ptr((i32)i));
                }
                offset += nkeypointsLevel;

                // Scale keypoint coordinates
                if (level != 0)
                {
                    r32 scale = _state.pyramidScaleFactors[level]; //getScale(level, firstLevel, scaleFactor);
                    for (std::vector<cv::KeyPoint>::iterator keypoint    = keyPointsForLevel.begin(),
                                                             keypointEnd = keyPointsForLevel.end();
                         keypoint != keypointEnd;
                         keypoint++)
                        keypoint->pt *= scale;
                }

                // And add the keypoints to the output
                keyPoints.insert(keyPoints.end(), keyPointsForLevel.begin(), keyPointsForLevel.end());
            }

            if (!numberOfKeyPoints)
            {
                return;
            }

            std::vector<cv::KeyPoint> undistortedKeyPoints;
            undistortKeyPoints(cameraMat,
                               distortionMat,
                               keyPoints,
                               numberOfKeyPoints,
                               undistortedKeyPoints);

            r32 maxX, maxY, minX, minY;
            if (distortionMat.at<r32>(0) != 0.0)
            {
                cv::Mat mat(4, 2, CV_32F);
                mat.at<float>(0, 0) = 0.0;
                mat.at<float>(0, 1) = 0.0;
                mat.at<float>(1, 0) = cameraFrame.cols;
                mat.at<float>(1, 1) = 0.0;
                mat.at<float>(2, 0) = 0.0;
                mat.at<float>(2, 1) = cameraFrame.rows;
                mat.at<float>(3, 0) = cameraFrame.cols;
                mat.at<float>(3, 1) = cameraFrame.rows;

                // Undistort corners
                mat = mat.reshape(2);
                cv::undistortPoints(mat, mat, cameraMat, distortionMat, cv::Mat(), cameraMat);
                mat = mat.reshape(1);

                minX = (float)std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
                maxX = (float)std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
                minY = (float)std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
                maxY = (float)std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));
            }
            else
            {
                minX = 0.0f;
                maxX = cameraFrame.cols;
                minY = 0.0f;
                maxY = cameraFrame.rows;
            }

            _state.invGridElementWidth  = static_cast<r32>(FRAME_GRID_COLS) / static_cast<r32>(maxX - minX);
            _state.invGridElementHeight = static_cast<r32>(FRAME_GRID_ROWS) / static_cast<r32>(maxY - minY);

            _state.fx    = cameraMat.at<r32>(0, 0);
            _state.fy    = cameraMat.at<r32>(1, 1);
            _state.cx    = cameraMat.at<r32>(0, 2);
            _state.cy    = cameraMat.at<r32>(1, 2);
            _state.invfx = 1.0f / _state.fx;
            _state.invfy = 1.0f / _state.fy;
        }
        break;
    }
}
