#include <WAIModeOrbSlam2DataOriented.h>
#include <WAIOrbExtraction.cpp>

#define ROTATION_HISTORY_LENGTH 30
#define MATCHER_DISTANCE_THRESHOLD_LOW 50
#define MATCHER_DISTANCE_THRESHOLD_HIGH 100

void computeThreeMaxima(std::vector<i32>* rotationHistory,
                        const i32         historyLength,
                        i32&              ind1,
                        i32&              ind2,
                        i32&              ind3)
{
    i32 max1 = 0;
    i32 max2 = 0;
    i32 max3 = 0;

    for (i32 i = 0; i < historyLength; i++)
    {
        const i32 s = rotationHistory[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (r32)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (r32)max1)
    {
        ind3 = -1;
    }
}

i32 descriptorDistance(const cv::Mat& a,
                       const cv::Mat& b)
{
    const i32* pa = a.ptr<int32_t>();
    const i32* pb = b.ptr<int32_t>();

    i32 dist = 0;

    for (i32 i = 0; i < 8; i++, pa++, pb++)
    {
        u32 v = *pa ^ *pb;
        v     = v - ((v >> 1) & 0x55555555);
        v     = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

bool32 calculateKeyPointGridCell(const cv::KeyPoint& keyPoint,
                                 const r32           minX,
                                 const r32           minY,
                                 const r32           invGridElementWidth,
                                 const r32           invGridElementHeight,
                                 i32*                posX,
                                 i32*                posY)
{
    bool32 result = false;

    i32 x = (i32)round((keyPoint.pt.x - minX) * invGridElementWidth);
    i32 y = (i32)round((keyPoint.pt.y - minY) * invGridElementHeight);

    // Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (x < 0 || x >= FRAME_GRID_COLS || y < 0 || y >= FRAME_GRID_ROWS)
    {
        result = false;
    }
    else
    {
        *posX  = x;
        *posY  = y;
        result = true;
    }

    return result;
}

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
    for (i32 i = 0; i < numberOfKeyPoints; i++)
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
    for (i32 i = 0; i < numberOfKeyPoints; i++)
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
    i32 numberOfFeatures   = 2000; // TODO(jan): 2000 for initialization, 1000 otherwise
    i32 orbPatchSize       = 31;
    i32 orbHalfPatchSize   = 15;

    _state.status                  = OrbSlamStatus_Initializing;
    _state.pyramidScaleLevels      = pyramidScaleLevels;
    _state.numberOfFeatures        = numberOfFeatures;
    _state.orbOctTreePatchSize     = orbPatchSize;
    _state.orbOctTreeHalfPatchSize = orbHalfPatchSize;
    _state.initialFastThreshold    = 20;
    _state.minimalFastThreshold    = 7;
    _state.edgeThreshold           = 19;

    const i32        npoints  = 512;
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

    r32 inverseScaleFactor            = 1.0f / scaleFactor;
    r32 numberOfFeaturesPerScaleLevel = numberOfFeatures * (1.0f - inverseScaleFactor) / (1.0f - pow((r64)inverseScaleFactor, (r64)pyramidScaleLevels));
    i32 sumFeatures                   = 0;
    for (i32 level = 0; level < pyramidScaleLevels - 1; level++)
    {
        _state.numberOfFeaturesPerScaleLevel[level] = cvRound(numberOfFeaturesPerScaleLevel);
        sumFeatures += _state.numberOfFeaturesPerScaleLevel[level];
        numberOfFeaturesPerScaleLevel *= inverseScaleFactor;
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

    _camera->subscribeToUpdate(this);
}

static void initializeKeyFrame(const OrbSlamState*        state,
                               const cv::Mat&             cameraFrame,
                               const cv::Mat&             cameraMat,
                               const cv::Mat&             distortionMat,
                               i32&                       numberOfKeyPoints,
                               std::vector<cv::KeyPoint>& keyPoints,
                               std::vector<cv::KeyPoint>& undistortedKeyPoints,
                               std::vector<size_t>        keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                               cv::Mat&                   descriptors)
{
    numberOfKeyPoints = 0;
    keyPoints.clear();
    undistortedKeyPoints.clear();

    for (i32 i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (i32 j = 0; j < FRAME_GRID_ROWS; j++)
        {
            keyPointIndexGrid[i][j].clear();
        }
    }

    std::vector<cv::Mat> imagePyramid;
    imagePyramid.resize(state->pyramidScaleLevels);

    // Compute scaled images according to scale factors
    computeScalePyramid(cameraFrame,
                        state->pyramidScaleLevels,
                        state->inversePyramidScaleFactors,
                        state->edgeThreshold,
                        imagePyramid);

    // Compute key points, distributed in an evenly spaced grid
    // on every scale level
    std::vector<std::vector<cv::KeyPoint>> allKeyPoints;
    computeKeyPointsInOctTree(state->pyramidScaleLevels,
                              imagePyramid,
                              state->edgeThreshold,
                              state->numberOfFeatures,
                              state->numberOfFeaturesPerScaleLevel,
                              state->initialFastThreshold,
                              state->minimalFastThreshold,
                              state->orbOctTreePatchSize,
                              state->orbOctTreeHalfPatchSize,
                              state->pyramidScaleFactors,
                              state->umax,
                              allKeyPoints);

    for (i32 level = 0; level < state->pyramidScaleLevels; level++)
    {
        numberOfKeyPoints += (i32)allKeyPoints[level].size();
    }

    if (numberOfKeyPoints)
    {
        descriptors.create(numberOfKeyPoints, 32, CV_8U);
    }

    keyPoints.reserve(numberOfKeyPoints);

    i32 offset = 0;
    for (i32 level = 0; level < state->pyramidScaleLevels; ++level)
    {
        i32                        tOffset           = level * 3;
        std::vector<cv::KeyPoint>& keyPointsForLevel = allKeyPoints[level];
        i32                        nkeypointsLevel   = (i32)keyPointsForLevel.size();

        if (nkeypointsLevel == 0) continue;

        // Preprocess the resized image
        cv::Mat workingMat = imagePyramid[level].clone();
        cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        // Compute the descriptors
        cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);

        for (size_t i = 0; i < keyPointsForLevel.size(); i++)
        {
            computeOrbDescriptor(keyPointsForLevel[i], cameraFrame, &state->pattern[0], desc.ptr((i32)i));
        }
        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0)
        {
            r32 scale = state->pyramidScaleFactors[level]; //getScale(level, firstLevel, scaleFactor);
            for (std::vector<cv::KeyPoint>::iterator keypoint    = keyPointsForLevel.begin(),
                                                     keypointEnd = keyPointsForLevel.end();
                 keypoint != keypointEnd;
                 keypoint++)
                keypoint->pt *= scale;
        }

        // Add the keypoints to the output
        keyPoints.insert(keyPoints.end(), keyPointsForLevel.begin(), keyPointsForLevel.end());
    }

    if (!numberOfKeyPoints)
    {
        return;
    }

    undistortKeyPoints(cameraMat,
                       distortionMat,
                       keyPoints,
                       numberOfKeyPoints,
                       undistortedKeyPoints);

    i32 nReserve = 0.5f * numberOfKeyPoints / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (u32 i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (u32 j = 0; j < FRAME_GRID_ROWS; j++)
        {
            keyPointIndexGrid[i][j].reserve(nReserve);
        }
    }

    for (i32 i = 0; i < numberOfKeyPoints; i++)
    {
        const cv::KeyPoint& kp = undistortedKeyPoints[i];

        i32    xPos, yPos;
        bool32 keyPointIsInGrid = calculateKeyPointGridCell(kp, state->minX, state->minY, state->invGridElementWidth, state->invGridElementWidth, &xPos, &yPos);
        if (keyPointIsInGrid)
        {
            keyPointIndexGrid[xPos][yPos].push_back(i);
        }
    }

    // TODO(jan): 'retain image' functionality
}

static std::vector<size_t> getFeatureIndicesForArea(const i32                       numberOfKeyPoints,
                                                    const r32                       searchWindowSize,
                                                    const r32                       x,
                                                    const r32                       y,
                                                    const r32                       minX,
                                                    const r32                       minY,
                                                    const r32                       invGridElementWidth,
                                                    const r32                       invGridElementHeight,
                                                    const i32                       minLevel,
                                                    const i32                       maxLevel,
                                                    const std::vector<size_t>       keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                                                    const std::vector<cv::KeyPoint> undistortedKeyPoints)
{
    std::vector<size_t> result;

    result.reserve(numberOfKeyPoints);

    const i32 nMinCellX = std::max(0, (i32)floor((x - minX - searchWindowSize) * invGridElementWidth));
    if (nMinCellX >= FRAME_GRID_COLS)
        return result;

    const i32 nMaxCellX = std::min((i32)FRAME_GRID_COLS - 1, (i32)ceil((x - minX + searchWindowSize) * invGridElementWidth));
    if (nMaxCellX < 0)
        return result;

    const i32 nMinCellY = std::max(0, (i32)floor((y - minY - searchWindowSize) * invGridElementHeight));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return result;

    const i32 nMaxCellY = std::min((i32)FRAME_GRID_ROWS - 1, (i32)ceil((y - minY + searchWindowSize) * invGridElementHeight));
    if (nMaxCellY < 0)
        return result;

    const bool32 checkLevels = (minLevel > 0) || (maxLevel >= 0);

    for (i32 ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (i32 iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const std::vector<size_t> vCell = keyPointIndexGrid[ix][iy];

            if (vCell.empty()) continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint& kpUn = undistortedKeyPoints[vCell[j]];
                if (checkLevels)
                {
                    if (kpUn.octave < minLevel) continue;
                    if (maxLevel >= 0 && kpUn.octave > maxLevel) continue;
                }

                const r32 distx = kpUn.pt.x - x;
                const r32 disty = kpUn.pt.y - y;

                if (fabs(distx) < searchWindowSize && fabs(disty) < searchWindowSize)
                {
                    result.push_back(vCell[j]);
                }
            }
        }
    }

    return result;
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

            if (!_state.referenceKeyFrame)
            {
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

                    _state.minX = (float)std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
                    _state.maxX = (float)std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
                    _state.minY = (float)std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
                    _state.maxY = (float)std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));
                }
                else
                {
                    _state.minX = 0.0f;
                    _state.maxX = cameraFrame.cols;
                    _state.minY = 0.0f;
                    _state.maxY = cameraFrame.rows;
                }

                _state.invGridElementWidth  = static_cast<r32>(FRAME_GRID_COLS) / static_cast<r32>(_state.maxX - _state.minX);
                _state.invGridElementHeight = static_cast<r32>(FRAME_GRID_ROWS) / static_cast<r32>(_state.maxY - _state.minY);

                _state.fx    = cameraMat.at<r32>(0, 0);
                _state.fy    = cameraMat.at<r32>(1, 1);
                _state.cx    = cameraMat.at<r32>(0, 2);
                _state.cy    = cameraMat.at<r32>(1, 2);
                _state.invfx = 1.0f / _state.fx;
                _state.invfy = 1.0f / _state.fy;

                _state.referenceKeyFrame = new KeyFrame();

                initializeKeyFrame(&_state,
                                   cameraFrame,
                                   cameraMat,
                                   distortionMat,
                                   _state.referenceKeyFrame->numberOfKeyPoints,
                                   _state.referenceKeyFrame->keyPoints,
                                   _state.referenceKeyFrame->undistortedKeyPoints,
                                   _state.referenceKeyFrame->keyPointIndexGrid,
                                   _state.referenceKeyFrame->descriptors);

                if (_state.referenceKeyFrame->numberOfKeyPoints <= 100)
                {
                    delete _state.referenceKeyFrame;
                    _state.referenceKeyFrame = nullptr;
                }
                else
                {
                    _state.previouslyMatchedKeyPoints.resize(_state.referenceKeyFrame->numberOfKeyPoints);
                    for (i32 i = 0; i < _state.referenceKeyFrame->numberOfKeyPoints; i++)
                    {
                        _state.previouslyMatchedKeyPoints[i] = _state.referenceKeyFrame->undistortedKeyPoints[i].pt;
                    }

                    std::fill(_state.initializationMatches.begin(), _state.initializationMatches.end(), -1);
                }
            }
            else
            {
                KeyFrame currentKeyFrame = {};

                initializeKeyFrame(&_state,
                                   cameraFrame,
                                   cameraMat,
                                   distortionMat,
                                   currentKeyFrame.numberOfKeyPoints,
                                   currentKeyFrame.keyPoints,
                                   currentKeyFrame.undistortedKeyPoints,
                                   currentKeyFrame.keyPointIndexGrid,
                                   currentKeyFrame.descriptors);

                if (currentKeyFrame.numberOfKeyPoints > 100)
                {
                    _state.initializationMatches                 = std::vector<i32>(_state.referenceKeyFrame->numberOfKeyPoints, -1);
                    bool32 checkOrientation                      = true;
                    r32    shortestToSecondShortestDistanceRatio = 0.9f;

                    i32 numberOfMatches = 0;

                    std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
                    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
                    {
                        rotHist[i].reserve(500);
                    }

                    const r32 factor = 1.0f / ROTATION_HISTORY_LENGTH;

                    std::vector<i32> matchesDistances(currentKeyFrame.numberOfKeyPoints, INT_MAX);
                    std::vector<i32> matchesKeyPointIndices(currentKeyFrame.numberOfKeyPoints, -1);

                    for (size_t i1 = 0, iend1 = _state.referenceKeyFrame->numberOfKeyPoints;
                         i1 < iend1;
                         i1++)
                    {
                        cv::KeyPoint keyPointReferenceKeyFrame = _state.referenceKeyFrame->undistortedKeyPoints[i1];

                        i32 level1 = keyPointReferenceKeyFrame.octave;
                        if (level1 > 0) continue;

                        std::vector<size_t> keyPointIndicesCurrentFrame =
                          getFeatureIndicesForArea(currentKeyFrame.numberOfKeyPoints,
                                                   100,
                                                   _state.previouslyMatchedKeyPoints[i1].x,
                                                   _state.previouslyMatchedKeyPoints[i1].y,
                                                   _state.minX,
                                                   _state.minY,
                                                   _state.invGridElementWidth,
                                                   _state.invGridElementHeight,
                                                   level1,
                                                   level1,
                                                   currentKeyFrame.keyPointIndexGrid,
                                                   currentKeyFrame.undistortedKeyPoints);

                        if (keyPointIndicesCurrentFrame.empty()) continue;

                        cv::Mat d1 = _state.referenceKeyFrame->descriptors.row(i1);

                        // smaller is better
                        i32 shortestDist       = INT_MAX;
                        i32 secondShortestDist = INT_MAX;
                        i32 shortestDistId     = -1;

                        for (std::vector<size_t>::iterator vit = keyPointIndicesCurrentFrame.begin();
                             vit != keyPointIndicesCurrentFrame.end();
                             vit++)
                        {
                            size_t i2 = *vit;

                            cv::Mat d2 = currentKeyFrame.descriptors.row(i2);

                            i32 dist = descriptorDistance(d1, d2);

                            if (matchesDistances[i2] <= dist) continue;

                            if (dist < shortestDist)
                            {
                                secondShortestDist = shortestDist;
                                shortestDist       = dist;
                                shortestDistId     = i2;
                            }
                            else if (dist < secondShortestDist)
                            {
                                secondShortestDist = dist;
                            }
                        }

                        if (shortestDist <= MATCHER_DISTANCE_THRESHOLD_LOW)
                        {
                            // test that shortest distance is unambiguous
                            if (shortestDist < shortestToSecondShortestDistanceRatio * (r32)secondShortestDist)
                            {
                                // delete previous match, if it exists
                                if (matchesKeyPointIndices[shortestDistId] >= 0)
                                {
                                    i32 previouslyMatchedKeyPointId                           = matchesKeyPointIndices[shortestDistId];
                                    _state.initializationMatches[previouslyMatchedKeyPointId] = -1;
                                    numberOfMatches--;
                                }

                                _state.initializationMatches[i1]       = shortestDistId;
                                matchesKeyPointIndices[shortestDistId] = i1;
                                matchesDistances[shortestDistId]       = shortestDist;
                                numberOfMatches++;

                                if (checkOrientation)
                                {
                                    r32 rot = _state.referenceKeyFrame->undistortedKeyPoints[i1].angle - currentKeyFrame.undistortedKeyPoints[shortestDistId].angle;
                                    if (rot < 0.0) rot += 360.0f;

                                    i32 bin = round(rot * factor);
                                    if (bin == ROTATION_HISTORY_LENGTH) bin = 0;

                                    assert(bin >= 0 && bin < ROTATION_HISTORY_LENGTH);

                                    rotHist[bin].push_back(i1);
                                }
                            }
                        }
                    }

                    if (checkOrientation)
                    {
                        i32 ind1 = -1;
                        i32 ind2 = -1;
                        i32 ind3 = -1;

                        computeThreeMaxima(rotHist, ROTATION_HISTORY_LENGTH, ind1, ind2, ind3);

                        for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
                        {
                            if (i == ind1 || i == ind2 || i == ind3) continue;

                            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                            {
                                i32 idx1 = rotHist[i][j];
                                if (_state.initializationMatches[idx1] >= 0)
                                {
                                    _state.initializationMatches[idx1] = -1;
                                    numberOfMatches--;
                                }
                            }
                        }
                    }

                    // update prev matched
                    for (size_t i1 = 0, iend1 = _state.initializationMatches.size();
                         i1 < iend1;
                         i1++)
                    {
                        if (_state.initializationMatches[i1] >= 0)
                        {
                            _state.previouslyMatchedKeyPoints[i1] = currentKeyFrame.undistortedKeyPoints[_state.initializationMatches[i1]].pt;
                        }
                    }

                    // Check if there are enough matches
                    if (numberOfMatches >= 100)
                    {
                        printf("Enough matches found\n");
                    }
                    else
                    {
                        delete _state.referenceKeyFrame;
                        _state.referenceKeyFrame = nullptr;
                    }

                    for (u32 i = 0; i < _state.referenceKeyFrame->keyPoints.size(); i++)
                    {
                        cv::rectangle(_camera->getImageRGB(),
                                      _state.referenceKeyFrame->keyPoints[i].pt,
                                      cv::Point(_state.referenceKeyFrame->keyPoints[i].pt.x + 3, _state.referenceKeyFrame->keyPoints[i].pt.y + 3),
                                      cv::Scalar(0, 0, 255));
                    }

                    //ghm1: decorate image with tracked matches
                    for (u32 i = 0; i < _state.initializationMatches.size(); i++)
                    {
                        if (_state.initializationMatches[i] >= 0)
                        {
                            cv::line(_camera->getImageRGB(),
                                     _state.referenceKeyFrame->keyPoints[i].pt,
                                     currentKeyFrame.keyPoints[_state.initializationMatches[i]].pt,
                                     cv::Scalar(0, 255, 0));
                        }
                    }
                }
                else
                {
                    delete _state.referenceKeyFrame;
                    _state.referenceKeyFrame = nullptr;
                }
            }
        }
        break;
    }
}
