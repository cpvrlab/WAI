#include <thread>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <DUtils/Random.h>

#include <WAIModeOrbSlam2DataOriented.h>
#include <WAIOrbExtraction.cpp>
#include <WAIOrbMatching.cpp>
#include <WAIOrbSlamInitialization.cpp>

#define ROTATION_HISTORY_LENGTH 30
#define MATCHER_DISTANCE_THRESHOLD_LOW 50
#define MATCHER_DISTANCE_THRESHOLD_HIGH 100

WAI::ModeOrbSlam2DataOriented::ModeOrbSlam2DataOriented(SensorCamera* camera)
  : _camera(camera)
{
    _pose = cv::Mat::eye(4, 4, CV_32F);

    r32 scaleFactor        = 1.2f;
    i32 pyramidScaleLevels = 8;
    i32 numberOfFeatures   = 2000; // TODO(jan): 2000 for initialization, 1000 otherwise
    i32 orbPatchSize       = 31;
    i32 orbHalfPatchSize   = 15;

    _state                                       = {};
    _state.status                                = OrbSlamStatus_Initializing;
    _state.imagePyramidStats.numberOfScaleLevels = pyramidScaleLevels;
    _state.orbOctTreePatchSize                   = orbPatchSize;
    _state.orbOctTreeHalfPatchSize               = orbHalfPatchSize;
    _state.edgeThreshold                         = 19;

    _state.fastFeatureConstraints.numberOfFeatures = numberOfFeatures;
    _state.fastFeatureConstraints.initialThreshold = 20;
    _state.fastFeatureConstraints.minimalThreshold = 7;

    const i32        npoints  = 512;
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(_state.pattern));

    _state.imagePyramidStats.scaleFactors.resize(pyramidScaleLevels);
    _state.imagePyramidStats.inverseScaleFactors.resize(pyramidScaleLevels);
    _state.imagePyramidStats.sigmaSquared.resize(pyramidScaleLevels);
    _state.imagePyramidStats.inverseSigmaSquared.resize(pyramidScaleLevels);
    _state.imagePyramidStats.numberOfFeaturesPerScaleLevel.resize(pyramidScaleLevels);

    _state.imagePyramidStats.scaleFactors[0]        = 1.0f;
    _state.imagePyramidStats.inverseScaleFactors[0] = 1.0f;
    _state.imagePyramidStats.sigmaSquared[0]        = 1.0f;
    _state.imagePyramidStats.inverseSigmaSquared[0] = 1.0f;

    for (i32 i = 1; i < pyramidScaleLevels; i++)
    {
        r32 sigma                                       = _state.imagePyramidStats.scaleFactors[i - 1] * scaleFactor;
        _state.imagePyramidStats.scaleFactors[i]        = sigma;
        _state.imagePyramidStats.sigmaSquared[i]        = sigma * sigma;
        _state.imagePyramidStats.inverseScaleFactors[i] = 1.0f / _state.imagePyramidStats.scaleFactors[i];
        _state.imagePyramidStats.inverseSigmaSquared[i] = 1.0f / (sigma * sigma);
    }

    r32 inverseScaleFactor            = 1.0f / scaleFactor;
    r32 numberOfFeaturesPerScaleLevel = numberOfFeatures * (1.0f - inverseScaleFactor) / (1.0f - pow((r64)inverseScaleFactor, (r64)pyramidScaleLevels));
    i32 sumFeatures                   = 0;
    for (i32 level = 0; level < pyramidScaleLevels - 1; level++)
    {
        _state.imagePyramidStats.numberOfFeaturesPerScaleLevel[level] = cvRound(numberOfFeaturesPerScaleLevel);
        sumFeatures += _state.imagePyramidStats.numberOfFeaturesPerScaleLevel[level];
        numberOfFeaturesPerScaleLevel *= inverseScaleFactor;
    }
    _state.imagePyramidStats.numberOfFeaturesPerScaleLevel[pyramidScaleLevels - 1] = std::max(numberOfFeatures - sumFeatures, 0);

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

static g2o::SE3Quat convertCvMatToG2OSE3Quat(const cv::Mat& mat)
{
    Eigen::Matrix<r64, 3, 3> R;
    R << mat.at<r32>(0, 0), mat.at<r32>(0, 1), mat.at<r32>(0, 2),
      mat.at<r32>(1, 0), mat.at<r32>(1, 1), mat.at<r32>(1, 2),
      mat.at<r32>(2, 0), mat.at<r32>(2, 1), mat.at<r32>(2, 2);

    Eigen::Matrix<r64, 3, 1> t(mat.at<r32>(0, 3), mat.at<r32>(1, 3), mat.at<r32>(2, 3));

    g2o::SE3Quat result = g2o::SE3Quat(R, t);

    return result;
}

static cv::Mat convertG2OSE3QuatToCvMat(const g2o::SE3Quat se3)
{
    Eigen::Matrix<r64, 4, 4> mat = se3.to_homogeneous_matrix();

    cv::Mat result = cv::Mat(4, 4, CV_32F);
    for (i32 i = 0; i < 4; i++)
    {
        for (i32 j = 0; j < 4; j++)
        {
            result.at<r32>(i, j) = (r32)mat(i, j);
        }
    }

    return result;
}

static Eigen::Matrix<r64, 3, 1> convertCvMatToEigenVector3D(const cv::Mat& mat)
{
    Eigen::Matrix<r64, 3, 1> result;

    result << mat.at<r32>(0), mat.at<r32>(1), mat.at<r32>(2);

    return result;
}

static cv::Mat convertEigenVector3DToCvMat(const Eigen::Matrix<r64, 3, 1>& vec3d)
{
    cv::Mat result = cv::Mat(3, 1, CV_32F);

    for (int i = 0; i < 3; i++)
    {
        result.at<r32>(i) = (r32)vec3d(i);
    }

    return result;
}

static void initializeKeyFrame(const ImagePyramidStats&      imagePyramidStats,
                               const FastFeatureConstraints& fastFeatureConstraints,
                               const i32                     edgeThreshold,
                               const cv::Mat&                cameraFrame,
                               const cv::Mat&                cameraMat,
                               const cv::Mat&                distortionMat,
                               const i32                     orbOctTreePatchSize,
                               const i32                     orbOctTreeHalfPatchSize,
                               const std::vector<i32>        umax,
                               const std::vector<cv::Point>  orbPattern,
                               const GridConstraints         gridConstraints,
                               KeyFrame*                     keyFrame)
{
    keyFrame->numberOfKeyPoints = 0;
    keyFrame->keyPoints.clear();
    keyFrame->undistortedKeyPoints.clear();

    for (i32 i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (i32 j = 0; j < FRAME_GRID_ROWS; j++)
        {
            keyFrame->keyPointIndexGrid[i][j].clear();
        }
    }

    std::vector<cv::Mat> imagePyramid;
    imagePyramid.resize(imagePyramidStats.numberOfScaleLevels);

    // Compute scaled images according to scale factors
    computeScalePyramid(cameraFrame,
                        imagePyramidStats,
                        edgeThreshold,
                        imagePyramid);

    PyramidOctTree octTree = {};
    computePyramidOctTree(imagePyramidStats,
                          imagePyramid,
                          edgeThreshold,
                          &octTree);

    // Compute key points, distributed in an evenly spaced grid
    // on every scale level
    std::vector<std::vector<cv::KeyPoint>> allKeyPoints;
    computeKeyPointsInOctTree(octTree,
                              imagePyramidStats,
                              imagePyramid,
                              fastFeatureConstraints,
                              edgeThreshold,
                              orbOctTreePatchSize,
                              orbOctTreeHalfPatchSize,
                              umax,
                              allKeyPoints);

    for (i32 level = 0; level < imagePyramidStats.numberOfScaleLevels; level++)
    {
        keyFrame->numberOfKeyPoints += (i32)allKeyPoints[level].size();
    }

    if (keyFrame->numberOfKeyPoints)
    {
        keyFrame->descriptors.create(keyFrame->numberOfKeyPoints, 32, CV_8U);
    }

    keyFrame->keyPoints.reserve(keyFrame->numberOfKeyPoints);
    keyFrame->mapPointIndices = std::vector<i32>(keyFrame->numberOfKeyPoints, -1);

    i32 offset = 0;
    for (i32 level = 0; level < imagePyramidStats.numberOfScaleLevels; level++)
    {
        i32                        tOffset           = level * 3;
        std::vector<cv::KeyPoint>& keyPointsForLevel = allKeyPoints[level];
        i32                        nkeypointsLevel   = (i32)keyPointsForLevel.size();

        if (nkeypointsLevel == 0) continue;

        // Preprocess the resized image
        cv::Mat workingMat = imagePyramid[level].clone();
        cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        // Compute the descriptors
        cv::Mat desc = keyFrame->descriptors.rowRange(offset, offset + nkeypointsLevel);

        for (size_t i = 0; i < keyPointsForLevel.size(); i++)
        {
            computeOrbDescriptor(keyPointsForLevel[i], cameraFrame, &orbPattern[0], desc.ptr((i32)i));
        }
        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0)
        {
            r32 scale = imagePyramidStats.scaleFactors[level];
            for (std::vector<cv::KeyPoint>::iterator keypoint    = keyPointsForLevel.begin(),
                                                     keypointEnd = keyPointsForLevel.end();
                 keypoint != keypointEnd;
                 keypoint++)
            {
                keypoint->pt *= scale;
            }
        }

        // Add the keypoints to the output
        keyFrame->keyPoints.insert(keyFrame->keyPoints.end(), keyPointsForLevel.begin(), keyPointsForLevel.end());
    }

    if (!keyFrame->numberOfKeyPoints)
    {
        return;
    }

    undistortKeyPoints(cameraMat,
                       distortionMat,
                       keyFrame->keyPoints,
                       keyFrame->numberOfKeyPoints,
                       keyFrame->undistortedKeyPoints);

    i32 nReserve = 0.5f * keyFrame->numberOfKeyPoints / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (u32 i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (u32 j = 0; j < FRAME_GRID_ROWS; j++)
        {
            keyFrame->keyPointIndexGrid[i][j].reserve(nReserve);
        }
    }

    for (i32 i = 0; i < keyFrame->numberOfKeyPoints; i++)
    {
        const cv::KeyPoint& kp = keyFrame->undistortedKeyPoints[i];

        i32    xPos, yPos;
        bool32 keyPointIsInGrid = calculateKeyPointGridCell(kp, gridConstraints, &xPos, &yPos);
        if (keyPointIsInGrid)
        {
            keyFrame->keyPointIndexGrid[xPos][yPos].push_back(i);
        }
    }

    // TODO(jan): 'retain image' functionality
}

void computeGridConstraints(const cv::Mat&   cameraFrame,
                            const cv::Mat&   cameraMat,
                            const cv::Mat&   distortionMat,
                            GridConstraints* gridConstraints)
{
    r32 minX, maxX, minY, maxY;

    if (distortionMat.at<r32>(0) != 0.0)
    {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<r32>(0, 0) = 0.0;
        mat.at<r32>(0, 1) = 0.0;
        mat.at<r32>(1, 0) = cameraFrame.cols;
        mat.at<r32>(1, 1) = 0.0;
        mat.at<r32>(2, 0) = 0.0;
        mat.at<r32>(2, 1) = cameraFrame.rows;
        mat.at<r32>(3, 0) = cameraFrame.cols;
        mat.at<r32>(3, 1) = cameraFrame.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, cameraMat, distortionMat, cv::Mat(), cameraMat);
        mat = mat.reshape(1);

        minX = (r32)std::min(mat.at<r32>(0, 0), mat.at<r32>(2, 0));
        maxX = (r32)std::max(mat.at<r32>(1, 0), mat.at<r32>(3, 0));
        minY = (r32)std::min(mat.at<r32>(0, 1), mat.at<r32>(1, 1));
        maxY = (r32)std::max(mat.at<r32>(2, 1), mat.at<r32>(3, 1));
    }
    else
    {
        minX = 0.0f;
        maxX = cameraFrame.cols;
        minY = 0.0f;
        maxY = cameraFrame.rows;
    }

    gridConstraints->minX                 = minX;
    gridConstraints->minY                 = minY;
    gridConstraints->invGridElementWidth  = static_cast<r32>(FRAME_GRID_COLS) / static_cast<r32>(maxX - minX);
    gridConstraints->invGridElementHeight = static_cast<r32>(FRAME_GRID_ROWS) / static_cast<r32>(maxY - minY);
}

void findInitializationMatches(const KeyFrame*                 referenceKeyFrame,
                               const KeyFrame*                 currentKeyFrame,
                               const std::vector<cv::Point2f>& previouslyMatchedKeyPoints,
                               const GridConstraints&          gridConstraints,
                               const r32                       shortestToSecondShortestDistanceRatio,
                               const bool32                    checkOrientation,
                               std::vector<i32>&               initializationMatches,
                               i32&                            numberOfMatches)
{
    std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
    {
        rotHist[i].reserve(500);
    }

    const r32 factor = 1.0f / ROTATION_HISTORY_LENGTH;

    std::vector<i32> matchesDistances(currentKeyFrame->numberOfKeyPoints, INT_MAX);
    std::vector<i32> matchesKeyPointIndices(currentKeyFrame->numberOfKeyPoints, -1);

    for (size_t i1 = 0, iend1 = referenceKeyFrame->numberOfKeyPoints;
         i1 < iend1;
         i1++)
    {
        cv::KeyPoint keyPointReferenceKeyFrame = referenceKeyFrame->undistortedKeyPoints[i1];

        i32 level1 = keyPointReferenceKeyFrame.octave;
        if (level1 > 0) continue;

        std::vector<size_t> keyPointIndicesCurrentFrame =
          getFeatureIndicesForArea(currentKeyFrame->numberOfKeyPoints,
                                   100,
                                   previouslyMatchedKeyPoints[i1].x,
                                   previouslyMatchedKeyPoints[i1].y,
                                   gridConstraints,
                                   level1,
                                   level1,
                                   currentKeyFrame->keyPointIndexGrid,
                                   currentKeyFrame->undistortedKeyPoints);

        if (keyPointIndicesCurrentFrame.empty()) continue;

        cv::Mat d1 = referenceKeyFrame->descriptors.row(i1);

        // smaller is better
        i32 shortestDist       = INT_MAX;
        i32 secondShortestDist = INT_MAX;
        i32 shortestDistId     = -1;

        for (std::vector<size_t>::iterator vit = keyPointIndicesCurrentFrame.begin();
             vit != keyPointIndicesCurrentFrame.end();
             vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = currentKeyFrame->descriptors.row(i2);

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
                    i32 previouslyMatchedKeyPointId                    = matchesKeyPointIndices[shortestDistId];
                    initializationMatches[previouslyMatchedKeyPointId] = -1;
                    numberOfMatches--;
                }

                initializationMatches[i1]              = shortestDistId;
                matchesKeyPointIndices[shortestDistId] = i1;
                matchesDistances[shortestDistId]       = shortestDist;
                numberOfMatches++;

                if (checkOrientation)
                {
                    r32 rot = referenceKeyFrame->undistortedKeyPoints[i1].angle - currentKeyFrame->undistortedKeyPoints[shortestDistId].angle;
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
                if (initializationMatches[idx1] >= 0)
                {
                    initializationMatches[idx1] = -1;
                    numberOfMatches--;
                }
            }
        }
    }
}

bool32 computeGeometricalModel(const KeyFrame*           referenceKeyFrame,
                               const KeyFrame*           currentKeyFrame,
                               const std::vector<i32>&   initializationMatches,
                               const cv::Mat&            cameraMat,
                               cv::Mat&                  rcw,
                               cv::Mat&                  tcw,
                               std::vector<bool32>&      keyPointTriangulatedFlags,
                               std::vector<cv::Point3f>& initialPoints)
{
    bool32 result = false;

    const i32 maxRansacIterations = 200;
    const r32 sigma               = 1.0f;

    std::vector<Match>  matches;
    std::vector<bool32> matched;

    matches.reserve(currentKeyFrame->undistortedKeyPoints.size());
    matched.resize(referenceKeyFrame->undistortedKeyPoints.size());
    for (size_t i = 0, iend = initializationMatches.size(); i < iend; i++)
    {
        if (initializationMatches[i] >= 0)
        {
            matches.push_back(std::make_pair(i, initializationMatches[i]));
            matched[i] = true;
        }
        else
        {
            matched[i] = false;
        }
    }

    const i32 N = matches.size();

    // Indices for minimum set selection
    std::vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    std::vector<size_t> vAvailableIndices;

    for (i32 i = 0; i < N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    std::vector<std::vector<size_t>> ransacSets = std::vector<std::vector<size_t>>(maxRansacIterations, std::vector<size_t>(8, 0));

    DUtils::Random::SeedRandOnce(0);

    for (i32 it = 0; it < maxRansacIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for (size_t j = 0; j < 8; j++)
        {
            i32 randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
            i32 idx   = vAvailableIndices[randi];

            ransacSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    std::vector<bool32> vbMatchesInliersH, vbMatchesInliersF;
    r32                 scoreHomography, scoreFundamental;
    cv::Mat             homography, fundamental;

    std::thread threadHomography(&findHomography,
                                 std::ref(matches),
                                 std::ref(referenceKeyFrame->undistortedKeyPoints),
                                 std::ref(currentKeyFrame->undistortedKeyPoints),
                                 maxRansacIterations,
                                 std::ref(ransacSets),
                                 sigma,
                                 std::ref(scoreHomography),
                                 std::ref(vbMatchesInliersH),
                                 std ::ref(homography));
    std::thread threadFundamental(&findFundamental,
                                  std::ref(matches),
                                  std::ref(referenceKeyFrame->undistortedKeyPoints),
                                  std::ref(currentKeyFrame->undistortedKeyPoints),
                                  maxRansacIterations,
                                  std::ref(ransacSets),
                                  sigma,
                                  std::ref(scoreFundamental),
                                  std::ref(vbMatchesInliersF),
                                  std::ref(fundamental));

    // Wait until both threads have finished
    threadHomography.join();
    threadFundamental.join();

    // Compute ratio of scores
    r32 ratioHomographyToFundamental = scoreHomography / (scoreHomography + scoreFundamental);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if (ratioHomographyToFundamental > 0.40)
    {
        result = reconstructHomography(matches,
                                       referenceKeyFrame->undistortedKeyPoints,
                                       currentKeyFrame->undistortedKeyPoints,
                                       referenceKeyFrame->descriptors,
                                       currentKeyFrame->descriptors,
                                       sigma,
                                       matched,
                                       homography,
                                       cameraMat,
                                       rcw,
                                       tcw,
                                       initialPoints,
                                       keyPointTriangulatedFlags,
                                       1.0,
                                       50);
        printf("Assuming homography (ratio: %f)\n", ratioHomographyToFundamental);
    }
    else
    {
        result = reconstructFundamental(matches,
                                        referenceKeyFrame->undistortedKeyPoints,
                                        currentKeyFrame->undistortedKeyPoints,
                                        referenceKeyFrame->descriptors,
                                        currentKeyFrame->descriptors,
                                        sigma,
                                        vbMatchesInliersF,
                                        fundamental,
                                        cameraMat,
                                        rcw,
                                        tcw,
                                        initialPoints,
                                        keyPointTriangulatedFlags,
                                        1.0,
                                        50);
        printf("Assuming fundamental (ratio: %f)\n", ratioHomographyToFundamental);
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

            if (!_state.keyFrameCount)
            {
                computeGridConstraints(cameraFrame,
                                       cameraMat,
                                       distortionMat,
                                       &_state.gridConstraints);

                _state.fx    = cameraMat.at<r32>(0, 0);
                _state.fy    = cameraMat.at<r32>(1, 1);
                _state.cx    = cameraMat.at<r32>(0, 2);
                _state.cy    = cameraMat.at<r32>(1, 2);
                _state.invfx = 1.0f / _state.fx;
                _state.invfy = 1.0f / _state.fy;

                KeyFrame referenceKeyFrame = {};

                initializeKeyFrame(_state.imagePyramidStats,
                                   _state.fastFeatureConstraints,
                                   _state.edgeThreshold,
                                   cameraFrame,
                                   cameraMat,
                                   distortionMat,
                                   _state.orbOctTreePatchSize,
                                   _state.orbOctTreeHalfPatchSize,
                                   _state.umax,
                                   _state.pattern,
                                   _state.gridConstraints,
                                   &referenceKeyFrame);

                if (referenceKeyFrame.numberOfKeyPoints > 100)
                {
                    _state.previouslyMatchedKeyPoints.resize(referenceKeyFrame.numberOfKeyPoints);
                    for (i32 i = 0; i < referenceKeyFrame.numberOfKeyPoints; i++)
                    {
                        _state.previouslyMatchedKeyPoints[i] = referenceKeyFrame.undistortedKeyPoints[i].pt;
                    }

                    _state.keyFrames.resize(100); // TODO(jan): totally arbitrary number
                    _state.keyFrames[0] = referenceKeyFrame;
                    _state.keyFrameCount++;
                }
            }
            else
            {
                KeyFrame currentKeyFrame = {};

                initializeKeyFrame(_state.imagePyramidStats,
                                   _state.fastFeatureConstraints,
                                   _state.edgeThreshold,
                                   cameraFrame,
                                   cameraMat,
                                   distortionMat,
                                   _state.orbOctTreePatchSize,
                                   _state.orbOctTreeHalfPatchSize,
                                   _state.umax,
                                   _state.pattern,
                                   _state.gridConstraints,
                                   &currentKeyFrame);

                if (currentKeyFrame.numberOfKeyPoints > 100)
                {
                    // NOTE(jan): initialization matches contains the index of the matched keypoint
                    // of the current keyframe for every keypoint of the reference keyframe or
                    // -1 if not matched.
                    // currentKeyFrame->keyPoints[initializationMatches[i]] is matched to referenceKeyFrame->keyPoints[i]
                    std::vector<i32> initializationMatches                 = std::vector<i32>(_state.keyFrames[0].numberOfKeyPoints, -1);
                    bool32           checkOrientation                      = true;
                    r32              shortestToSecondShortestDistanceRatio = 0.9f;

                    i32 numberOfMatches = 0;

                    findInitializationMatches(&_state.keyFrames[0],
                                              &currentKeyFrame,
                                              _state.previouslyMatchedKeyPoints,
                                              _state.gridConstraints,
                                              shortestToSecondShortestDistanceRatio,
                                              checkOrientation,
                                              initializationMatches,
                                              numberOfMatches);

                    // update prev matched
                    for (size_t i1 = 0, iend1 = initializationMatches.size();
                         i1 < iend1;
                         i1++)
                    {
                        if (initializationMatches[i1] >= 0)
                        {
                            _state.previouslyMatchedKeyPoints[i1] = currentKeyFrame.undistortedKeyPoints[initializationMatches[i1]].pt;
                        }
                    }

#if 0 // Draw keypoints in reference keyframe
                    for (u32 i = 0; i < _state.keyFrames[0].keyPoints.size(); i++)
                    {
                        cv::rectangle(_camera->getImageRGB(),
                                      _state.keyFrames[0].keyPoints[i].pt,
                                      cv::Point(_state.keyFrames[0].keyPoints[i].pt.x + 3, _state.keyFrames[0].keyPoints[i].pt.y + 3),
                                      cv::Scalar(0, 0, 255));
                    }
#endif

                    //ghm1: decorate image with tracked matches
                    for (u32 i = 0; i < initializationMatches.size(); i++)
                    {
                        if (initializationMatches[i] >= 0)
                        {
                            cv::line(_camera->getImageRGB(),
                                     _state.keyFrames[0].keyPoints[i].pt,
                                     currentKeyFrame.keyPoints[initializationMatches[i]].pt,
                                     cv::Scalar(0, 255, 0));
                        }
                    }

                    // Check if there are enough matches
                    if (numberOfMatches >= 100)
                    {
                        cv::Mat                  rcw;                       // Current Camera Rotation
                        cv::Mat                  tcw;                       // Current Camera Translation
                        std::vector<bool32>      keyPointTriangulatedFlags; // Triangulated Correspondences (mvIniMatches)
                        std::vector<cv::Point3f> initialPoints;

                        bool32 validModelFound = computeGeometricalModel(&_state.keyFrames[0],
                                                                         &currentKeyFrame,
                                                                         initializationMatches,
                                                                         cameraMat,
                                                                         rcw,
                                                                         tcw,
                                                                         keyPointTriangulatedFlags,
                                                                         initialPoints);

                        if (validModelFound)
                        {
                            i32 mapPointIndex = 0;
                            _state.mapPoints.clear();
                            for (size_t i = 0, iend = initializationMatches.size(); i < iend; i++)
                            {
                                if ((initializationMatches[i] >= 0 && !keyPointTriangulatedFlags[i]) ||
                                    initializationMatches[i] < 0)
                                {
                                    continue;
                                }

                                MapPoint mapPoint     = {};
                                mapPoint.normalVector = cv::Mat::zeros(3, 1, CV_32F);
                                mapPoint.position     = cv::Mat(initialPoints[i]);
                                mapPoint.observations.resize(2);

                                mapPoint.observations[0].first  = 0;
                                mapPoint.observations[0].second = i;
                                mapPoint.observations[1].first  = 1;
                                mapPoint.observations[1].second = initializationMatches[i];

                                _state.keyFrames[0].mapPointIndices[i]                    = mapPointIndex;
                                currentKeyFrame.mapPointIndices[initializationMatches[i]] = mapPointIndex;

                                mapPointIndex++;

                                _state.mapPoints.push_back(mapPoint);
                            }

                            _state.keyFrames[0].cTw = cv::Mat::eye(4, 4, CV_32F);
                            currentKeyFrame.cTw     = cv::Mat::eye(4, 4, CV_32F);
                            rcw.copyTo(currentKeyFrame.cTw.rowRange(0, 3).colRange(0, 3));
                            tcw.copyTo(currentKeyFrame.cTw.rowRange(0, 3).col(3));

                            _state.keyFrameCount++;
                            _state.keyFrames[1] = currentKeyFrame;

                            printf("New Map created with %i points\n", _state.mapPoints.size());

#if 1
                            { // (Global) bundle adjustment
                                const i32 numberOfIterations = 20;

                                std::vector<bool32> vbNotIncludedMP;
                                vbNotIncludedMP.resize(_state.mapPoints.size());

                                g2o::SparseOptimizer                    optimizer;
                                g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

                                linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

                                g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

                                g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
                                optimizer.setAlgorithm(solver);

                                u64 maxKFid = 0;

                                // Set WAIKeyFrame vertices
                                for (i32 i = 0; i < _state.keyFrameCount; i++)
                                {
                                    KeyFrame* keyFrame = &_state.keyFrames[i];

                                    // TODO(jan): bad check

                                    g2o::SE3Quat se3Quat = convertCvMatToG2OSE3Quat(keyFrame->cTw);

                                    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
                                    vSE3->setEstimate(se3Quat);
                                    vSE3->setId(i);
                                    vSE3->setFixed(i == 0);
                                    optimizer.addVertex(vSE3);
                                }

                                const r32 thHuber2D = sqrt(5.99);

                                for (i32 i = 0; i < _state.mapPoints.size(); i++)
                                {
                                    MapPoint* mapPoint = &_state.mapPoints[i];

                                    // TODO(jan): bad check

                                    Eigen::Matrix<r64, 3, 1> vec3d = convertCvMatToEigenVector3D(mapPoint->position);

                                    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
                                    vPoint->setEstimate(vec3d);
                                    const i32 id = i + _state.keyFrameCount;
                                    vPoint->setId(id);
                                    vPoint->setMarginalized(true);
                                    optimizer.addVertex(vPoint);

                                    const std::vector<std::pair<i32, i32>> observations = mapPoint->observations;

                                    i32 nEdges = 0;

                                    //SET EDGES
                                    for (i32 j = 0; j < observations.size(); j++)
                                    {
                                        std::pair<i32, i32> observation = observations[j];

                                        i32       keyFrameIndex = observation.first;
                                        KeyFrame* keyFrame      = &_state.keyFrames[keyFrameIndex];

                                        //if (pKF->isBad() || pKF->mnId > maxKFid)
                                        //continue;

                                        nEdges++;

                                        const cv::KeyPoint& kpUn = keyFrame->undistortedKeyPoints[observation.second];

                                        Eigen::Matrix<r64, 2, 1> obs;
                                        obs << kpUn.pt.x, kpUn.pt.y;

                                        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(keyFrameIndex)));
                                        e->setMeasurement(obs);
                                        const float& invSigma2 = _state.imagePyramidStats.inverseSigmaSquared[kpUn.octave];
                                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                                        //if (bRobust)
                                        //{
                                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                        e->setRobustKernel(rk);
                                        rk->setDelta(thHuber2D);
                                        //}

                                        e->fx = _state.fx;
                                        e->fy = _state.fy;
                                        e->cx = _state.cx;
                                        e->cy = _state.cy;

                                        optimizer.addEdge(e);
                                    }

                                    if (nEdges == 0)
                                    {
                                        optimizer.removeVertex(vPoint);
                                        vbNotIncludedMP[i] = true;
                                    }
                                    else
                                    {
                                        vbNotIncludedMP[i] = false;
                                    }
                                }

                                // Optimize!
                                optimizer.initializeOptimization();
                                optimizer.optimize(numberOfIterations);

                                // Recover optimized data

                                for (i32 i = 0; i < _state.keyFrames.size(); i++)
                                {
                                    KeyFrame* keyFrame = &_state.keyFrames[i];

                                    //if (pKF->isBad())
                                    //continue;

                                    g2o::VertexSE3Expmap* vSE3    = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
                                    g2o::SE3Quat          se3quat = vSE3->estimate();

                                    keyFrame->cTw = convertG2OSE3QuatToCvMat(se3quat);

                                    /*if (nLoopKF == 0)
                                    {
                                    pKF->SetPose(Converter::toCvMat(SE3quat));
                                    }
                                    else
                                    {
                                        pKF->mTcwGBA.create(4, 4, CV_32F);
                                        Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
                                        pKF->mnBAGlobalForKF = nLoopKF;
                                    }*/
                                }

                                for (size_t i = 0; i < _state.mapPoints.size(); i++)
                                {
                                    if (vbNotIncludedMP[i]) continue;

                                    MapPoint* mapPoint = &_state.mapPoints[i];

                                    //if (pMP->isBad())
                                    //continue;

                                    g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i + _state.keyFrameCount));
                                    mapPoint->position             = convertEigenVector3DToCvMat(vPoint->estimate());

                                    // TODO(jan): update normal and depth

                                    /*if (nLoopKF == 0)
                                    {
                                        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                                        pMP->UpdateNormalAndDepth();
                                    }
                                    else
                                    {
                                        pMP->mPosGBA.create(3, 1, CV_32F);
                                        Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                                        pMP->mnBAGlobalForKF = nLoopKF;
                                    }*/
                                }
                            }
#endif

                            r32 medianDepth;
                            // WAIKeyFrame->ComputeSceneMedianDepth
                            {
                                std::vector<i32> mapPointIndices = _state.keyFrames[0].mapPointIndices;

                                std::vector<r32> depths;
                                depths.reserve(mapPointIndices.size());
                                cv::Mat Rcw2 = _state.keyFrames[0].cTw.row(2).colRange(0, 3);
                                Rcw2         = Rcw2.t();
                                r32 zcw      = _state.keyFrames[0].cTw.at<r32>(2, 3);
                                for (i32 i = 0; i < mapPointIndices.size(); i++)
                                {
                                    if (mapPointIndices[i] < 0) continue;

                                    MapPoint* mapPoint = &_state.mapPoints[mapPointIndices[i]];
                                    cv::Mat   x3Dw     = mapPoint->position;
                                    r32       z        = Rcw2.dot(x3Dw) + zcw;
                                    depths.push_back(z);
                                }

                                std::sort(depths.begin(), depths.end());

                                medianDepth = depths[(depths.size() - 1) / 2];
                            }

                            // TODO(jan): is the check for tracked map points necessary,
                            // as we have the same check already higher up?
                            i32 trackedMapPoints;
                            // WAIKeyFrame->TrackedMapPoints
                            {
                                std::vector<i32> mapPointIndices = _state.keyFrames[0].mapPointIndices;
                                for (i32 i = 0; i < mapPointIndices.size(); i++)
                                {
                                    if (mapPointIndices[i] < 0) continue;

                                    MapPoint* mapPoint = &_state.mapPoints[mapPointIndices[i]];

                                    // TODO(jan): check mapPoints->isBad
                                    if (mapPoint->observations.size() > 0)
                                    {
                                        trackedMapPoints++;
                                    }
                                }
                            }

                            r32 invMedianDepth = 1.0f / medianDepth;

                            if (medianDepth > 0.0f && trackedMapPoints >= 100)
                            {
                                cv::Mat scaledPose               = currentKeyFrame.cTw;
                                scaledPose.col(3).rowRange(0, 3) = scaledPose.col(3).rowRange(0, 3) * invMedianDepth;
                                currentKeyFrame.cTw              = scaledPose;

                                for (i32 i = 0; i < _state.mapPoints.size(); i++)
                                {
                                    MapPoint* mapPoint = &_state.mapPoints[i];
                                    mapPoint->position *= invMedianDepth;
                                }

                                // TODO(jan): add keyframes to local mapper
                                // TODO(jan): save stuff for camera trajectory

                                _state.status = OrbSlamStatus_Tracking;
                            }
                            else
                            {
                                _state.keyFrameCount = 0;
                            }
                        }
                    }
                    else
                    {
                        _state.keyFrameCount = 0;
                    }
                }
                else
                {
                    _state.keyFrameCount = 0;
                }
            }
        }
        break;

        case OrbSlamStatus_Tracking:
        {
            cv::Mat cameraMat     = _camera->getCameraMatrix();
            cv::Mat distortionMat = _camera->getDistortionMatrix();
            cv::Mat cameraFrame   = _camera->getImageGray();

            KeyFrame currentFrame = {};
            initializeKeyFrame(_state.imagePyramidStats,
                               _state.fastFeatureConstraints,
                               _state.edgeThreshold,
                               cameraFrame,
                               cameraMat,
                               distortionMat,
                               _state.orbOctTreePatchSize,
                               _state.orbOctTreeHalfPatchSize,
                               _state.umax,
                               _state.pattern,
                               _state.gridConstraints,
                               &currentFrame);

            KeyFrame* referenceKeyFrame = &_state.keyFrames[1];

            i32 matchCount = 0;
            {
                std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
                for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
                {
                    rotHist[i].reserve(500);
                }
                const r32 factor = 1.0f / ROTATION_HISTORY_LENGTH;

                std::vector<bool32> mapPointMatched      = std::vector<bool32>(referenceKeyFrame->mapPointIndices.size(), 0);
                std::vector<i32>    matchedMapPointIndex = std::vector<i32>(currentFrame.keyPoints.size(), -1);

                for (i32 i = 0; i < currentFrame.undistortedKeyPoints.size(); i++)
                {
                    i32 bestMatchIndex = -1;
                    i32 bestDist       = INT_MAX;
                    i32 secondBestDist = INT_MAX;
                    for (i32 j = 0; j < referenceKeyFrame->mapPointIndices.size(); j++)
                    {
                        if (referenceKeyFrame->mapPointIndices[j] < 0) continue;
                        if (mapPointMatched[j]) continue;

                        i32 dist = descriptorDistance(referenceKeyFrame->descriptors.row(j),
                                                      currentFrame.descriptors.row(i));

                        if (dist < bestDist)
                        {
                            bestMatchIndex = j;
                            secondBestDist = bestDist;
                            bestDist       = dist;
                        }
                    }

                    if (bestDist <= MATCHER_DISTANCE_THRESHOLD_LOW)
                    {
                        if ((r32)bestDist < 0.75f * (r32)secondBestDist)
                        {
                            mapPointMatched[bestMatchIndex] = true;
                            matchedMapPointIndex[i]         = bestMatchIndex;

                            // compute orientation
                            {
                                r32 rot = currentFrame.keyPoints[i].angle - referenceKeyFrame->keyPoints[bestMatchIndex].angle;
                                if (rot < 0.0)
                                {
                                    rot += 360.0f;
                                }

                                i32 bin = round(rot * factor);
                                if (bin == ROTATION_HISTORY_LENGTH)
                                {
                                    bin = 0;
                                }

                                assert(bin >= 0 && bin < ROTATION_HISTORY_LENGTH);
                                rotHist[bin].push_back(bestMatchIndex);
                            }

                            matchCount++;
                        }
                    }
                }

                i32 ind1 = -1;
                i32 ind2 = -1;
                i32 ind3 = -1;

                computeThreeMaxima(rotHist, ROTATION_HISTORY_LENGTH, ind1, ind2, ind3);

                // check orientation
                {
                    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
                    {
                        if (i == ind1 || i == ind2 || i == ind3) continue;

                        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                        {
                            mapPointMatched[rotHist[i][j]] = false;
                            matchCount--;
                        }
                    }
                }

                for (i32 i = 0; i < matchedMapPointIndex.size(); i++)
                {
                    i32 mapPointIndexReferenceKeyframe = matchedMapPointIndex[i];

                    if (mapPointMatched[mapPointIndexReferenceKeyframe])
                    {
                        currentFrame.mapPointIndices[i] = referenceKeyFrame->mapPointIndices[mapPointIndexReferenceKeyframe];
                    }
                }

                currentFrame.mapPointIsOutlier = std::vector<bool32>(currentFrame.keyPoints.size(), false);
            }

            currentFrame.cTw   = referenceKeyFrame->cTw.clone();
            i32 goodMatchCount = 0;

            // pose otimization
            {
                //ghm1: Attention, we add every map point associated to a keypoint to the optimizer
                g2o::SparseOptimizer                    optimizer;
                g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

                linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

                g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

                g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
                optimizer.setAlgorithm(solver);

                i32 nInitialCorrespondences = 0;

                g2o::SE3Quat se3quat = convertCvMatToG2OSE3Quat(currentFrame.cTw);

                // Set Frame vertex
                g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
                vSE3->setEstimate(se3quat);
                vSE3->setId(0);
                vSE3->setFixed(false);
                optimizer.addVertex(vSE3);

                // Set WAIMapPoint vertices
                const int N = currentFrame.numberOfKeyPoints;

                std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
                std::vector<size_t>                          vnIndexEdgeMono;
                vpEdgesMono.reserve(N);
                vnIndexEdgeMono.reserve(N);

                const float deltaMono = sqrt(5.991);

                {
                    for (i32 i = 0; i < currentFrame.mapPointIndices.size(); i++)
                    {
                        i32 mapPointIndex = currentFrame.mapPointIndices[i];

                        if (mapPointIndex >= 0)
                        {
                            MapPoint* mapPoint = &_state.mapPoints[mapPointIndex];

                            nInitialCorrespondences++;
                            currentFrame.mapPointIsOutlier[i] = false;

                            Eigen::Matrix<r64, 2, 1> obs;
                            const cv::KeyPoint&      kpUn = currentFrame.undistortedKeyPoints[i];
                            obs << kpUn.pt.x, kpUn.pt.y;

                            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                            e->setMeasurement(obs);
                            const r32 invSigma2 = _state.imagePyramidStats.inverseSigmaSquared[kpUn.octave];
                            e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(deltaMono);

                            e->fx      = _state.fx;
                            e->fy      = _state.fy;
                            e->cx      = _state.cx;
                            e->cy      = _state.cy;
                            cv::Mat Xw = mapPoint->position;
                            e->Xw[0]   = Xw.at<r32>(0);
                            e->Xw[1]   = Xw.at<r32>(1);
                            e->Xw[2]   = Xw.at<r32>(2);

                            optimizer.addEdge(e);

                            vpEdgesMono.push_back(e);
                            vnIndexEdgeMono.push_back(i);
                        }
                    }
                }

                if (nInitialCorrespondences > 3)
                {
                    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
                    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
                    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
                    const int   its[4]      = {10, 10, 10, 10};

                    int nBad = 0;
                    for (size_t it = 0; it < 4; it++)
                    {
                        se3quat = convertCvMatToG2OSE3Quat(currentFrame.cTw);
                        vSE3->setEstimate(se3quat);
                        optimizer.initializeOptimization(0);
                        optimizer.optimize(its[it]);

                        nBad = 0;
                        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
                        {
                            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

                            const size_t idx = vnIndexEdgeMono[i];

                            if (currentFrame.mapPointIsOutlier[idx])
                            {
                                e->computeError();
                            }

                            const float chi2 = e->chi2();

                            if (chi2 > chi2Mono[it])
                            {
                                currentFrame.mapPointIsOutlier[idx] = true;
                                e->setLevel(1);
                                nBad++;
                            }
                            else
                            {
                                currentFrame.mapPointIsOutlier[idx] = false;
                                e->setLevel(0);
                            }

                            if (it == 2)
                            {
                                e->setRobustKernel(0);
                            }
                        }

                        if (optimizer.edges().size() < 10) break;
                    }

                    // Recover optimized pose and return number of inliers
                    g2o::VertexSE3Expmap* vSE3_recov    = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
                    g2o::SE3Quat          SE3quat_recov = vSE3_recov->estimate();
                    cv::Mat               pose;

                    // Converter::toCvMat
                    {
                        Eigen::Matrix<double, 4, 4> eigMat = SE3quat_recov.to_homogeneous_matrix();
                        pose                               = cv::Mat(4, 4, CV_32F);
                        for (int i = 0; i < 4; i++)
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                pose.at<r32>(i, j) = (r32)eigMat(i, j);
                            }
                        }
                    }

                    currentFrame.cTw = pose;

                    goodMatchCount = nInitialCorrespondences - nBad;
                }
            }

            printf("Found %i good matches (out of %i)\n", goodMatchCount, matchCount);

            _pose = currentFrame.cTw.clone();

            // TODO(jan): track local map if tracking ok
        }
        break;
    }
}

std::vector<MapPoint> WAI::ModeOrbSlam2DataOriented::getMapPoints()
{
    std::vector<MapPoint> result = _state.mapPoints;

    return result;
}

bool WAI::ModeOrbSlam2DataOriented::getPose(cv::Mat* pose)
{
    *pose = _pose;

    return true;
}
