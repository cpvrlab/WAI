#include <thread>

#include <DUtils/Random.h>

#include <WAIModeOrbSlam2DataOriented.h>
#include <WAIConverter.cpp>
#include <WAIOrbExtraction.cpp>
#include <WAIOrbMatching.cpp>
#include <WAIOrbSlamInitialization.cpp>

static void calculateNumberOfFeaturesPerScaleLevel(r32                scaleFactor,
                                                   i32                numberOfFeatures,
                                                   ImagePyramidStats* imagePyramidStats)
{
    r32 inverseScaleFactor            = 1.0f / scaleFactor;
    r32 numberOfFeaturesPerScaleLevel = numberOfFeatures * (1.0f - inverseScaleFactor) / (1.0f - pow((r64)inverseScaleFactor, (r64)imagePyramidStats->numberOfScaleLevels));
    i32 sumFeatures                   = 0;
    for (i32 level = 0; level < imagePyramidStats->numberOfScaleLevels - 1; level++)
    {
        imagePyramidStats->numberOfFeaturesPerScaleLevel[level] = cvRound(numberOfFeaturesPerScaleLevel);
        sumFeatures += imagePyramidStats->numberOfFeaturesPerScaleLevel[level];
        numberOfFeaturesPerScaleLevel *= inverseScaleFactor;
    }
    imagePyramidStats->numberOfFeaturesPerScaleLevel[imagePyramidStats->numberOfScaleLevels - 1] = std::max(numberOfFeatures - sumFeatures, 0);
}

WAI::ModeOrbSlam2DataOriented::ModeOrbSlam2DataOriented(SensorCamera* camera, std::string vocabularyPath)
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
    _state.scaleFactor                           = scaleFactor;

    _state.orbVocabulary    = new ORBVocabulary();
    bool32 vocabularyLoaded = _state.orbVocabulary->loadFromBinaryFile(vocabularyPath);
    if (!vocabularyLoaded)
    {
        printf("Path to ORBVocabulary %s not correct. Could not load vocabulary. Exiting.\n", vocabularyPath.c_str());
        exit(0);
    }

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

    calculateNumberOfFeaturesPerScaleLevel(scaleFactor, numberOfFeatures, &_state.imagePyramidStats);

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

static void computeBoW(const ORBVocabulary*  orbVocabulary,
                       const cv::Mat&        descriptors,
                       DBoW2::BowVector&     bowVector,
                       DBoW2::FeatureVector& featureVector)
{
    if (bowVector.empty() || featureVector.empty())
    {
        std::vector<cv::Mat> currentDescriptor = convertCvMatToDescriptorVector(descriptors);

        orbVocabulary->transform(currentDescriptor, bowVector, featureVector, 4);
    }
}

static void computeBoW(const ORBVocabulary* orbVocabulary,
                       KeyFrame*            keyFrame)
{
    computeBoW(orbVocabulary,
               keyFrame->descriptors,
               keyFrame->bowVector,
               keyFrame->featureVector);
}

static void setKeyFramePose(const cv::Mat& cTw,
                            cv::Mat&       cTwKF,
                            cv::Mat&       wTcKF,
                            cv::Mat&       owKF)
{
    cTw.copyTo(cTwKF);

    cv::Mat crw = cTw.rowRange(0, 3).colRange(0, 3);
    cv::Mat ctw = cTw.rowRange(0, 3).col(3);
    cv::Mat wrc = crw.t();

    owKF = -wrc * ctw;

    wTcKF = cv::Mat::eye(4, 4, cTw.type());
    wrc.copyTo(wTcKF.rowRange(0, 3).colRange(0, 3));
    owKF.copyTo(wTcKF.rowRange(0, 3).col(3));

    // TODO(jan): calculate world center, if needed
    //cv::Mat center = (cv::Mat_<r32>(4, 1) << mHalfBaseline, 0, 0, 1);
    //Cw             = Twc * center;
}

static std::vector<KeyFrame*> getBestCovisibilityKeyFrames(const i32                     maxNumberOfKeyFramesToGet,
                                                           const std::vector<KeyFrame*>& orderedConnectedKeyFrames)
{
    std::vector<KeyFrame*> result;

    if (orderedConnectedKeyFrames.size() < maxNumberOfKeyFramesToGet)
    {
        result = orderedConnectedKeyFrames;
    }
    else
    {
        result = std::vector<KeyFrame*>(orderedConnectedKeyFrames.begin(), orderedConnectedKeyFrames.begin() + maxNumberOfKeyFramesToGet);
    }

    return result;
}

static r32 computeSceneMedianDepthForKeyFrame(const KeyFrame* keyFrame)
{
    std::vector<MapPoint*> mapPoints = keyFrame->mapPointMatches;

    std::vector<r32> depths;
    depths.reserve(mapPoints.size());

    cv::Mat crw           = keyFrame->cTw.row(2).colRange(0, 3);
    cv::Mat wrc           = crw.t();
    r32     keyFrameDepth = keyFrame->cTw.at<r32>(2, 3);

    for (i32 i = 0; i < mapPoints.size(); i++)
    {
        if (!mapPoints[i]) continue;

        const MapPoint* mapPoint = mapPoints[i];
        cv::Mat         position = mapPoint->position;
        r32             depth    = wrc.dot(position) + keyFrameDepth;

        depths.push_back(depth);
    }

    std::sort(depths.begin(), depths.end());

    r32 result = depths[(depths.size() - 1) / 2];

    return result;
}

cv::Mat computeF12(const KeyFrame* keyFrame1,
                   const KeyFrame* keyFrame2,
                   const cv::Mat&  cameraMat)
{
    cv::Mat wr1 = getKeyFrameRotation(keyFrame1);
    cv::Mat wt1 = getKeyFrameTranslation(keyFrame1);
    cv::Mat wr2 = getKeyFrameRotation(keyFrame2);
    cv::Mat wt2 = getKeyFrameTranslation(keyFrame2);

    cv::Mat R12 = wr1 * wr2.t();
    cv::Mat t12 = -wr1 * wr2.t() * wt2 + wt1;

    cv::Mat t12x = (cv::Mat_<float>(3, 3) << 0.0f, -t12.at<float>(2), t12.at<float>(1), t12.at<float>(2), 0.0f, -t12.at<float>(0), -t12.at<float>(1), t12.at<float>(0), 0.0f); //SkewSymmetricMatrix(t12);

    const cv::Mat& K1 = cameraMat;
    const cv::Mat& K2 = cameraMat;

    cv::Mat result = K1.t().inv() * t12x * R12 * K2.inv();
    return result;
}

void calculateMapPointNormalAndDepth(const cv::Mat&            position,
                                     std::map<KeyFrame*, i32>& observations,
                                     KeyFrame*                 referenceKeyFrame,
                                     const std::vector<r32>    scaleFactors,
                                     const i32                 numberOfScaleLevels,
                                     r32*                      minDistance,
                                     r32*                      maxDistance,
                                     cv::Mat*                  normalVector)
{
    if (observations.empty()) return;

    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
    i32     n      = 0;
    for (std::map<KeyFrame*, i32>::iterator it = observations.begin(), itend = observations.end();
         it != itend;
         it++)
    {
        const KeyFrame* keyFrame = it->first;

        cv::Mat Owi     = getKeyFrameCameraCenter(keyFrame);
        cv::Mat normali = position - Owi;
        normal          = normal + normali / cv::norm(normali);
        n++;
    }

    cv::Mat   PC               = position - getKeyFrameCameraCenter(referenceKeyFrame);
    const r32 dist             = cv::norm(PC);
    const i32 level            = referenceKeyFrame->undistortedKeyPoints[observations[referenceKeyFrame]].octave;
    const r32 levelScaleFactor = scaleFactors[level];

    *maxDistance  = dist * levelScaleFactor;
    *minDistance  = *maxDistance / scaleFactors[numberOfScaleLevels - 1];
    *normalVector = normal / n;
}

static bool32 computeBestDescriptorFromObservations(const std::map<KeyFrame*, i32>& observations,
                                                    cv::Mat*                        descriptor)
{
    bool32 result = false;

    // Retrieve all observed descriptors
    std::vector<cv::Mat> descriptors;

    if (!observations.empty())
    {
        descriptors.reserve(observations.size());

        for (std::map<KeyFrame*, i32>::const_iterator mit = observations.begin(), mend = observations.end();
             mit != mend;
             mit++)
        {
            const KeyFrame* keyFrame = mit->first;

            //if (!pKF->isBad())
            descriptors.push_back(keyFrame->descriptors.row(mit->second));
        }

        if (!descriptors.empty())
        {
            // Compute distances between them
            const i32 descriptorsCount = descriptors.size();

            r32 distances[descriptorsCount][descriptorsCount];
            for (i32 i = 0; i < descriptorsCount; i++)
            {
                distances[i][i] = 0;
                for (i32 j = i + 1; j < descriptorsCount; j++)
                {
                    i32 distij      = descriptorDistance(descriptors[i], descriptors[j]);
                    distances[i][j] = distij;
                    distances[j][i] = distij;
                }
            }

            // Take the descriptor with least median distance to the rest
            i32 bestMedian = INT_MAX;
            i32 bestIndex  = 0;
            for (i32 i = 0; i < descriptorsCount; i++)
            {
                std::vector<i32> sortedDistances(distances[i], distances[i] + descriptorsCount);
                std::sort(sortedDistances.begin(), sortedDistances.end());
                i32 median = sortedDistances[0.5 * (descriptorsCount - 1)];

                if (median < bestMedian)
                {
                    bestMedian = median;
                    bestIndex  = i;
                }
            }

            *descriptor = descriptors[bestIndex].clone();

            result = true;
        }
    }

    return result;
}

static void updateKeyFrameOrderedCovisibilityVectors(std::map<KeyFrame*, i32>& connectedKeyFrameWeights,
                                                     std::vector<KeyFrame*>&   orderedConnectedKeyFrames,
                                                     std::vector<i32>&         orderedWeights)
{
    std::vector<std::pair<i32, KeyFrame*>> vPairs;

    vPairs.reserve(connectedKeyFrameWeights.size());
    for (std::map<KeyFrame*, i32>::iterator mit = connectedKeyFrameWeights.begin(), mend = connectedKeyFrameWeights.end();
         mit != mend;
         mit++)
    {
        vPairs.push_back(std::make_pair(mit->second, mit->first));
    }

    std::sort(vPairs.begin(), vPairs.end());

    std::list<KeyFrame*> connectedKeyFrames;
    std::list<i32>       weights;
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
    {
        connectedKeyFrames.push_front(vPairs[i].second);
        weights.push_front(vPairs[i].first);
    }

    orderedConnectedKeyFrames = std::vector<KeyFrame*>(connectedKeyFrames.begin(), connectedKeyFrames.end());
    orderedWeights            = std::vector<i32>(weights.begin(), weights.end());
}

static void addKeyFrameCovisibilityConnection(KeyFrame*                 keyFrame,
                                              const i32                 weight,
                                              std::map<KeyFrame*, i32>& connectedKeyFrameWeights,
                                              std::vector<KeyFrame*>&   orderedConnectedKeyFrames,
                                              std::vector<i32>&         orderedWeights)
{
    if (!connectedKeyFrameWeights.count(keyFrame))
    {
        connectedKeyFrameWeights[keyFrame] = weight;
    }
    else if (connectedKeyFrameWeights[keyFrame] != weight)
    {
        connectedKeyFrameWeights[keyFrame] = weight;
    }
    else
    {
        return;
    }

    updateKeyFrameOrderedCovisibilityVectors(connectedKeyFrameWeights, orderedConnectedKeyFrames, orderedWeights);
}

static void updateKeyFrameConnections(KeyFrame* keyFrame)
{
    std::map<KeyFrame*, i32> keyFrameCounter; // first is the index of the keyframe in keyframes, second is the number of common mapPoints

    //For all map points in keyframe check in which other keyframes are they seeing
    //Increase counter for those keyframes
    for (std::vector<MapPoint*>::const_iterator vit = keyFrame->mapPointMatches.begin(), vend = keyFrame->mapPointMatches.end();
         vit != vend;
         vit++)
    {
        MapPoint* mapPoint = *vit;

        if (!mapPoint) continue;
        if (mapPoint->bad) continue;

        std::map<KeyFrame*, i32> observations = mapPoint->observations;

        for (std::map<KeyFrame*, i32>::iterator mit = observations.begin(), mend = observations.end();
             mit != mend;
             mit++)
        {
            if (mit->first == keyFrame) continue;

            keyFrameCounter[mit->first]++;
        }
    }

    // This should not happen
    if (keyFrameCounter.empty()) return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    i32       maxCommonMapPointCount      = 0;
    KeyFrame* keyFrameWithMaxCommonPoints = nullptr;
    i32       threshold                   = 15;

    std::vector<std::pair<i32, KeyFrame*>> vPairs;
    vPairs.reserve(keyFrameCounter.size());
    for (std::map<KeyFrame*, i32>::iterator mit = keyFrameCounter.begin(), mend = keyFrameCounter.end();
         mit != mend;
         mit++)
    {
        KeyFrame* connectedKeyFrame = mit->first;
        i32       weight            = mit->second;

        if (weight > maxCommonMapPointCount)
        {
            maxCommonMapPointCount      = weight;
            keyFrameWithMaxCommonPoints = connectedKeyFrame;
        }
        if (weight >= threshold)
        {
            vPairs.push_back(std::make_pair(weight, connectedKeyFrame));

            addKeyFrameCovisibilityConnection(keyFrame,
                                              weight,
                                              connectedKeyFrame->connectedKeyFrameWeights,
                                              connectedKeyFrame->orderedConnectedKeyFrames,
                                              connectedKeyFrame->orderedWeights);
        }
    }

    if (vPairs.empty())
    {
        vPairs.push_back(std::make_pair(maxCommonMapPointCount, keyFrameWithMaxCommonPoints));

        KeyFrame* connectedKeyFrame = keyFrameWithMaxCommonPoints;

        addKeyFrameCovisibilityConnection(keyFrame,
                                          maxCommonMapPointCount,
                                          connectedKeyFrame->connectedKeyFrameWeights,
                                          connectedKeyFrame->orderedConnectedKeyFrames,
                                          connectedKeyFrame->orderedWeights);
    }

    sort(vPairs.begin(), vPairs.end());

    std::list<KeyFrame*> connectedKeyFrames;
    std::list<i32>       weights;
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
    {
        connectedKeyFrames.push_front(vPairs[i].second);
        weights.push_front(vPairs[i].first);
    }

    keyFrame->connectedKeyFrameWeights  = keyFrameCounter;
    keyFrame->orderedConnectedKeyFrames = std::vector<KeyFrame*>(connectedKeyFrames.begin(), connectedKeyFrames.end());
    keyFrame->orderedWeights            = std::vector<i32>(weights.begin(), weights.end());

    // TODO(jan): add child and parent connections
#if 0
    if (mbFirstConnection && mnId != 0)
    {
        mpParent = mvpOrderedConnectedKeyFrames.front();
        mpParent->AddChild(this);
        mbFirstConnection = false;
    }
#endif
}

static std::vector<MapPoint*> createNewMapPoints(KeyFrame*                     keyFrame,
                                                 const std::vector<KeyFrame*>& orderedConnectedKeyFrames,
                                                 const r32                     fx,
                                                 const r32                     fy,
                                                 const r32                     cx,
                                                 const r32                     cy,
                                                 const r32                     invfx,
                                                 const r32                     invfy,
                                                 const i32                     numberOfScaleLevels,
                                                 const r32                     scaleFactor,
                                                 const cv::Mat&                cameraMat,
                                                 const std::vector<r32>&       sigmaSquared,
                                                 const std::vector<r32>&       scaleFactors,
                                                 std::vector<MapPoint*>&       mapPoints,
                                                 i32*                          nextMapPointIndex)
{
    std::vector<MapPoint*> result = std::vector<MapPoint*>();

    const std::vector<KeyFrame*> neighboringKeyFrames = getBestCovisibilityKeyFrames(20,
                                                                                     orderedConnectedKeyFrames);

    cv::Mat crw1 = getKeyFrameRotation(keyFrame);
    cv::Mat wrc1 = crw1.t();
    cv::Mat ctw1 = getKeyFrameTranslation(keyFrame);
    cv::Mat cTw1(3, 4, CV_32F);

    crw1.copyTo(cTw1.colRange(0, 3));
    ctw1.copyTo(cTw1.col(3));
    cv::Mat origin1 = getKeyFrameCameraCenter(keyFrame);

    std::cout << origin1 << std::endl;

    const r32& fx1    = fx;
    const r32& fy1    = fy;
    const r32& cx1    = cx;
    const r32& cy1    = cy;
    const r32& invfx1 = invfx;
    const r32& invfy1 = invfy;

    const r32 ratioFactor = 1.5f * scaleFactor;

    // Search matches with epipolar restriction and triangulate
    for (i32 i = 0; i < neighboringKeyFrames.size(); i++)
    {
        // TODO(jan): reactivate
        //if (i > 0 && CheckNewKeyFrames()) return;

        KeyFrame* neighboringKeyFrame = neighboringKeyFrames[i];

        // Check first that baseline is not too short
        cv::Mat origin2   = getKeyFrameCameraCenter(neighboringKeyFrame);
        cv::Mat vBaseline = origin2 - origin1;

        std::cout << origin2 << std::endl;

        const r32 baseline = cv::norm(vBaseline);

        const r32 medianDepthNeighboringKeyFrame = computeSceneMedianDepthForKeyFrame(neighboringKeyFrame);
        const r32 ratioBaselineDepth             = baseline / medianDepthNeighboringKeyFrame;

        if (ratioBaselineDepth < 0.01) continue;

        // Compute Fundamental Matrix
        cv::Mat F12 = computeF12(keyFrame, neighboringKeyFrame, cameraMat);

        // Search matches that fullfil epipolar constraint
        std::vector<std::pair<size_t, size_t>> matchedIndices;
        searchMapPointMatchesForTriangulation(keyFrame, neighboringKeyFrame, fx, fy, cx, cy, sigmaSquared, scaleFactors, F12, false, matchedIndices);

        cv::Mat crw2 = getKeyFrameRotation(neighboringKeyFrame);
        cv::Mat wrc2 = crw2.t();
        cv::Mat ctw2 = getKeyFrameTranslation(neighboringKeyFrame);
        cv::Mat cTw2(3, 4, CV_32F);
        crw2.copyTo(cTw2.colRange(0, 3));
        ctw2.copyTo(cTw2.col(3));

        const r32& fx2    = fx;
        const r32& fy2    = fy;
        const r32& cx2    = cx;
        const r32& cy2    = cy;
        const r32& invfx2 = invfx;
        const r32& invfy2 = invfy;

        // Triangulate each match
        const i32 matchCount = matchedIndices.size();
        for (i32 matchIndex = 0; matchIndex < matchCount; matchIndex++)
        {
            const i32& keyPointIndex1 = matchedIndices[matchIndex].first;
            const i32& keyPointIndex2 = matchedIndices[matchIndex].second;

            const cv::KeyPoint& kp1 = keyFrame->undistortedKeyPoints[keyPointIndex1];
            const cv::KeyPoint& kp2 = neighboringKeyFrame->undistortedKeyPoints[keyPointIndex2];

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<r32>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<r32>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

            cv::Mat   ray1            = wrc1 * xn1;
            cv::Mat   ray2            = wrc2 * xn2;
            const r32 cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

            r32 cosParallaxStereo = cosParallaxRays + 1;

            cv::Mat x3D;
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && cosParallaxRays < 0.9998)
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<r32>(0) * cTw1.row(2) - cTw1.row(0);
                A.row(1) = xn1.at<r32>(1) * cTw1.row(2) - cTw1.row(1);
                A.row(2) = xn2.at<r32>(0) * cTw2.row(2) - cTw2.row(0);
                A.row(3) = xn2.at<r32>(1) * cTw2.row(2) - cTw2.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<r32>(3) == 0) continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0, 3) / x3D.at<r32>(3);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            r32 z1 = crw1.row(2).dot(x3Dt) + ctw1.at<r32>(2);
            if (z1 <= 0) continue;

            r32 z2 = crw2.row(2).dot(x3Dt) + ctw2.at<r32>(2);
            if (z2 <= 0) continue;

            //Check reprojection error in first keyframe
            const r32& sigmaSquare1 = sigmaSquared[kp1.octave];
            const r32  x1           = crw1.row(0).dot(x3Dt) + ctw1.at<r32>(0);
            const r32  y1           = crw1.row(1).dot(x3Dt) + ctw1.at<r32>(1);
            const r32  invz1        = 1.0 / z1;

            r32 u1    = fx1 * x1 * invz1 + cx1;
            r32 v1    = fy1 * y1 * invz1 + cy1;
            r32 errX1 = u1 - kp1.pt.x;
            r32 errY1 = v1 - kp1.pt.y;

            if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1) continue;

            //Check reprojection error in second keyframe
            const r32 sigmaSquare2 = sigmaSquared[kp2.octave];
            const r32 x2           = crw2.row(0).dot(x3Dt) + ctw2.at<r32>(0);
            const r32 y2           = crw2.row(1).dot(x3Dt) + ctw2.at<r32>(1);
            const r32 invz2        = 1.0 / z2;

            r32 u2    = fx2 * x2 * invz2 + cx2;
            r32 v2    = fy2 * y2 * invz2 + cy2;
            r32 errX2 = u2 - kp2.pt.x;
            r32 errY2 = v2 - kp2.pt.y;

            if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2) continue;

            //Check scale consistency
            cv::Mat normal1 = x3D - origin1;
            r32     dist1   = cv::norm(normal1);

            cv::Mat normal2 = x3D - origin2;
            r32     dist2   = cv::norm(normal2);

            if (dist1 == 0 || dist2 == 0) continue;

            const r32 ratioDist   = dist2 / dist1;
            const r32 ratioOctave = scaleFactors[kp1.octave] / scaleFactors[kp2.octave];

            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor) continue;

            // Triangulation is succesful
            MapPoint* mapPoint = new MapPoint();
            mapPoint->index    = *nextMapPointIndex;
            (*nextMapPointIndex)++;
            mapPoint->firstObservationKeyFrame = keyFrame;
            mapPoint->position                 = x3D;

            keyFrame->mapPointMatches[keyPointIndex1]            = mapPoint;
            neighboringKeyFrame->mapPointMatches[keyPointIndex2] = mapPoint;

            mapPoint->observations[keyFrame]            = keyPointIndex1;
            mapPoint->observations[neighboringKeyFrame] = keyPointIndex2;

            computeBestDescriptorFromObservations(mapPoint->observations,
                                                  &mapPoint->descriptor);
            calculateMapPointNormalAndDepth(mapPoint->position,
                                            mapPoint->observations,
                                            keyFrame,
                                            scaleFactors,
                                            numberOfScaleLevels,
                                            &mapPoint->minDistance,
                                            &mapPoint->maxDistance,
                                            &mapPoint->normalVector);

            mapPoints.push_back(mapPoint);
            result.push_back(mapPoint);
        }
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
                               KeyFrame*                     keyFrame,
                               i32                           index)
{
    keyFrame->numberOfKeyPoints = 0;
    keyFrame->keyPoints.clear();
    keyFrame->undistortedKeyPoints.clear();

    keyFrame->index = index;

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
    keyFrame->mapPointMatches   = std::vector<MapPoint*>(keyFrame->numberOfKeyPoints, nullptr);
    keyFrame->mapPointIsOutlier = std::vector<bool32>(keyFrame->numberOfKeyPoints, false);

    i32 offset = 0;
    for (i32 level = 0; level < imagePyramidStats.numberOfScaleLevels;
         level++)
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
    gridConstraints->maxX                 = maxX;
    gridConstraints->maxY                 = maxY;
    gridConstraints->invGridElementWidth  = static_cast<r32>(FRAME_GRID_COLS) / static_cast<r32>(maxX - minX);
    gridConstraints->invGridElementHeight = static_cast<r32>(FRAME_GRID_ROWS) / static_cast<r32>(maxY - minY);
}

i32 countMapPointsObservedByKeyFrame(const KeyFrame* keyFrame,
                                     const i32       minObservationsCount)
{
    i32 result = 0;

    const bool32 checkObservations = minObservationsCount > 0;

    if (checkObservations)
    {
        for (i32 i = 0; i < keyFrame->mapPointMatches.size(); i++)
        {
            MapPoint* mapPoint = keyFrame->mapPointMatches[i];

            if (mapPoint)
            {
                if (!mapPoint->bad)
                {
                    if (mapPoint->observations.size() >= minObservationsCount)
                    {
                        result++;
                    }
                }
            }
        }
    }
    else
    {
        for (i32 i = 0; i < keyFrame->mapPointMatches.size(); i++)
        {
            MapPoint* mapPoint = keyFrame->mapPointMatches[i];

            if (mapPoint)
            {
                if (!mapPoint->bad)
                {
                    result++;
                }
            }
        }
    }

    return result;
}

i32 searchMapPointsByProjection(std::vector<MapPoint*>& mapPoints,
                                const std::vector<r32>& scaleFactors,
                                const GridConstraints&  gridConstraints,
                                const i32               thresholdHigh,
                                const r32               bestToSecondBestRatio,
                                KeyFrame*               keyFrame,
                                i32                     threshold)
{
    //for every map point
    i32 result = 0;

    const bool32 factor = threshold != 1.0;

    for (i32 i = 0; i < mapPoints.size(); i++)
    {
        MapPoint*                    mapPoint     = mapPoints[i];
        const MapPointTrackingInfos* trackingInfo = &mapPoint->trackingInfos;

        if (!trackingInfo->inView) continue;
        if (mapPoint->bad) continue;

        const i32& predictedLevel = trackingInfo->scaleLevel;

        // The size of the window will depend on the viewing direction
        r32 r = (trackingInfo->viewCos > 0.998f) ? 2.5f : 4.0f;

        if (factor)
        {
            r *= (r32)threshold;
        }

        std::vector<size_t> indices =
          getFeatureIndicesForArea(keyFrame->numberOfKeyPoints,
                                   r * scaleFactors[predictedLevel],
                                   trackingInfo->projX,
                                   trackingInfo->projY,
                                   gridConstraints,
                                   predictedLevel - 1,
                                   predictedLevel,
                                   keyFrame->keyPointIndexGrid,
                                   keyFrame->undistortedKeyPoints);

        if (indices.empty()) continue;

        const cv::Mat descriptor1 = mapPoint->descriptor.clone();

        i32 bestDist   = 256;
        i32 bestLevel  = -1;
        i32 bestDist2  = 256;
        i32 bestLevel2 = -1;
        i32 bestIdx    = -1;

        // Get best and second matches with near keypoints
        for (std::vector<size_t>::const_iterator vit = indices.begin(), vend = indices.end();
             vit != vend;
             vit++)
        {
            const i32 idx      = *vit;
            MapPoint* mapPoint = keyFrame->mapPointMatches[idx];

            if (mapPoint && mapPoint->observations.size() > 0) continue;

            const cv::Mat& descriptor2 = keyFrame->descriptors.row(idx);

            const i32 dist = descriptorDistance(descriptor1, descriptor2);

            if (dist < bestDist)
            {
                bestDist2  = bestDist;
                bestDist   = dist;
                bestLevel2 = bestLevel;
                bestLevel  = keyFrame->undistortedKeyPoints[idx].octave;
                bestIdx    = idx;
            }
            else if (dist < bestDist2)
            {
                bestLevel2 = keyFrame->undistortedKeyPoints[idx].octave;
                bestDist2  = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if (bestDist <= thresholdHigh)
        {
            if (bestLevel == bestLevel2 && bestDist > bestToSecondBestRatio * bestDist2) continue;

            keyFrame->mapPointMatches[bestIdx] = mapPoint;
            result++;
        }
    }

    return result;
}

static bool32 needNewKeyFrame(const i32       currentFrameId,
                              const i32       lastKeyFrameId,
                              const i32       lastRelocalizationKeyFrameId,
                              const i32       minFramesBetweenKeyFrames,
                              const i32       maxFramesBetweenKeyFrames,
                              const i32       mapPointMatchCount,
                              const KeyFrame* referenceKeyFrame,
                              const i32       keyFrameCount)
{
    // TODO(jan): check if local mapper is stopped

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (currentFrameId < lastRelocalizationKeyFrameId + maxFramesBetweenKeyFrames &&
        keyFrameCount > maxFramesBetweenKeyFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    i32 minObservationsCount = 3;
    if (keyFrameCount <= 2)
    {
        minObservationsCount = 2;
    }

    i32 referenceMatchCount = countMapPointsObservedByKeyFrame(referenceKeyFrame, minObservationsCount);

// Local Mapping accept keyframes?
// TODO(jan): local mapping
#if 0
    bool32 localMappingIsIdle = mpLocalMapper->AcceptKeyFrames();
#else
    bool32 localMappingIsIdle = true;
#endif

    // Thresholds
    r32 thRefRatio = 0.9f;
    if (keyFrameCount < 2)
    {
        thRefRatio = 0.4f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool32 c1a = currentFrameId >= lastKeyFrameId + maxFramesBetweenKeyFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool32 c1b = (currentFrameId >= lastKeyFrameId + minFramesBetweenKeyFrames && localMappingIsIdle);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool32 c2 = ((mapPointMatchCount < referenceMatchCount * thRefRatio) && mapPointMatchCount > 15);

    if ((c1a || c1b) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (localMappingIsIdle)
        {
            std::cout << "[WAITrackedMapping] NeedNewKeyFrame: YES bLocalMappingIdle!" << std::endl;
            return true;
        }
        else
        {
            // TODO(jan): local mapping
            //mpLocalMapper->InterruptBA();
            std::cout << "[WAITrackedMapping] NeedNewKeyFrame: NO InterruptBA!" << std::endl;
            return false;
        }
    }
    else
    {
        printf("NeedNewKeyFrame: NO!\n");
        return false;
    }
}

static i32 predictMapPointScale(const r32& currentDist,
                                const r32  maxDistance,
                                const r32  frameScaleFactor,
                                const r32  numberOfScaleLevels)
{
    r32 ratio = maxDistance / currentDist;

    i32 result = ceil(log(ratio) / frameScaleFactor);

    if (result < 0)
    {
        result = 0;
    }
    else if (result >= numberOfScaleLevels)
    {
        result = numberOfScaleLevels - 1;
    }

    return result;
}

static bool32 isMapPointInFrameFrustum(const KeyFrame*        keyFrame,
                                       const r32              fx,
                                       const r32              fy,
                                       const r32              cx,
                                       const r32              cy,
                                       const r32              minX,
                                       const r32              maxX,
                                       const r32              minY,
                                       const r32              maxY,
                                       MapPoint*              mapPoint,
                                       r32                    viewingCosLimit,
                                       MapPointTrackingInfos* trackingInfos)
{
    //pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = mapPoint->position;

    // 3D in camera coordinates
    cv::Mat       crw = keyFrame->cTw.rowRange(0, 3).colRange(0, 3); // mRcw in WAIFrame
    cv::Mat       ctw = keyFrame->cTw.rowRange(0, 3).col(3);         // mtcw in WAIFrame
    const cv::Mat Pc  = crw * P + ctw;
    const r32&    PcX = Pc.at<r32>(0);
    const r32&    PcY = Pc.at<r32>(1);
    const r32&    PcZ = Pc.at<r32>(2);

    // Check positive depth
    if (PcZ < 0.0f) return false;

    // Project in image and check it is not outside
    const r32 invz = 1.0f / PcZ;
    const r32 u    = fx * PcX * invz + cx;
    const r32 v    = fy * PcY * invz + cy;

    if (u < minX || u > maxX) return false;
    if (v < minY || v > maxY) return false;

    // Check distance is in the scale invariance region of the WAIMapPoint
    // TODO(jan): magic numbers
    const r32     maxDistance = 1.2f * mapPoint->maxDistance;
    const r32     minDistance = 0.8f * mapPoint->minDistance;
    const cv::Mat PO          = P - (-crw.t() * ctw); // mOw in WAIFrame
    const r32     dist        = cv::norm(PO);

    if (dist < minDistance || dist > maxDistance) return false;

    // Check viewing angle
    cv::Mat Pn = mapPoint->normalVector;

    const r32 viewCos = PO.dot(Pn) / dist;

    if (viewCos < viewingCosLimit) return false;

    // Predict scale in the image
    // TODO(jan): magic numbers
    const i32 predictedLevel = predictMapPointScale(dist, maxDistance, 1.2f, 8);

    // Data used by the tracking
    trackingInfos->inView     = true;
    trackingInfos->projX      = u;
    trackingInfos->projY      = v;
    trackingInfos->scaleLevel = predictedLevel;
    trackingInfos->viewCos    = viewCos;

    return true;
}

// pose otimization
static i32 optimizePose(const std::vector<r32> inverseSigmaSquared,
                        const r32              fx,
                        const r32              fy,
                        const r32              cx,
                        const r32              cy,
                        KeyFrame*              frame)
{
    i32 result = 0;

    //ghm1: Attention, we add every map point associated to a keypoint to the optimizer
    g2o::SparseOptimizer                    optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    i32 initialCorrespondenceCount = 0;

    g2o::SE3Quat quat = convertCvMatToG2OSE3Quat(frame->cTw);

    // Set Frame vertex
    g2o::VertexSE3Expmap* vertex = new g2o::VertexSE3Expmap();
    vertex->setEstimate(quat);
    vertex->setId(0);
    vertex->setFixed(false);
    optimizer.addVertex(vertex);

    // Set WAIMapPoint vertices
    const i32 keyPointCount = frame->numberOfKeyPoints;

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edges;
    std::vector<size_t>                          edgeIndices;
    edges.reserve(keyPointCount);
    edgeIndices.reserve(keyPointCount);

    const r32 kernelDelta = sqrt(5.991);

    {
        for (i32 i = 0; i < frame->mapPointMatches.size(); i++)
        {
            MapPoint* mapPoint = frame->mapPointMatches[i];

            if (mapPoint)
            {
                initialCorrespondenceCount++;
                frame->mapPointIsOutlier[i] = false;

                Eigen::Matrix<r64, 2, 1> observationMatrix;
                const cv::KeyPoint&      undistortKeyPoint = frame->undistortedKeyPoints[i];
                observationMatrix << undistortKeyPoint.pt.x, undistortKeyPoint.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* edge = new g2o::EdgeSE3ProjectXYZOnlyPose();

                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                edge->setMeasurement(observationMatrix);
                const r32 invSigmaSquared = inverseSigmaSquared[undistortKeyPoint.octave];
                edge->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquared);

                g2o::RobustKernelHuber* kernel = new g2o::RobustKernelHuber;
                edge->setRobustKernel(kernel);
                kernel->setDelta(kernelDelta);

                edge->fx    = fx;
                edge->fy    = fy;
                edge->cx    = cx;
                edge->cy    = cy;
                cv::Mat Xw  = mapPoint->position;
                edge->Xw[0] = Xw.at<r32>(0);
                edge->Xw[1] = Xw.at<r32>(1);
                edge->Xw[2] = Xw.at<r32>(2);

                optimizer.addEdge(edge);

                edges.push_back(edge);
                edgeIndices.push_back(i);
            }
        }
    }

    if (initialCorrespondenceCount >= 3)
    {
        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const r32 chiSquared[4]      = {5.991, 5.991, 5.991, 5.991};
        const i32 iterationCounts[4] = {10, 10, 10, 10};

        i32 badMapPointCount = 0;
        for (i32 iteration = 0; iteration < 4; iteration++)
        {
            quat = convertCvMatToG2OSE3Quat(frame->cTw);
            vertex->setEstimate(quat);
            optimizer.initializeOptimization(0);
            optimizer.optimize(iterationCounts[iteration]);

            badMapPointCount = 0;
            for (i32 i = 0, iend = edges.size(); i < iend; i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* edge = edges[i];

                const i32 edgeIndex = edgeIndices[i];

                if (frame->mapPointIsOutlier[edgeIndex])
                {
                    edge->computeError();
                }

                const r32 chi2 = edge->chi2();

                if (chi2 > chiSquared[iteration])
                {
                    frame->mapPointIsOutlier[edgeIndex] = true;
                    edge->setLevel(1);
                    badMapPointCount++;
                }
                else
                {
                    frame->mapPointIsOutlier[edgeIndex] = false;
                    edge->setLevel(0);
                }

                if (iteration == 2)
                {
                    edge->setRobustKernel(0);
                }
            }

            if (optimizer.edges().size() < 10) break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap* vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        g2o::SE3Quat          quat   = vertex->estimate();
        cv::Mat               pose   = convertG2OSE3QuatToCvMat(quat);

        setKeyFramePose(pose,
                        frame->cTw,
                        frame->wTc,
                        frame->worldOrigin);
        result = initialCorrespondenceCount - badMapPointCount;
    }

    return result;
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

static void setMapPointToBad(MapPoint* mapPoint)
{
    mapPoint->bad = true;

    std::map<KeyFrame*, i32> observations = mapPoint->observations;
    mapPoint->observations.clear();

    for (std::map<KeyFrame*, i32>::iterator observationIterator = observations.begin(), observationsEnd = observations.end();
         observationIterator != observationsEnd;
         observationIterator++)
    {
        KeyFrame* keyFrame                                     = observationIterator->first;
        keyFrame->mapPointMatches[observationIterator->second] = nullptr;
    }
}

static void runLocalMapping(LocalMappingState*       localMapping,
                            i32                      keyFrameCount,
                            std::vector<MapPoint*>&  mapPoints,
                            const ImagePyramidStats& imagePyramidStats,
                            r32                      fx,
                            r32                      fy,
                            r32                      cx,
                            r32                      cy,
                            r32                      invfx,
                            r32                      invfy,
                            r32                      scaleFactor,
                            cv::Mat&                 cameraMat,
                            i32*                     nextMapPointIndex,
                            ORBVocabulary*           orbVocabulary)
{
    if (!localMapping->newKeyFrames.empty())
    {
        KeyFrame* keyFrame = localMapping->newKeyFrames.front();
        localMapping->newKeyFrames.pop_front();

        computeBoW(orbVocabulary, keyFrame);

        for (i32 i = 0; i < keyFrame->mapPointMatches.size(); i++)
        {
            MapPoint* mapPoint = keyFrame->mapPointMatches[i];

            if (mapPoint)
            {
                if (mapPoint->bad) continue;

                mapPoint->observations[keyFrame] = i;

                calculateMapPointNormalAndDepth(mapPoint->position,
                                                mapPoint->observations,
                                                keyFrame,
                                                imagePyramidStats.scaleFactors,
                                                imagePyramidStats.numberOfScaleLevels,
                                                &mapPoint->minDistance,
                                                &mapPoint->maxDistance,
                                                &mapPoint->normalVector);
                computeBestDescriptorFromObservations(mapPoint->observations,
                                                      &mapPoint->descriptor);
            }
        }

        updateKeyFrameConnections(keyFrame);

#if 0
        { // MapPointCulling
            std::vector<MapPoint*>::iterator newMapPointIterator = newMapPoints.begin();

            const i32 observationThreshold = 2;

            while (newMapPointIterator != newMapPoints.end())
            {
                MapPoint* mapPoint = *newMapPointIterator;

                if (mapPoint->bad)
                {
                    newMapPointIterator = newMapPoints.erase(newMapPointIterator);
                }
                else
                {
                    r32 foundRatio = (r32)(mapPoint->foundInKeyFrameCounter) / (r32)(mapPoint->visibleInKeyFrameCounter);
                    if (foundRatio < 0.25f)
                    {
                        setMapPointToBad(mapPoint);
                        newMapPointIterator = newMapPoints.erase(newMapPointIterator);
                    }
                    else if ((keyFrameCount - mapPoint->firstObservationKeyFrame->index) >= 2 && mapPoint->observations.size() <= observationThreshold)
                    {
                        setMapPointToBad(mapPoint);
                        newMapPointIterator = newMapPoints.erase(newMapPointIterator);
                    }
                    else if ((keyFrameCount - mapPoint->firstObservationKeyFrame->index) >= 3)
                    {
                        newMapPointIterator = newMapPoints.erase(newMapPointIterator);
                    }
                    else
                    {
                        newMapPointIterator++;
                    }
                }
            }
        }
#endif

        std::vector<MapPoint*> newMapPoints = createNewMapPoints(keyFrame,
                                                                 keyFrame->orderedConnectedKeyFrames,
                                                                 fx,
                                                                 fy,
                                                                 cx,
                                                                 cy,
                                                                 invfx,
                                                                 invfy,
                                                                 imagePyramidStats.numberOfScaleLevels,
                                                                 scaleFactor,
                                                                 cameraMat,
                                                                 imagePyramidStats.sigmaSquared,
                                                                 imagePyramidStats.scaleFactors,
                                                                 mapPoints,
                                                                 nextMapPointIndex);

        printf("Created %i new mapPoints\n", newMapPoints.size());
    }
}

static void addMapPointObservation(MapPoint* mapPoint,
                                   KeyFrame* observingKeyFrame,
                                   i32       keyPointIndex)
{
    if (mapPoint->observations.count(observingKeyFrame))
    {
        return;
    }

    mapPoint->observations[observingKeyFrame] = keyPointIndex;
}

void WAI::ModeOrbSlam2DataOriented::notifyUpdate()
{
    _state.frameCounter++;

    switch (_state.status)
    {
        case OrbSlamStatus_Initializing:
        {
            cv::Mat cameraMat     = _camera->getCameraMatrix();
            cv::Mat distortionMat = _camera->getDistortionMatrix();
            cv::Mat cameraFrame   = _camera->getImageGray();

            if (!_state.nextKeyFrameId)
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

                KeyFrame* referenceKeyFrame = new KeyFrame();

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
                                   referenceKeyFrame,
                                   _state.nextKeyFrameId);

                if (referenceKeyFrame->numberOfKeyPoints > 100)
                {
                    _state.previouslyMatchedKeyPoints.resize(referenceKeyFrame->numberOfKeyPoints);
                    for (i32 i = 0; i < referenceKeyFrame->numberOfKeyPoints; i++)
                    {
                        _state.previouslyMatchedKeyPoints[i] = referenceKeyFrame->undistortedKeyPoints[i].pt;
                    }

                    std::fill(_state.initializationMatches.begin(), _state.initializationMatches.end(), -1);

                    _state.keyFrames.push_back(referenceKeyFrame);
                    _state.nextKeyFrameId = 1;

                    printf("First initialization keyFrame at frame %i\n", _state.frameCounter);
                }
            }
            else
            {
                KeyFrame* currentKeyFrame = new KeyFrame();

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
                                   currentKeyFrame,
                                   _state.nextKeyFrameId);

                if (currentKeyFrame->numberOfKeyPoints > 100)
                {
                    // NOTE(jan): initialization matches contains the index of the matched keypoint
                    // of the current keyframe for every keypoint of the reference keyframe or
                    // -1 if not matched.
                    // currentKeyFrame->keyPoints[initializationMatches[i]] is matched to referenceKeyFrame->keyPoints[i]
                    _state.initializationMatches                 = std::vector<i32>(_state.keyFrames[0]->numberOfKeyPoints, -1);
                    bool32 checkOrientation                      = true;
                    r32    shortestToSecondShortestDistanceRatio = 0.9f;

                    i32 numberOfMatches = 0;

                    findInitializationMatches(_state.keyFrames[0],
                                              currentKeyFrame,
                                              _state.previouslyMatchedKeyPoints,
                                              _state.gridConstraints,
                                              shortestToSecondShortestDistanceRatio,
                                              checkOrientation,
                                              _state.initializationMatches,
                                              numberOfMatches);

                    // update prev matched
                    for (size_t i1 = 0, iend1 = _state.initializationMatches.size();
                         i1 < iend1;
                         i1++)
                    {
                        if (_state.initializationMatches[i1] >= 0)
                        {
                            _state.previouslyMatchedKeyPoints[i1] = currentKeyFrame->undistortedKeyPoints[_state.initializationMatches[i1]].pt;
                        }
                    }

                    //ghm1: decorate image with tracked matches
                    for (u32 i = 0; i < _state.initializationMatches.size(); i++)
                    {
                        if (_state.initializationMatches[i] >= 0)
                        {
                            cv::line(_camera->getImageRGB(),
                                     _state.keyFrames[0]->keyPoints[i].pt,
                                     currentKeyFrame->keyPoints[_state.initializationMatches[i]].pt,
                                     cv::Scalar(0, 255, 0));
                        }
                    }

                    // Check if there are enough matches
                    if (numberOfMatches >= 100)
                    {
                        cv::Mat                  crw;                       // Current Camera Rotation
                        cv::Mat                  ctw;                       // Current Camera Translation
                        std::vector<bool32>      keyPointTriangulatedFlags; // Triangulated Correspondences (mvIniMatches)
                        std::vector<cv::Point3f> initialPoints;

                        bool32 validModelFound = computeGeometricalModel(_state.keyFrames[0],
                                                                         currentKeyFrame,
                                                                         _state.initializationMatches,
                                                                         cameraMat,
                                                                         crw,
                                                                         ctw,
                                                                         keyPointTriangulatedFlags,
                                                                         initialPoints);

                        if (validModelFound)
                        {
                            for (i32 i = 0; i < _state.initializationMatches.size(); i++)
                            {
                                if (_state.initializationMatches[i] >= 0 && !keyPointTriangulatedFlags[i])
                                {
                                    _state.initializationMatches[i] = -1;
                                    numberOfMatches--;
                                }
                            }

                            setKeyFramePose(cv::Mat::eye(4, 4, CV_32F),
                                            _state.keyFrames[0]->cTw,
                                            _state.keyFrames[0]->wTc,
                                            _state.keyFrames[0]->worldOrigin);

                            cv::Mat cTw = cv::Mat::eye(4, 4, CV_32F);
                            crw.copyTo(cTw.rowRange(0, 3).colRange(0, 3));
                            ctw.copyTo(cTw.rowRange(0, 3).col(3));

                            setKeyFramePose(cTw,
                                            currentKeyFrame->cTw,
                                            currentKeyFrame->wTc,
                                            currentKeyFrame->worldOrigin);

                            computeBoW(_state.orbVocabulary,
                                       _state.keyFrames[0]);
                            computeBoW(_state.orbVocabulary,
                                       currentKeyFrame);

                            i32 mapPointIndex = 0;
                            _state.mapPoints.clear();

                            _state.nextKeyFrameId = 2;
                            _state.keyFrames.push_back(currentKeyFrame);

                            for (size_t i = 0, iend = _state.initializationMatches.size(); i < iend; i++)
                            {
                                if (_state.initializationMatches[i] < 0) continue;

                                MapPoint* mapPoint                 = new MapPoint();
                                mapPoint->index                    = _state.nextMapPointId;
                                mapPoint->firstObservationKeyFrame = currentKeyFrame;
                                mapPoint->normalVector             = cv::Mat::zeros(3, 1, CV_32F);

                                cv::Mat worldPosition(initialPoints[i]);
                                worldPosition.copyTo(mapPoint->position);

                                _state.nextMapPointId++;

                                _state.keyFrames[0]->mapPointMatches[i]                           = mapPoint;
                                currentKeyFrame->mapPointMatches[_state.initializationMatches[i]] = mapPoint;

                                addMapPointObservation(mapPoint, _state.keyFrames[0], i);
                                addMapPointObservation(mapPoint, currentKeyFrame, _state.initializationMatches[i]);

                                computeBestDescriptorFromObservations(mapPoint->observations,
                                                                      &mapPoint->descriptor);
                                calculateMapPointNormalAndDepth(mapPoint->position,
                                                                mapPoint->observations,
                                                                currentKeyFrame,
                                                                _state.imagePyramidStats.scaleFactors,
                                                                _state.imagePyramidStats.numberOfScaleLevels,
                                                                &mapPoint->minDistance,
                                                                &mapPoint->maxDistance,
                                                                &mapPoint->normalVector);

                                currentKeyFrame->mapPointIsOutlier[_state.initializationMatches[i]] = false;

                                _state.mapPoints.push_back(mapPoint);
                            }

                            updateKeyFrameConnections(_state.keyFrames[0]);
                            updateKeyFrameConnections(currentKeyFrame);

                            printf("New Map created with %i points\n", _state.mapPoints.size());

                            { // (Global) bundle adjustment
                                const i32 numberOfIterations = 20;

                                std::vector<bool32> mapPointNotIncludedFlags;
                                mapPointNotIncludedFlags.resize(_state.mapPoints.size());

                                g2o::SparseOptimizer                    optimizer;
                                g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

                                linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

                                g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

                                g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
                                optimizer.setAlgorithm(solver);

                                //if(pbStopFlag)
                                //optimizer.setForceStopFlag(pbStopFlag);

                                u64 maxKFid = 0;

                                // Set WAIKeyFrame vertices
                                for (i32 i = 0; i < _state.keyFrames.size(); i++)
                                {
                                    KeyFrame* keyFrame = _state.keyFrames[i];

                                    // TODO(jan): bad check

                                    g2o::SE3Quat quat = convertCvMatToG2OSE3Quat(keyFrame->cTw);

                                    g2o::VertexSE3Expmap* vertex = new g2o::VertexSE3Expmap();
                                    vertex->setEstimate(quat);
                                    vertex->setId(keyFrame->index);
                                    vertex->setFixed(keyFrame->index == 0);
                                    optimizer.addVertex(vertex);

                                    if (keyFrame->index > maxKFid)
                                    {
                                        maxKFid = keyFrame->index;
                                    }
                                }

                                const r32 thHuber2D   = sqrt(5.99);
                                i32       vertexCount = 0;

                                for (i32 i = 0; i < _state.mapPoints.size(); i++)
                                {
                                    MapPoint* mapPoint = _state.mapPoints[i];

                                    if (mapPoint->bad) continue;

                                    Eigen::Matrix<r64, 3, 1> vec3d = convertCvMatToEigenVector3D(mapPoint->position);

                                    g2o::VertexSBAPointXYZ* vertex = new g2o::VertexSBAPointXYZ();
                                    vertex->setEstimate(vec3d);
                                    const i32 id = mapPoint->index + maxKFid + 1;
                                    vertex->setId(id);
                                    vertex->setMarginalized(true);
                                    optimizer.addVertex(vertex);

                                    vertexCount++;

                                    const std::map<KeyFrame*, i32> observations = mapPoint->observations;

                                    i32 edgeCount = 0;

                                    //SET EDGES
                                    for (std::map<KeyFrame*, i32>::const_iterator it = observations.begin(), itend = observations.end();
                                         it != itend;
                                         it++)
                                    {
                                        KeyFrame* keyFrame = it->first;

                                        //if (pKF->isBad() || pKF->mnId > maxKFid) continue;

                                        edgeCount++;

                                        const cv::KeyPoint& undistortedKeyPoint = keyFrame->undistortedKeyPoints[it->second];

                                        Eigen::Matrix<r64, 2, 1> observationMatrix;
                                        observationMatrix << undistortedKeyPoint.pt.x, undistortedKeyPoint.pt.y;

                                        g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();

                                        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                                        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(keyFrame->index)));
                                        edge->setMeasurement(observationMatrix);
                                        const r32& invSigmaSquared = _state.imagePyramidStats.inverseSigmaSquared[undistortedKeyPoint.octave];
                                        edge->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquared);

                                        //if (bRobust)
                                        //{
                                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                        edge->setRobustKernel(rk);
                                        rk->setDelta(thHuber2D);
                                        //}

                                        edge->fx = _state.fx;
                                        edge->fy = _state.fy;
                                        edge->cx = _state.cx;
                                        edge->cy = _state.cy;

                                        optimizer.addEdge(edge);
                                    }

                                    if (edgeCount == 0)
                                    {
                                        optimizer.removeVertex(vertex);
                                        mapPointNotIncludedFlags[i] = true;
                                        vertexCount--;
                                    }
                                    else
                                    {
                                        mapPointNotIncludedFlags[i] = false;
                                    }
                                }

                                // Optimize!
                                if (vertexCount)
                                {
                                    optimizer.initializeOptimization();
                                    optimizer.optimize(numberOfIterations);
                                }

                                // Recover optimized data

                                for (i32 i = 0; i < _state.keyFrames.size(); i++)
                                {
                                    KeyFrame* keyFrame = _state.keyFrames[i];

                                    //if (pKF->isBad()) continue;

                                    g2o::VertexSE3Expmap* vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
                                    g2o::SE3Quat          quat   = vertex->estimate();

                                    setKeyFramePose(convertG2OSE3QuatToCvMat(quat), keyFrame->cTw, keyFrame->wTc, keyFrame->worldOrigin);

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
                                    if (mapPointNotIncludedFlags[i]) continue;

                                    MapPoint* mapPoint = _state.mapPoints[i];

                                    if (mapPoint->bad) continue;

                                    g2o::VertexSBAPointXYZ* vertex = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i + _state.nextKeyFrameId));
                                    mapPoint->position             = convertEigenVector3DToCvMat(vertex->estimate());
                                    calculateMapPointNormalAndDepth(mapPoint->position,
                                                                    mapPoint->observations,
                                                                    _state.keyFrames[0],
                                                                    _state.imagePyramidStats.scaleFactors,
                                                                    _state.imagePyramidStats.numberOfScaleLevels,
                                                                    &mapPoint->minDistance,
                                                                    &mapPoint->maxDistance,
                                                                    &mapPoint->normalVector);

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

                            r32 medianDepth    = computeSceneMedianDepthForKeyFrame(_state.keyFrames[0]);
                            r32 invMedianDepth = 1.0f / medianDepth;

                            // TODO(jan): is the check for tracked map points necessary,
                            // as we have the same check already higher up?
                            i32 trackedMapPoints;

                            { // WAIKeyFrame->TrackedMapPoints
                                std::vector<MapPoint*> mapPointMatches = _state.keyFrames[0]->mapPointMatches;
                                for (i32 i = 0; i < mapPointMatches.size(); i++)
                                {
                                    MapPoint* mapPoint = mapPointMatches[i];

                                    if (!mapPoint) continue;
                                    if (mapPoint->bad) continue;

                                    if (mapPoint->observations.size() > 0)
                                    {
                                        trackedMapPoints++;
                                    }
                                }
                            }

                            if (medianDepth > 0.0f && trackedMapPoints >= 100)
                            {
                                cv::Mat scaledPose               = currentKeyFrame->cTw;
                                scaledPose.col(3).rowRange(0, 3) = scaledPose.col(3).rowRange(0, 3) * invMedianDepth;
                                setKeyFramePose(scaledPose,
                                                currentKeyFrame->cTw,
                                                currentKeyFrame->wTc,
                                                currentKeyFrame->worldOrigin);

                                // Scale points
                                std::vector<MapPoint*> allMapPoints = _state.keyFrames[0]->mapPointMatches;
                                for (i32 i = 0; i < allMapPoints.size(); i++)
                                {
                                    if (allMapPoints[i])
                                    {
                                        MapPoint* mapPoint     = allMapPoints[i];
                                        cv::Mat   unscaledPose = mapPoint->position.clone();
                                        cv::Mat   scaledPose   = unscaledPose * invMedianDepth;
                                        scaledPose.copyTo(mapPoint->position);
                                    }
                                }

                                _state.localMapping.newKeyFrames.push_back(_state.keyFrames[0]);
                                _state.localMapping.newKeyFrames.push_back(currentKeyFrame);

                                _state.localKeyFrames.push_back(_state.keyFrames[0]);
                                _state.localKeyFrames.push_back(currentKeyFrame);

                                i32 numberOfFeatures                           = 1000;
                                _state.fastFeatureConstraints.numberOfFeatures = numberOfFeatures;
                                calculateNumberOfFeaturesPerScaleLevel(_state.scaleFactor, numberOfFeatures, &_state.imagePyramidStats);

                                // TODO(jan): save stuff for camera trajectory
                                // TODO(jan): set reference map points

                                runLocalMapping(&_state.localMapping,
                                                _state.keyFrames.size(),
                                                _state.mapPoints,
                                                _state.imagePyramidStats,
                                                _state.fx,
                                                _state.fy,
                                                _state.cx,
                                                _state.cy,
                                                _state.invfx,
                                                _state.invfy,
                                                _state.scaleFactor,
                                                cameraMat,
                                                &_state.nextMapPointId,
                                                _state.orbVocabulary);
                                runLocalMapping(&_state.localMapping,
                                                _state.keyFrames.size(),
                                                _state.mapPoints,
                                                _state.imagePyramidStats,
                                                _state.fx,
                                                _state.fy,
                                                _state.cx,
                                                _state.cy,
                                                _state.invfx,
                                                _state.invfy,
                                                _state.scaleFactor,
                                                cameraMat,
                                                &_state.nextMapPointId,
                                                _state.orbVocabulary);

                                _state.referenceKeyFrame = currentKeyFrame;
                                _state.lastKeyFrameId    = currentKeyFrame->index;

                                _state.status        = OrbSlamStatus_Tracking;
                                _state.trackingWasOk = true;

                                printf("Second initialization keyFrame at frame %i\n", _state.frameCounter);

                                std::cout << _state.keyFrames[0]->cTw << std::endl;
                                std::cout << currentKeyFrame->cTw << std::endl;
                            }
                            else
                            {
                                delete _state.keyFrames[0];
                                delete currentKeyFrame;
                                _state.keyFrames.clear();

                                _state.nextKeyFrameId = 0;
                            }
                        }
                    }
                    else
                    {
                        delete _state.keyFrames[0];
                        delete currentKeyFrame;
                        _state.keyFrames.clear();

                        _state.nextKeyFrameId = 0;
                    }
                }
                else
                {
                    delete _state.keyFrames[0];
                    delete currentKeyFrame;
                    _state.keyFrames.clear();

                    _state.nextKeyFrameId = 0;

                    std::fill(_state.initializationMatches.begin(), _state.initializationMatches.end(), -1);
                }
            }
        }
        break;

        case OrbSlamStatus_Tracking:
        {
            cv::Mat cameraMat     = _camera->getCameraMatrix();
            cv::Mat distortionMat = _camera->getDistortionMatrix();
            cv::Mat cameraFrame   = _camera->getImageGray();

            KeyFrame* currentFrame = new KeyFrame();
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
                               currentFrame,
                               _state.nextKeyFrameId);
            _state.nextKeyFrameId++;

            KeyFrame* referenceKeyFrame = _state.referenceKeyFrame;

            bool32 trackingIsOk = false;

            if (_state.trackingWasOk)
            {
                // TODO(jan): checkReplacedInLastFrame
                // TODO(jan): velocity model

                computeBoW(_state.orbVocabulary, currentFrame);

                std::vector<MapPoint*> mapPointMatches;
                i32                    matchCount = findMapPointMatchesByBoW(referenceKeyFrame,
                                                          currentFrame,
                                                          mapPointMatches);

                // TODO(jan): magic number
                if (matchCount > 15)
                {
                    currentFrame->mapPointMatches = mapPointMatches;
                    currentFrame->cTw             = referenceKeyFrame->cTw.clone();

                    optimizePose(_state.imagePyramidStats.inverseSigmaSquared,
                                 _state.fx,
                                 _state.fy,
                                 _state.cx,
                                 _state.cy,
                                 currentFrame);

                    i32 goodMatchCount = 0;

                    // discard outliers
                    for (i32 i = 0; i < currentFrame->numberOfKeyPoints; i++)
                    {
                        MapPoint* mapPoint = currentFrame->mapPointMatches[i];
                        if (mapPoint)
                        {
                            if (currentFrame->mapPointIsOutlier[i])
                            {
                                currentFrame->mapPointMatches[i]   = nullptr;
                                currentFrame->mapPointIsOutlier[i] = false;
                                //matchCount--;
                            }
                            else if (!mapPoint->observations.empty())
                            {
                                goodMatchCount++;
                            }
                        }
                    }

                    if (goodMatchCount > 10)
                    {
                        trackingIsOk = true;
                    }

                    printf("Found %i good matches (out of %i)\n", goodMatchCount, matchCount);
                }
                else
                {
                    printf("Only found %i matches\n", matchCount);
                }
            }
            else
            {
                // TODO(jan): relocalization if tracking was not OK
                printf("Tracking lost, exiting\n");
                exit(0);
            }

            // TODO(jan): set current frame reference keyframe

            i32 inlierMatches = 0;

            if (trackingIsOk)
            {
                {     // Track local map
                    { // update local keyframes
                        // Each map point votes for the keyframes in which it has been observed
                        std::map<KeyFrame*, i32> keyframeCounter;
                        for (i32 i = 0; i < currentFrame->numberOfKeyPoints; i++)
                        {
                            MapPoint* mapPoint = currentFrame->mapPointMatches[i];

                            if (mapPoint)
                            {
                                if (!mapPoint->bad)
                                {
                                    std::map<KeyFrame*, i32> observations = mapPoint->observations;

                                    for (std::map<KeyFrame*, i32>::const_iterator it = observations.begin(), itend = observations.end();
                                         it != itend;
                                         it++)
                                    {
                                        keyframeCounter[it->first]++;
                                    }
                                }
                                else
                                {
                                    currentFrame->mapPointMatches[i] = nullptr;
                                }
                            }
                        }

                        if (!keyframeCounter.empty())
                        {
                            i32       maxObservations             = 0;
                            KeyFrame* keyFrameWithMaxObservations = nullptr;

                            _state.localKeyFrames.clear();
                            _state.localKeyFrames.reserve(3 * keyframeCounter.size());

                            // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
                            for (std::map<KeyFrame*, i32>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
                                 it != itEnd;
                                 it++)
                            {
                                KeyFrame* keyFrame = it->first;

                                //if (pKF->isBad()) continue;

                                if (it->second > maxObservations)
                                {
                                    maxObservations             = it->second;
                                    keyFrameWithMaxObservations = keyFrame;
                                }

                                _state.localKeyFrames.push_back(it->first);
                                keyFrame->trackReferenceForFrame = currentFrame->index;
                            }

                            // Include also some not-already-included keyframes that are neighbors to already-included keyframes
                            for (std::vector<KeyFrame*>::const_iterator itKF = _state.localKeyFrames.begin(), itEndKF = _state.localKeyFrames.end();
                                 itKF != itEndKF;
                                 itKF++)
                            {
                                // Limit the number of keyframes
                                if (_state.localKeyFrames.size() > 80) break;

                                KeyFrame* keyFrame = *itKF;

                                std::vector<KeyFrame*> neighborKeyFrames = getBestCovisibilityKeyFrames(10,
                                                                                                        currentFrame->orderedConnectedKeyFrames);

                                for (std::vector<KeyFrame*>::const_iterator itNeighKF = neighborKeyFrames.begin(), itEndNeighKF = neighborKeyFrames.end();
                                     itNeighKF != itEndNeighKF;
                                     itNeighKF++)
                                {
                                    KeyFrame* neighborKeyFrame = *itNeighKF;
                                    //if (!pNeighKF->isBad())
                                    {
                                        if (neighborKeyFrame->trackReferenceForFrame != currentFrame->index)
                                        {
                                            _state.localKeyFrames.push_back(neighborKeyFrame);
                                            neighborKeyFrame->trackReferenceForFrame = currentFrame->index;
                                            break;
                                        }
                                    }
                                }

                                const std::vector<KeyFrame*> childrenKeyFrames = keyFrame->children;
                                for (std::vector<KeyFrame*>::const_iterator sit = childrenKeyFrames.begin(), send = childrenKeyFrames.end();
                                     sit != send;
                                     sit++)
                                {
                                    KeyFrame* childKeyFrame = *sit;
                                    //if (!pChildKF->isBad())
                                    {
                                        if (childKeyFrame->trackReferenceForFrame != currentFrame->index)
                                        {
                                            _state.localKeyFrames.push_back(childKeyFrame);
                                            childKeyFrame->trackReferenceForFrame = currentFrame->index;
                                            break;
                                        }
                                    }
                                }

                                KeyFrame* parentKeyFrame = keyFrame->parent;
                                if (parentKeyFrame)
                                {
                                    if (parentKeyFrame->trackReferenceForFrame != currentFrame->index)
                                    {
                                        _state.localKeyFrames.push_back(parentKeyFrame);
                                        parentKeyFrame->trackReferenceForFrame = currentFrame->index;
                                        break;
                                    }
                                }
                            }

                            if (keyFrameWithMaxObservations)
                            {
                                _state.referenceKeyFrame        = keyFrameWithMaxObservations;
                                currentFrame->referenceKeyFrame = keyFrameWithMaxObservations;
                            }
                        }
                    }

                    { // update local points
                        // TODO(jan): as we always clear the localMapPoints, is it necessary to keep it in state?
                        _state.localMapPoints.clear();

                        for (std::vector<KeyFrame*>::const_iterator itKF = _state.localKeyFrames.begin(), itEndKF = _state.localKeyFrames.end();
                             itKF != itEndKF;
                             itKF++)
                        {
                            KeyFrame*                    keyFrame        = *itKF;
                            const std::vector<MapPoint*> mapPointMatches = keyFrame->mapPointMatches;

                            for (std::vector<MapPoint*>::const_iterator itMP = mapPointMatches.begin(), itEndMP = mapPointMatches.end();
                                 itMP != itEndMP;
                                 itMP++)
                            {
                                MapPoint* mapPoint = *itMP;

                                if (!mapPoint) continue;
                                if (mapPoint->trackReferenceForFrame == currentFrame->index) continue;

                                if (!mapPoint->bad)
                                {
                                    _state.localMapPoints.push_back(mapPoint);
                                    mapPoint->trackReferenceForFrame = currentFrame->index;
                                }
                            }
                        }
                    }

                    { // search local points
                        std::vector<bool32> mapPointAlreadyMatched = std::vector<bool32>(_state.mapPoints.size(), false);

                        // Do not search map points already matched
                        for (std::vector<MapPoint*>::iterator vit = currentFrame->mapPointMatches.begin(), vend = currentFrame->mapPointMatches.end();
                             vit != vend;
                             vit++)
                        {
                            MapPoint* mapPoint = *vit;

                            if (mapPoint)
                            {
                                if (mapPoint->bad)
                                {
                                    *vit = nullptr;
                                }
                                else
                                {
                                    mapPoint->visibleInKeyFrameCounter++;
                                    mapPoint->lastFrameSeen = currentFrame->index;

                                    /*
                                    pMP->IncreaseVisible();
                                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                                    pMP->mbTrackInView   = false;
                                    */
                                }
                            }
                        }

                        i32 numberOfMapPointsToMatch = 0;

                        // Project points in frame and check its visibility
                        for (std::vector<MapPoint*>::iterator vit = _state.localMapPoints.begin(), vend = _state.localMapPoints.end();
                             vit != vend;
                             vit++)
                        {
                            MapPoint* mapPoint = *vit;

                            if (mapPoint)
                            {
                                if (mapPoint->lastFrameSeen == currentFrame->index) continue;
                                if (mapPoint->bad) continue;

                                // Project (this fills WAIMapPoint variables for matching)
                                if (isMapPointInFrameFrustum(currentFrame,
                                                             _state.fx,
                                                             _state.fy,
                                                             _state.cx,
                                                             _state.cy,
                                                             _state.gridConstraints.minX,
                                                             _state.gridConstraints.maxX,
                                                             _state.gridConstraints.minY,
                                                             _state.gridConstraints.maxY,
                                                             mapPoint,
                                                             0.5f,
                                                             &mapPoint->trackingInfos))
                                {
                                    mapPoint->visibleInKeyFrameCounter++;
                                    numberOfMapPointsToMatch++;
                                }
                            }
                        }

                        if (numberOfMapPointsToMatch > 0)
                        {
                            i32 threshold = 1;

                            // TODO(jan): reactivate this
                            // If the camera has been relocalised recently, perform a coarser search
                            /*if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                        {
                            threshold = 5;
                        }*/

                            i32 localMatches = searchMapPointsByProjection(_state.localMapPoints,
                                                                           _state.imagePyramidStats.scaleFactors,
                                                                           _state.gridConstraints,
                                                                           MATCHER_DISTANCE_THRESHOLD_HIGH,
                                                                           0.8f,
                                                                           currentFrame,
                                                                           threshold);
                            printf("Found %i mapPoints by projection of local map\n", localMatches);
                        }
                    }

                    // TODO(jan): we call this function twice... is this necessary?
                    optimizePose(_state.imagePyramidStats.inverseSigmaSquared,
                                 _state.fx,
                                 _state.fy,
                                 _state.cx,
                                 _state.cy,
                                 currentFrame);

                    for (i32 i = 0; i < currentFrame->numberOfKeyPoints; i++)
                    {
                        MapPoint* mapPoint = currentFrame->mapPointMatches[i];
                        if (mapPoint)
                        {
                            if (!currentFrame->mapPointIsOutlier[i])
                            {
                                mapPoint->foundInKeyFrameCounter++;

                                if (mapPoint->observations.size() > 0)
                                {
                                    inlierMatches++;
                                }
                            }
                        }
                    }

                    // TODO(jan): reactivate this
                    // Decide if the tracking was succesful
                    // More restrictive if there was a relocalization recently
                    /*if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
                    {
                        //cout << "mnMatchesInliers: " << mnMatchesInliers << endl;
                        return false;
                    }*/

                    if (inlierMatches < 30)
                    {
                        trackingIsOk = false;
                    }
                }
            }

            _state.trackingWasOk = trackingIsOk;

            if (trackingIsOk)
            {
                // TODO(jan): update motion model

                _pose = currentFrame->cTw.clone();

                // Clean VO matches
                for (int i = 0; i < currentFrame->numberOfKeyPoints; i++)
                {
                    MapPoint* mapPoint = currentFrame->mapPointMatches[i];
                    if (mapPoint)
                    {
                        if (mapPoint->observations.size() < 1)
                        {
                            currentFrame->mapPointMatches[i]   = nullptr;
                            currentFrame->mapPointIsOutlier[i] = false;
                        }
                    }
                }

                // TODO(jan): delete temporal map points (needed in motion model)

                // TODO(jan): magic numbers
                bool32 addNewKeyFrame = needNewKeyFrame(currentFrame->index,
                                                        _state.lastKeyFrameId,
                                                        _state.lastRelocalizationKeyFrameId,
                                                        0,
                                                        30,
                                                        inlierMatches,
                                                        _state.referenceKeyFrame,
                                                        _state.keyFrames.size());

                if (addNewKeyFrame)
                {
                    _state.keyFrames.push_back(currentFrame);
                    _state.referenceKeyFrame = currentFrame;

                    _state.localMapping.newKeyFrames.push_back(currentFrame);

                    runLocalMapping(&_state.localMapping,
                                    _state.keyFrames.size(),
                                    _state.mapPoints,
                                    _state.imagePyramidStats,
                                    _state.fx,
                                    _state.fy,
                                    _state.cx,
                                    _state.cy,
                                    _state.invfx,
                                    _state.invfy,
                                    _state.scaleFactor,
                                    cameraMat,
                                    &_state.nextMapPointId,
                                    _state.orbVocabulary);

                    _state.lastKeyFrameId = currentFrame->index;

                    // TODO(jan): camera trajectory stuff
                }
            }
        }
        break;
    }
}

std::vector<MapPoint*> WAI::ModeOrbSlam2DataOriented::getMapPoints()
{
    std::vector<MapPoint*> result = _state.mapPoints;

    return result;
}

bool WAI::ModeOrbSlam2DataOriented::getPose(cv::Mat* pose)
{
    *pose = _pose;

    return true;
}
