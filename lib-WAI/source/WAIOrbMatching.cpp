#include <WAIOrbMatching.h>

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

bool32 calculateKeyPointGridCell(const cv::KeyPoint&   keyPoint,
                                 const GridConstraints gridConstraints,
                                 i32*                  posX,
                                 i32*                  posY)
{
    bool32 result = false;

    i32 x = (i32)round((keyPoint.pt.x - gridConstraints.minX) * gridConstraints.invGridElementWidth);
    i32 y = (i32)round((keyPoint.pt.y - gridConstraints.minY) * gridConstraints.invGridElementHeight);

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
        mat.at<r32>(i, 0) = keyPoints[i].pt.x;
        mat.at<r32>(i, 1) = keyPoints[i].pt.y;
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
        kp.pt.x                 = mat.at<r32>(i, 0);
        kp.pt.y                 = mat.at<r32>(i, 1);
        undistortedKeyPoints[i] = kp;
    }
}

void computeScalePyramid(const cv::Mat            image,
                         const ImagePyramidStats& imagePyramidStats,
                         const i32                edgeThreshold,
                         std::vector<cv::Mat>&    imagePyramid)
{
    for (i32 level = 0; level < imagePyramidStats.numberOfScaleLevels; level++)
    {
        r32      scale = imagePyramidStats.inverseScaleFactors[level];
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

static std::vector<size_t> getFeatureIndicesForArea(const i32                       numberOfKeyPoints,
                                                    const r32                       searchWindowSize,
                                                    const r32                       x,
                                                    const r32                       y,
                                                    const GridConstraints           gridConstraints,
                                                    const i32                       minLevel,
                                                    const i32                       maxLevel,
                                                    const std::vector<size_t>       keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                                                    const std::vector<cv::KeyPoint> undistortedKeyPoints)
{
    std::vector<size_t> result;

    result.reserve(numberOfKeyPoints);

    const i32 nMinCellX = std::max(0, (i32)floor((x - gridConstraints.minX - searchWindowSize) * gridConstraints.invGridElementWidth));
    if (nMinCellX >= FRAME_GRID_COLS)
        return result;

    const i32 nMaxCellX = std::min((i32)FRAME_GRID_COLS - 1, (i32)ceil((x - gridConstraints.minX + searchWindowSize) * gridConstraints.invGridElementWidth));
    if (nMaxCellX < 0)
        return result;

    const i32 nMinCellY = std::max(0, (i32)floor((y - gridConstraints.minY - searchWindowSize) * gridConstraints.invGridElementHeight));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return result;

    const i32 nMaxCellY = std::min((i32)FRAME_GRID_ROWS - 1, (i32)ceil((y - gridConstraints.minY + searchWindowSize) * gridConstraints.invGridElementHeight));
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

static void findInitializationMatches(const KeyFrame*                 keyFrame1,
                                      const KeyFrame*                 keyFrame2,
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

    std::vector<i32> matchesDistances(keyFrame2->numberOfKeyPoints, INT_MAX);
    std::vector<i32> matchesKeyPointIndices(keyFrame2->numberOfKeyPoints, -1);

    for (size_t i1 = 0, iend1 = keyFrame1->undistortedKeyPoints.size();
         i1 < iend1;
         i1++)
    {
        cv::KeyPoint keyPoint1 = keyFrame1->undistortedKeyPoints[i1];

        i32 level1 = keyPoint1.octave;
        if (level1 > 0) continue;

        // TODO(jan): magic number
        std::vector<size_t> keyPointIndicesCurrentFrame =
          getFeatureIndicesForArea(keyFrame2->numberOfKeyPoints,
                                   100,
                                   previouslyMatchedKeyPoints[i1].x,
                                   previouslyMatchedKeyPoints[i1].y,
                                   gridConstraints,
                                   level1,
                                   level1,
                                   keyFrame2->keyPointIndexGrid,
                                   keyFrame2->undistortedKeyPoints);

        if (keyPointIndicesCurrentFrame.empty()) continue;

        cv::Mat d1 = keyFrame1->descriptors.row(i1);

        // smaller is better
        i32 shortestDist       = INT_MAX;
        i32 secondShortestDist = INT_MAX;
        i32 shortestDistId     = -1;

        for (std::vector<size_t>::iterator vit = keyPointIndicesCurrentFrame.begin();
             vit != keyPointIndicesCurrentFrame.end();
             vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = keyFrame2->descriptors.row(i2);

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
                    r32 rot = keyFrame1->undistortedKeyPoints[i1].angle - keyFrame2->undistortedKeyPoints[shortestDistId].angle;
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

static i32 findMapPointMatchesByBoW(const KeyFrame*         referenceKeyFrame,
                                    const KeyFrame*         currentFrame,
                                    std::vector<MapPoint*>& mapPoints)
{
    i32 result = 0;

    const r32 bestToSecondBestRatio = 0.7f;

    const std::vector<MapPoint*> referenceKeyFrameMapPoints = referenceKeyFrame->mapPointMatches;
    mapPoints                                               = std::vector<MapPoint*>(currentFrame->numberOfKeyPoints, nullptr);

    const DBoW2::FeatureVector& featureVectorRefKeyFrame  = referenceKeyFrame->featureVector;
    const DBoW2::FeatureVector& featureVectorCurrentFrame = currentFrame->featureVector;

    std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
    {
        rotHist[i].reserve(500);
    }
    const r32 factor = 1.0f / ROTATION_HISTORY_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit  = featureVectorRefKeyFrame.begin();
    DBoW2::FeatureVector::const_iterator Fit   = featureVectorCurrentFrame.begin();
    DBoW2::FeatureVector::const_iterator KFend = featureVectorRefKeyFrame.end();
    DBoW2::FeatureVector::const_iterator Fend  = featureVectorCurrentFrame.end();

    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            const vector<u32> vIndicesKF = KFit->second;
            const vector<u32> vIndicesF  = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const u32 realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = referenceKeyFrameMapPoints[realIdxKF];

                if (!pMP) continue;
                if (pMP->bad) continue;

                const cv::Mat& dKF = referenceKeyFrame->descriptors.row(realIdxKF);

                i32 bestDist1 = 256;
                i32 bestIdxF  = -1;
                i32 bestDist2 = 256;

                for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                {
                    const u32 realIdxF = vIndicesF[iF];

                    if (mapPoints[realIdxF]) continue;

                    const cv::Mat& dF = currentFrame->descriptors.row(realIdxF);

                    const i32 dist = descriptorDistance(dKF, dF);

                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdxF  = realIdxF;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 <= MATCHER_DISTANCE_THRESHOLD_LOW)
                {
                    if (static_cast<r32>(bestDist1) < bestToSecondBestRatio * static_cast<r32>(bestDist2))
                    {
                        mapPoints[bestIdxF] = pMP;

                        const cv::KeyPoint& kp = referenceKeyFrame->undistortedKeyPoints[realIdxKF];

                        //if (mbCheckOrientation)
                        //{
                        // TODO(jan): are we sure that we should not use undistorted keypoints here?
                        r32 rot = kp.angle - currentFrame->keyPoints[bestIdxF].angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        i32 bin = round(rot * factor);
                        if (bin == ROTATION_HISTORY_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < ROTATION_HISTORY_LENGTH);
                        rotHist[bin].push_back(bestIdxF);
                        //}
                        result++;
                    }
                }
            }

            KFit++;
            Fit++;
        }
        else if (KFit->first < Fit->first)
        {
            KFit = featureVectorRefKeyFrame.lower_bound(Fit->first);
        }
        else
        {
            Fit = featureVectorCurrentFrame.lower_bound(KFit->first);
        }
    }

    //if (mbCheckOrientation)
    //{
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    computeThreeMaxima(rotHist, ROTATION_HISTORY_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < ROTATION_HISTORY_LENGTH; i++)
    {
        if (i == ind1 || i == ind2 || i == ind3) continue;

        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
        {
            mapPoints[rotHist[i][j]] = nullptr;
            result--;
        }
    }
    //}

    return result;
}

static bool checkDistEpipolarLine(const cv::KeyPoint&     kp1,
                                  const cv::KeyPoint&     kp2,
                                  const cv::Mat&          F12,
                                  const std::vector<r32>& sigmaSquared)
{ // Epipolar line in second image l = x1'F12 = [a b c]
    const r32 a = kp1.pt.x * F12.at<r32>(0, 0) + kp1.pt.y * F12.at<r32>(1, 0) + F12.at<r32>(2, 0);
    const r32 b = kp1.pt.x * F12.at<r32>(0, 1) + kp1.pt.y * F12.at<r32>(1, 1) + F12.at<r32>(2, 1);
    const r32 c = kp1.pt.x * F12.at<r32>(0, 2) + kp1.pt.y * F12.at<r32>(1, 2) + F12.at<r32>(2, 2);

    const r32 num = a * kp2.pt.x + b * kp2.pt.y + c;

    const r32 den = a * a + b * b;

    if (den == 0) // TODO(jan): this is very bad practice, floating point inprecision
    {
        return false;
    }

    const r32 dsqr = num * num / den;

    return dsqr < 3.84 * sigmaSquared[kp2.octave];
}

static i32 searchMapPointMatchesForTriangulation(KeyFrame*                               keyFrame1,
                                                 KeyFrame*                               keyFrame2,
                                                 r32                                     fx,
                                                 r32                                     fy,
                                                 r32                                     cx,
                                                 r32                                     cy,
                                                 const std::vector<r32>&                 sigmaSquared,
                                                 const std::vector<r32>&                 scaleFactors,
                                                 cv::Mat&                                F12,
                                                 bool32                                  checkOrientation,
                                                 std::vector<std::pair<size_t, size_t>>& vMatchedPairs)
{
    const DBoW2::FeatureVector& vFeatVec1 = keyFrame1->featureVector;
    const DBoW2::FeatureVector& vFeatVec2 = keyFrame2->featureVector;

    //Compute epipole in second image
    cv::Mat Cw = getKeyFrameCameraCenter(keyFrame1);
    std::cout << "Cw: " << Cw << std::endl;

    cv::Mat R2w = getKeyFrameRotation(keyFrame2);
    std::cout << "R2w: " << R2w << std::endl;

    cv::Mat t2w = getKeyFrameTranslation(keyFrame2);
    std::cout << "t2w: " << t2w << std::endl;

    cv::Mat C2 = R2w * Cw + t2w;
    std::cout << "C2: " << C2 << std::endl;

    const r32 invz = 1.0f / C2.at<r32>(2);
    const r32 ex   = fx * C2.at<r32>(0) * invz + cx;
    const r32 ey   = fy * C2.at<r32>(1) * invz + cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    i32                 nmatches = 0;
    std::vector<bool32> vbMatched2(keyFrame2->numberOfKeyPoints, false);
    std::vector<i32>    vMatches12(keyFrame1->numberOfKeyPoints, -1);

    std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
    {
        rotHist[i].reserve(500);
    }

    const r32 factor = 1.0f / ROTATION_HISTORY_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it  = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it  = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = keyFrame1->mapPointMatches[idx1];

                // If there is already a MapPoint skip
                if (pMP1) continue;

                const cv::KeyPoint& kp1 = keyFrame1->undistortedKeyPoints[idx1];
                const cv::Mat&      d1  = keyFrame1->descriptors.row(idx1);

                i32 bestDist = MATCHER_DISTANCE_THRESHOLD_LOW;
                i32 bestIdx2 = -1;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = keyFrame2->mapPointMatches[idx2];

                    // If we have already matched or there is a MapPoint skip
                    if (vbMatched2[idx2] || pMP2) continue;

                    const cv::Mat& d2 = keyFrame2->descriptors.row(idx2);

                    const i32 dist = descriptorDistance(d1, d2);

                    if (dist > MATCHER_DISTANCE_THRESHOLD_LOW || dist > bestDist)
                    {
                        continue;
                    }

                    const cv::KeyPoint& kp2 = keyFrame2->undistortedKeyPoints[idx2];

                    //if (!bStereo1 && !bStereo2)
                    //{
                    const r32 distex = ex - kp2.pt.x;
                    const r32 distey = ey - kp2.pt.y;
                    if (distex * distex + distey * distey < 100 * scaleFactors[kp2.octave])
                    {
                        continue;
                    }
                    //}

                    if (checkDistEpipolarLine(kp1, kp2, F12, sigmaSquared))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if (bestIdx2 >= 0)
                {
                    const cv::KeyPoint& kp2 = keyFrame2->undistortedKeyPoints[bestIdx2];
                    vMatches12[idx1]        = bestIdx2;
                    nmatches++;

                    if (checkOrientation)
                    {
                        r32 rot = kp1.angle - kp2.angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == ROTATION_HISTORY_LENGTH)
                        {
                            bin = 0;
                        }

                        assert(bin >= 0 && bin < ROTATION_HISTORY_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (checkOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        computeThreeMaxima(rotHist, ROTATION_HISTORY_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < ROTATION_HISTORY_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                vMatches12[rotHist[i][j]] = -1;
                nmatches--;
            }
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (vMatches12[i] < 0) continue;

        vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
    }

    return nmatches;
}
