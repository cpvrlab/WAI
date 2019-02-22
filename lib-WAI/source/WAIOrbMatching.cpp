#include <WAIPlatform.h>

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
