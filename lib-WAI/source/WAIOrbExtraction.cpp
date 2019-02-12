#include <WAIOrbExtraction.h>

const r32   factorPI = (r32)(CV_PI / 180.f);
static void computeOrbDescriptor(const cv::KeyPoint& kpt,
                                 const cv::Mat&      img,
                                 const cv::Point*    pattern,
                                 uchar*              desc)
{
    r32 angle = (r32)kpt.angle * factorPI;
    r32 a = (r32)cos(angle), b = (r32)sin(angle);

    const u8* center = &img.at<u8>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const i32 step   = (i32)img.step;

#define GET_VALUE(idx) \
    center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + \
           cvRound(pattern[idx].x * a - pattern[idx].y * b)]

    for (i32 i = 0; i < 32; ++i, pattern += 16)
    {
        i32 t0, t1, val;
        t0  = GET_VALUE(0);
        t1  = GET_VALUE(1);
        val = t0 < t1;
        t0  = GET_VALUE(2);
        t1  = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

#undef GET_VALUE
}

static r32 computeKeyPointAngle(const cv::Mat&          image,
                                cv::Point2f             pt,
                                const std::vector<i32>& umax,
                                const i32               halfPatchSize)
{
    i32 m_01 = 0, m_10 = 0;

    const u8* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (i32 u = -halfPatchSize; u <= halfPatchSize; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    i32 step = (i32)image.step1();
    for (i32 v = 1; v <= halfPatchSize; ++v)
    {
        // Proceed over the two lines
        i32 v_sum = 0;
        i32 d     = umax[v];
        for (i32 u = -d; u <= d; ++u)
        {
            i32 val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    r32 result = cv::fastAtan2((r32)m_01, (r32)m_10);

    return result;
}

void divideOrbExtractorNode(OrbExtractorNode& parentNode,
                            OrbExtractorNode& n1,
                            OrbExtractorNode& n2,
                            OrbExtractorNode& n3,
                            OrbExtractorNode& n4)
{
    const i32 halfX = ceil(static_cast<r32>(parentNode.topRight.x - parentNode.topLeft.x) / 2);
    const i32 halfY = ceil(static_cast<r32>(parentNode.bottomRight.y - parentNode.topLeft.y) / 2);

    //Define boundaries of childs
    n1.topLeft     = parentNode.topLeft;
    n1.topRight    = cv::Point2i(parentNode.topLeft.x + halfX, parentNode.topLeft.y);
    n1.bottomLeft  = cv::Point2i(parentNode.topLeft.x, parentNode.topLeft.y + halfY);
    n1.bottomRight = cv::Point2i(parentNode.topLeft.x + halfX, parentNode.topLeft.y + halfY);
    n1.keys.reserve(parentNode.keys.size());

    n2.topLeft     = n1.topRight;
    n2.topRight    = parentNode.topRight;
    n2.bottomLeft  = n1.bottomRight;
    n2.bottomRight = cv::Point2i(parentNode.topRight.x, parentNode.topLeft.y + halfY);
    n2.keys.reserve(parentNode.keys.size());

    n3.topLeft     = n1.bottomLeft;
    n3.topRight    = n1.bottomRight;
    n3.bottomLeft  = parentNode.bottomLeft;
    n3.bottomRight = cv::Point2i(n1.bottomRight.x, parentNode.bottomLeft.y);
    n3.keys.reserve(parentNode.keys.size());

    n4.topLeft     = n3.topRight;
    n4.topRight    = n2.bottomRight;
    n4.bottomLeft  = n3.bottomRight;
    n4.bottomRight = parentNode.bottomRight;
    n4.keys.reserve(parentNode.keys.size());

    //Associate points to childs
    for (size_t i = 0; i < parentNode.keys.size(); i++)
    {
        const cv::KeyPoint& kp = parentNode.keys[i];
        if (kp.pt.x < n1.topRight.x)
        {
            if (kp.pt.y < n1.bottomRight.y)
                n1.keys.push_back(kp);
            else
                n3.keys.push_back(kp);
        }
        else if (kp.pt.y < n1.bottomRight.y)
            n2.keys.push_back(kp);
        else
            n4.keys.push_back(kp);
    }

    if (n1.keys.size() == 1)
        n1.noMoreSubdivision = true;
    if (n2.keys.size() == 1)
        n2.noMoreSubdivision = true;
    if (n3.keys.size() == 1)
        n3.noMoreSubdivision = true;
    if (n4.keys.size() == 1)
        n4.noMoreSubdivision = true;
}

std::vector<cv::KeyPoint> distributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                            const i32                        minX,
                                            const i32                        maxX,
                                            const i32                        minY,
                                            const i32                        maxY,
                                            const i32                        N,
                                            const i32                        level,
                                            const i32                        numberOfFeatures)
{
    // Compute how many initial nodes
    const i32 nIni = std::round(static_cast<r32>(maxX - minX) / (maxY - minY));

    const r32 hX = static_cast<r32>(maxX - minX) / nIni;

    std::list<OrbExtractorNode> lNodes;

    std::vector<OrbExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for (i32 i = 0; i < nIni; i++)
    {
        OrbExtractorNode ni = {};
        ni.topLeft          = cv::Point2i(hX * static_cast<r32>(i), 0);
        ni.topRight         = cv::Point2i(hX * static_cast<r32>(i + 1), 0);
        ni.bottomLeft       = cv::Point2i(ni.topLeft.x, maxY - minY);
        ni.bottomRight      = cv::Point2i(ni.topRight.x, maxY - minY);
        ni.keys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // Associate points to childs
    for (size_t i = 0; i < vToDistributeKeys.size(); i++)
    {
        const cv::KeyPoint& kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->keys.push_back(kp);
    }

    std::list<OrbExtractorNode>::iterator lit = lNodes.begin();

    while (lit != lNodes.end())
    {
        if (lit->keys.size() == 1)
        {
            lit->noMoreSubdivision = true;
            lit++;
        }
        else if (lit->keys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    i32 iteration = 0;

    std::vector<std::pair<i32, OrbExtractorNode*>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish)
    {
        iteration++;

        i32 prevSize = lNodes.size();

        lit = lNodes.begin();

        i32 nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while (lit != lNodes.end())
        {
            if (lit->noMoreSubdivision)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                OrbExtractorNode n1, n2, n3, n4;
                divideOrbExtractorNode(*lit, n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.keys.size() > 0)
                {
                    lNodes.push_front(n1);
                    if (n1.keys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n1.keys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n2.keys.size() > 0)
                {
                    lNodes.push_front(n2);
                    if (n2.keys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n2.keys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.keys.size() > 0)
                {
                    lNodes.push_front(n3);
                    if (n3.keys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n3.keys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.keys.size() > 0)
                {
                    lNodes.push_front(n4);
                    if (n4.keys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n4.keys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit = lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((i32)lNodes.size() >= N || (i32)lNodes.size() == prevSize)
        {
            bFinish = true;
        }
        else if (((i32)lNodes.size() + nToExpand * 3) > N)
        {

            while (!bFinish)
            {

                prevSize = (i32)lNodes.size();

                std::vector<std::pair<i32, OrbExtractorNode*>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
                for (i32 j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    OrbExtractorNode n1, n2, n3, n4;
                    divideOrbExtractorNode(*vPrevSizeAndPointerToNode[j].second, n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.keys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.keys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n1.keys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.keys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.keys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n2.keys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.keys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.keys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n3.keys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.keys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.keys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n4.keys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((i32)lNodes.size() >= N)
                        break;
                }

                if ((i32)lNodes.size() >= N || (i32)lNodes.size() == prevSize)
                    bFinish = true;
            }
        }
    }

    // Retain the best point in each node
    std::vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(numberOfFeatures);
    for (std::list<OrbExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
    {
        std::vector<cv::KeyPoint>& vNodeKeys   = lit->keys;
        cv::KeyPoint*              pKP         = &vNodeKeys[0];
        r32                        maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); k++)
        {
            if (vNodeKeys[k].response > maxResponse)
            {
                pKP         = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

/**
     * 1. Splits every level of the image into evenly sized cells
     * 2. Detects corners in a 7x7 cell area
     * 3. Make sure key points are well distributed
     * 4. Compute orientation of keypoints
     * @param allKeypoints
     */
void computeKeyPointsInOctTree(const i32                               numberOfLevels,
                               const std::vector<cv::Mat>&             imagePyramid,
                               const i32                               edgeThreshold,
                               const i32                               numberOfFeatures,
                               const std::vector<i32>                  numberOfFeaturesPerScaleLevel,
                               const r32                               initialFastThreshold,
                               const r32                               minimalFastThreshold,
                               const i32                               patchSize,
                               const i32                               halfPatchSize,
                               const std::vector<r32>                  pyramidScaleFactors,
                               const std::vector<i32>                  umax,
                               std::vector<std::vector<cv::KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(numberOfLevels);

    const r32 W = 30; // TODO(jan): number of cells?

    for (i32 level = 0; level < numberOfLevels; ++level)
    {
        const i32 minBorderX = edgeThreshold - 3;
        const i32 minBorderY = minBorderX;
        const i32 maxBorderX = imagePyramid[level].cols - edgeThreshold + 3;
        const i32 maxBorderY = imagePyramid[level].rows - edgeThreshold + 3;

        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(numberOfFeatures * 10);

        const r32 width  = (maxBorderX - minBorderX);
        const r32 height = (maxBorderY - minBorderY);

        const i32 nCols = width / W;
        const i32 nRows = height / W;
        const i32 wCell = ceil(width / nCols);
        const i32 hCell = ceil(height / nRows);

        for (i32 i = 0; i < nRows; i++)
        {
            const r32 iniY = minBorderY + i * hCell;
            r32       maxY = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3)
                continue;
            if (maxY > maxBorderY)
                maxY = (r32)maxBorderY;

            for (i32 j = 0; j < nCols; j++)
            {
                const r32 iniX = minBorderX + j * wCell;
                r32       maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6)
                    continue;
                if (maxX > maxBorderX)
                    maxX = (r32)maxBorderX;

                std::vector<cv::KeyPoint> vKeysCell;
                cv::FAST(imagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell,
                         initialFastThreshold,
                         true);

                if (vKeysCell.empty())
                {
                    FAST(imagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell,
                         minimalFastThreshold,
                         true);
                }

                if (!vKeysCell.empty())
                {
                    for (std::vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                    {
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }

        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(numberOfFeatures);

        keypoints = distributeOctTree(vToDistributeKeys,
                                      minBorderX,
                                      maxBorderX,
                                      minBorderY,
                                      maxBorderY,
                                      numberOfFeaturesPerScaleLevel[level],
                                      level,
                                      numberOfFeatures);

        const i32 scaledPatchSize = patchSize * pyramidScaleFactors[level];

        // Add border to coordinates and scale information
        const i32 nkps = keypoints.size();
        for (i32 i = 0; i < nkps; i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size   = (r32)scaledPatchSize;
        }
    }

    // compute orientations
    for (i32 level = 0; level < numberOfLevels; level++)
    {
        std::vector<cv::KeyPoint> keyPoints = allKeypoints[level];
        for (std::vector<cv::KeyPoint>::iterator keyPoint    = keyPoints.begin(),
                                                 keyPointEnd = keyPoints.end();
             keyPoint != keyPointEnd;
             keyPoint++)
        {
            keyPoint->angle = computeKeyPointAngle(imagePyramid[level],
                                                   keyPoint->pt,
                                                   umax,
                                                   halfPatchSize);
        }
    }
}
