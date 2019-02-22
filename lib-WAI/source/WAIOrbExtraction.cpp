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

std::vector<cv::KeyPoint> distributeOctTree(const std::vector<cv::KeyPoint>& keyPointsToDistribute,
                                            const i32                        minX,
                                            const i32                        maxX,
                                            const i32                        minY,
                                            const i32                        maxY,
                                            const i32                        requiredFeatureCount,
                                            const i32                        level)
{
    // Compute how many initial nodes
    const r32 regionWidth  = (r32)(maxX - minX);
    const r32 regionHeight = (r32)(maxY - minY);

    const i32 initialNodeCount = std::round(regionWidth / regionHeight);

    const r32 initialNodeWidth = regionWidth / initialNodeCount;

    std::list<OrbExtractorNode> nodes;

    std::vector<OrbExtractorNode*> initialNodes;
    initialNodes.resize(initialNodeCount);

    for (i32 i = 0; i < initialNodeCount; i++)
    {
        OrbExtractorNode node = {};

        r32 leftX  = initialNodeWidth * static_cast<r32>(i);
        r32 rightX = initialNodeWidth * static_cast<r32>(i + 1);

        node.topLeft     = cv::Point2i(leftX, 0);
        node.topRight    = cv::Point2i(rightX, 0);
        node.bottomLeft  = cv::Point2i(leftX, regionHeight);
        node.bottomRight = cv::Point2i(rightX, regionHeight);
        node.keys.reserve(keyPointsToDistribute.size());

        nodes.push_back(node);
        initialNodes[i] = &nodes.back();
    }

    // Assign keypoints to initial nodes
    for (size_t i = 0; i < keyPointsToDistribute.size(); i++)
    {
        const cv::KeyPoint& kp = keyPointsToDistribute[i];
        initialNodes[kp.pt.x / initialNodeWidth]->keys.push_back(kp);
    }

    std::list<OrbExtractorNode>::iterator nodeIterator = nodes.begin();

    // flag, delete or leave initial nodes, according to their keypoint count
    while (nodeIterator != nodes.end())
    {
        if (nodeIterator->keys.size() == 1)
        {
            nodeIterator->noMoreSubdivision = true;
            nodeIterator++;
        }
        else if (nodeIterator->keys.empty())
        {
            nodeIterator = nodes.erase(nodeIterator);
        }
        else
        {
            nodeIterator++;
        }
    }

    bool32 finish = false;

    i32 iteration = 0;

    std::vector<std::pair<i32, OrbExtractorNode*>> nodesToExpand;
    nodesToExpand.reserve(nodes.size() * 4);

    while (!finish)
    {
        iteration++;

        i32 prevNodeCount = nodes.size();

        nodeIterator = nodes.begin();

        i32 amountOfNodesToExpand = 0;

        nodesToExpand.clear();

        while (nodeIterator != nodes.end())
        {
            if (nodeIterator->noMoreSubdivision)
            {
                // If node only contains one point do not subdivide and continue
                nodeIterator++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                OrbExtractorNode n1 = {}, n2 = {}, n3 = {}, n4 = {};
                divideOrbExtractorNode(*nodeIterator, n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.keys.size() > 0)
                {
                    nodes.push_front(n1);
                    if (n1.keys.size() > 1)
                    {
                        amountOfNodesToExpand++;
                        nodesToExpand.push_back(std::make_pair(n1.keys.size(), &nodes.front()));
                        nodes.front().iteratorToNode = nodes.begin();
                    }
                }
                if (n2.keys.size() > 0)
                {
                    nodes.push_front(n2);
                    if (n2.keys.size() > 1)
                    {
                        amountOfNodesToExpand++;
                        nodesToExpand.push_back(std::make_pair(n2.keys.size(), &nodes.front()));
                        nodes.front().iteratorToNode = nodes.begin();
                    }
                }
                if (n3.keys.size() > 0)
                {
                    nodes.push_front(n3);
                    if (n3.keys.size() > 1)
                    {
                        amountOfNodesToExpand++;
                        nodesToExpand.push_back(std::make_pair(n3.keys.size(), &nodes.front()));
                        nodes.front().iteratorToNode = nodes.begin();
                    }
                }
                if (n4.keys.size() > 0)
                {
                    nodes.push_front(n4);
                    if (n4.keys.size() > 1)
                    {
                        amountOfNodesToExpand++;
                        nodesToExpand.push_back(std::make_pair(n4.keys.size(), &nodes.front()));
                        nodes.front().iteratorToNode = nodes.begin();
                    }
                }

                nodeIterator = nodes.erase(nodeIterator);
                //continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((i32)nodes.size() >= requiredFeatureCount || (i32)nodes.size() == prevNodeCount)
        {
            finish = true;
        }
        // continue dividing nodes until we have enough nodes to reach the required feature count
        else if (((i32)nodes.size() + amountOfNodesToExpand * 3) > requiredFeatureCount)
        {
            while (!finish)
            {
                prevNodeCount = (i32)nodes.size();

                std::vector<std::pair<i32, OrbExtractorNode*>> previousNodesToExpand = nodesToExpand;
                nodesToExpand.clear();

                std::sort(previousNodesToExpand.begin(), previousNodesToExpand.end());
                for (i32 j = previousNodesToExpand.size() - 1; j >= 0; j--)
                {
                    OrbExtractorNode n1 = {}, n2 = {}, n3 = {}, n4 = {};
                    divideOrbExtractorNode(*previousNodesToExpand[j].second, n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.keys.size() > 0)
                    {
                        nodes.push_front(n1);
                        if (n1.keys.size() > 1)
                        {
                            nodesToExpand.push_back(std::make_pair(n1.keys.size(), &nodes.front()));
                            nodes.front().iteratorToNode = nodes.begin();
                        }
                    }
                    if (n2.keys.size() > 0)
                    {
                        nodes.push_front(n2);
                        if (n2.keys.size() > 1)
                        {
                            nodesToExpand.push_back(std::make_pair(n2.keys.size(), &nodes.front()));
                            nodes.front().iteratorToNode = nodes.begin();
                        }
                    }
                    if (n3.keys.size() > 0)
                    {
                        nodes.push_front(n3);
                        if (n3.keys.size() > 1)
                        {
                            nodesToExpand.push_back(std::make_pair(n3.keys.size(), &nodes.front()));
                            nodes.front().iteratorToNode = nodes.begin();
                        }
                    }
                    if (n4.keys.size() > 0)
                    {
                        nodes.push_front(n4);
                        if (n4.keys.size() > 1)
                        {
                            nodesToExpand.push_back(std::make_pair(n4.keys.size(), &nodes.front()));
                            nodes.front().iteratorToNode = nodes.begin();
                        }
                    }

                    nodes.erase(previousNodesToExpand[j].second->iteratorToNode);

                    if ((i32)nodes.size() >= requiredFeatureCount) break;
                }

                if ((i32)nodes.size() >= requiredFeatureCount || (i32)nodes.size() == prevNodeCount)
                {
                    finish = true;
                }
            }
        }
    }

    // Retain the best point in each node
    std::vector<cv::KeyPoint> result;
    result.reserve(nodes.size());
    for (std::list<OrbExtractorNode>::iterator nodeIterator = nodes.begin();
         nodeIterator != nodes.end();
         nodeIterator++)
    {
        std::vector<cv::KeyPoint>& vNodeKeys   = nodeIterator->keys;
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

        result.push_back(*pKP);
    }

    return result;
}

void computePyramidOctTree(const ImagePyramidStats&    imagePyramidStats,
                           const std::vector<cv::Mat>& imagePyramid,
                           const i32                   edgeThreshold,
                           PyramidOctTree*             octTree)
{
    octTree->levels.resize(imagePyramidStats.numberOfScaleLevels);

    const r32 cellCount = 30;

    for (i32 level = 0; level < imagePyramidStats.numberOfScaleLevels; level++)
    {
        PyramidOctTreeLevel* octTreeLevel = &octTree->levels[level];
        octTreeLevel->cells.clear();

        octTreeLevel->minBorderX = edgeThreshold - 3;
        octTreeLevel->minBorderY = octTreeLevel->minBorderX;
        octTreeLevel->maxBorderX = imagePyramid[level].cols - edgeThreshold + 3;
        octTreeLevel->maxBorderY = imagePyramid[level].rows - edgeThreshold + 3;

        //std::vector<cv::KeyPoint> keyPointsToDistribute;
        //keyPointsToDistribute.reserve(fastFeatureConstraints.numberOfFeatures * 10);

        const r32 width  = (octTreeLevel->maxBorderX - octTreeLevel->minBorderX);
        const r32 height = (octTreeLevel->maxBorderY - octTreeLevel->minBorderY);

        const i32 columnCount = width / cellCount;
        const i32 rowCount    = height / cellCount;
        const i32 cellWidth   = ceil(width / columnCount);
        const i32 cellHeight  = ceil(height / rowCount);

        for (i32 i = 0; i < rowCount; i++)
        {
            const r32 iniY = octTreeLevel->minBorderY + i * cellHeight;
            r32       maxY = iniY + cellHeight + 6;

            if (iniY >= octTreeLevel->maxBorderY - 3)
            {
                continue;
            }

            if (maxY > octTreeLevel->maxBorderY)
            {
                maxY = (r32)octTreeLevel->maxBorderY;
            }

            for (i32 j = 0; j < columnCount; j++)
            {
                const r32 iniX = octTreeLevel->minBorderX + j * cellWidth;
                r32       maxX = iniX + cellWidth + 6;
                if (iniX >= octTreeLevel->maxBorderX - 6)
                {
                    continue;
                }

                if (maxX > octTreeLevel->maxBorderX)
                {
                    maxX = (r32)octTreeLevel->maxBorderX;
                }

                PyramidOctTreeCell cell = {};
                cell.minX               = iniX;
                cell.minY               = iniY;
                cell.maxX               = maxX;
                cell.maxY               = maxY;
                cell.xOffset            = j * cellWidth;
                cell.yOffset            = i * cellHeight;
                cell.imagePyramidLevel  = level;

                octTreeLevel->cells.push_back(cell);
            }
        }
    }
}

/**
 * 1. Splits every level of the image into evenly sized cells
 * 2. Detects corners in a 7x7 cell area
 * 3. Make sure key points are well distributed
 * 4. Compute orientation of keypoints
 */
void computeKeyPointsInOctTree(const PyramidOctTree&                   octTree,
                               const ImagePyramidStats&                imagePyramidStats,
                               const std::vector<cv::Mat>&             imagePyramid,
                               const FastFeatureConstraints&           fastFeatureConstraints,
                               const i32                               edgeThreshold,
                               const i32                               patchSize,
                               const i32                               halfPatchSize,
                               const std::vector<i32>                  umax,
                               std::vector<std::vector<cv::KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(imagePyramidStats.numberOfScaleLevels);

    for (i32 level = 0; level < imagePyramidStats.numberOfScaleLevels; level++)
    {
        std::vector<cv::KeyPoint> keyPointsToDistribute;
        keyPointsToDistribute.reserve(fastFeatureConstraints.numberOfFeatures * 10);

        for (i32 cellIndex = 0; cellIndex < octTree.levels[level].cells.size(); cellIndex++)
        {
            const PyramidOctTreeCell* cell = &octTree.levels[level].cells[cellIndex];

            std::vector<cv::KeyPoint> keyPointsInCell;
            cv::FAST(imagePyramid[level].rowRange(cell->minY, cell->maxY).colRange(cell->minX, cell->maxX),
                     keyPointsInCell,
                     fastFeatureConstraints.initialThreshold,
                     true);

            if (keyPointsInCell.empty())
            {
                FAST(imagePyramid[level].rowRange(cell->minY, cell->maxY).colRange(cell->minX, cell->maxX),
                     keyPointsInCell,
                     fastFeatureConstraints.minimalThreshold,
                     true);
            }

            if (!keyPointsInCell.empty())
            {
                for (std::vector<cv::KeyPoint>::iterator vit = keyPointsInCell.begin(); vit != keyPointsInCell.end(); vit++)
                {
                    //(*vit).pt.x += j * cell->width;
                    //(*vit).pt.y += i * cell->height;
                    (*vit).pt.x += cell->xOffset;
                    (*vit).pt.y += cell->yOffset;
                    keyPointsToDistribute.push_back(*vit);
                }
            }
        }

        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(fastFeatureConstraints.numberOfFeatures);

        keypoints = distributeOctTree(keyPointsToDistribute,
                                      octTree.levels[level].minBorderX,
                                      octTree.levels[level].maxBorderX,
                                      octTree.levels[level].minBorderY,
                                      octTree.levels[level].maxBorderY,
                                      imagePyramidStats.numberOfFeaturesPerScaleLevel[level],
                                      level);

        const i32 scaledPatchSize = patchSize * imagePyramidStats.scaleFactors[level];

        // Add border to coordinates and scale information
        for (i32 i = 0; i < keypoints.size(); i++)
        {
            keypoints[i].pt.x += octTree.levels[level].minBorderX;
            keypoints[i].pt.y += octTree.levels[level].minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size   = (r32)scaledPatchSize;
        }

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
