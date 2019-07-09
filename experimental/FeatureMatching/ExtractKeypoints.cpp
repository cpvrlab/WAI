#include "ExtractKeypoints.h"
#include <cstdlib>

typedef struct QuadTreeNode
{
    QuadTreeNode(){bNoMore = false;}
    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<QuadTreeNode>::iterator lit;
    bool bNoMore;
}QuadTreeNode;


static void DivideNode(QuadTreeNode * p, QuadTreeNode& n1, QuadTreeNode& n2, QuadTreeNode& n3, QuadTreeNode& n4)
{
    const int halfX = ceil(static_cast<float>(p->UR.x - p->UL.x) / 2);
    const int halfY = ceil(static_cast<float>(p->BR.y - p->UL.y) / 2);

    //Define boundaries of childs
    n1.UL = p->UL;
    n1.UR = cv::Point2i(p->UL.x + halfX, p->UL.y);
    n1.BL = cv::Point2i(p->UL.x, p->UL.y + halfY);
    n1.BR = cv::Point2i(p->UL.x + halfX, p->UL.y + halfY);
    n1.vKeys.reserve(p->vKeys.size());
    n1.bNoMore = false;

    n2.UL = n1.UR;
    n2.UR = p->UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(p->UR.x, p->UL.y + halfY);
    n2.vKeys.reserve(p->vKeys.size());
    n2.bNoMore = false;

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = p->BL;
    n3.BR = cv::Point2i(n1.BR.x, p->BL.y);
    n3.vKeys.reserve(p->vKeys.size());
    n3.bNoMore = false;

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = p->BR;
    n4.vKeys.reserve(p->vKeys.size());
    n4.bNoMore = false;

    //Associate points to childs
    for (size_t i = 0; i < p->vKeys.size(); i++)
    {
        const cv::KeyPoint& kp = p->vKeys[i];

        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
    if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
    if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
    if (n4.vKeys.size() == 1)
        n4.bNoMore = true;
}

static std::vector<cv::KeyPoint> DistributeQuadTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int& minX, const int& maxX, const int& minY, const int& maxY, const int& N, const int& level, int nfeatures)
{
    // Compute how many initial nodes
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

    const float hX = static_cast<float>(maxX - minX) / nIni;

    std::list<QuadTreeNode> lNodes;

    std::vector<QuadTreeNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for (int i = 0; i < nIni; i++)
    {
        QuadTreeNode ni;
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.bNoMore = false;
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    for (size_t i = 0; i < vToDistributeKeys.size(); i++)
    {
        const cv::KeyPoint& kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    std::list<QuadTreeNode>::iterator lit = lNodes.begin();

    while (lit != lNodes.end())
    {
        if (lit->vKeys.size() == 1)
        {
            lit->bNoMore = true;
            lit++;
        }
        else if (lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }


    bool bFinish = false;

    int iteration = 0;

    std::vector<std::pair<int, QuadTreeNode*>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();
        
        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while (lit != lNodes.end())
        {
            if (lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                QuadTreeNode n1, n2, n3, n4;
                DivideNode(&*lit, n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.vKeys.size() > 0)
                {
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n2.vKeys.size() > 0)
                {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0)
                {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0)
                {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit = lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
        {
            bFinish = true;
        }
        else if (((int)lNodes.size() + nToExpand * 3) > N)
        {
            while (!bFinish)
            {
                prevSize = (int)lNodes.size();

                std::vector<std::pair<int, QuadTreeNode*>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());

                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    QuadTreeNode n1, n2, n3, n4;
                    DivideNode(vPrevSizeAndPointerToNode[j].second, n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int)lNodes.size() >= N)
                        break;
                }

                if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                    bFinish = true;
            }
        }
    }

    // Retain the best point in each node
    std::vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for (auto lit = lNodes.begin(); lit != lNodes.end(); lit++)
    {
        std::vector<cv::KeyPoint>& vNodeKeys   = lit->vKeys;
        cv::KeyPoint*         pKP         = &vNodeKeys[0];
        float                 maxResponse = pKP->response;

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

static void IC_Angle(const cv::Mat& image, cv::KeyPoint &kp, const std::vector<int>& u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar>(cvRound(kp.pt.y), cvRound(kp.pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d     = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    kp.angle = atan2((float)m_01, (float)m_10);
}

static void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax)
{
    for (std::vector<cv::KeyPoint>::iterator keypoint    = keypoints.begin(),
                                             keypointEnd = keypoints.end();
         keypoint != keypointEnd;
         ++keypoint)
    {
        IC_Angle(image, *keypoint, umax);
    }
}


/**
* 1. Splits every level of the image into evenly sized cells
* 2. Detects corners in a 7x7 cell area
* 3. Make sure key points are well distributed
* 4. Compute orientation of keypoints
* @param allKeypoints
*/
void KeyPointExtract(std::vector<std::vector<cv::KeyPoint>>& allKeypoints, std::vector<cv::Mat> &image_pyramid, PyramidParameters &p, float iniThFAST, float minThFAST)
{
    allKeypoints.resize(image_pyramid.size());
    std::vector<int> umax;
    init_patch(umax);

    const float W = 30;

    for (int level = 0; level < image_pyramid.size(); ++level)
    {
        const int minBorderX = EDGE_THRESHOLD - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = image_pyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = image_pyramid[level].rows - EDGE_THRESHOLD + 3;

        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(p.total_features * 10);

        const float width  = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; i++)
        {
            const float iniY = minBorderY + i * hCell;
            float       maxY = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3)
                continue;
            if (maxY > maxBorderY)
                maxY = (float)maxBorderY;

            for (int j = 0; j < nCols; j++)
            {
                const float iniX = minBorderX + j * wCell;
                float       maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6)
                    continue;
                if (maxX > maxBorderX)
                    maxX = (float)maxBorderX;

                std::vector<cv::KeyPoint> vKeysCell;
                cv::FAST(image_pyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                        vKeysCell, iniThFAST, true);


                if (vKeysCell.empty())
                {
                    cv::FAST(image_pyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                             vKeysCell, minThFAST, true);
                }

                if (!vKeysCell.empty())
                {
                    for (auto vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                    {
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }

        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(p.total_features);

        keypoints = DistributeQuadTree(vToDistributeKeys,
                                       minBorderX,
                                       maxBorderX,
                                       minBorderY,
                                       maxBorderY,
                                       p.nb_feature_per_level[level],
                                       level,
                                       p.total_features);

        const int scaledPatchSize = PATCH_SIZE * p.scale_factors[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for (int i = 0; i < nkps; i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;

            keypoints[i].size   = (float)scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < p.scale_factors.size(); ++level)
    {
        computeOrientation(image_pyramid[level], allKeypoints[level], umax);
        std::cout << p.scale_factors[level] << std::endl;
    }
}


