#ifndef WAI_ORB_EXTRACTION_H
#define WAI_ORB_EXTRACTION_H

#include <WAIPlatform.h>

#include <opencv2/core/core.hpp>
#include <list>

struct OrbExtractorNode
{
    std::vector<cv::KeyPoint>             keys;
    cv::Point2i                           topLeft, topRight, bottomLeft, bottomRight;
    std::list<OrbExtractorNode>::iterator lit;
    bool32                                noMoreSubdivision;
};

#endif