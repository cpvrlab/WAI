#ifndef ORB_DESCRIPTOR_H
#define ORB_DESCRIPTOR_H

#include "tools.h"

void ComputeORBDescriptor(std::vector<std::vector<Descriptor>> &descriptors, std::vector<cv::Mat> image_pyramid, PyramidParameters &p, std::vector<std::vector<cv::KeyPoint>>& allKeypoints);

#endif

