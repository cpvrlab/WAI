#ifndef EXTRACTKEYPOINTS_H
#define EXTRACTKEYPOINTS_H

#include "tools.h"

void KeyPointExtract(std::vector<std::vector<cv::KeyPoint>>& allKeypoints, std::vector<cv::Mat> &image_pyramid, PyramidParameters &p, float iniThFAST, float minThFAST);

#endif

