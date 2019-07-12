#ifndef CONVERT_H
#define CONVERT_H

#include "tools.h"

cv::Mat rgb_to_grayscale(cv::Mat &img);

std::vector<cv::Mat> rgb_to_luv(const cv::Mat &input_color_image);

#endif

