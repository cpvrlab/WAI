#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>

//Return gx gy magnitude
std::vector<cv::Mat> image_gradient(const cv::Mat &input_rgb_image);

// Return L U V
std::vector<cv::Mat> rgb_to_luv(const cv::Mat &input_color_image);

void filters_open(std::string path, std::vector<float> &param, std::vector<float> &bias, std::vector<std::vector<float>> &coeffs, std::vector<cv::Mat> &filters, std::vector<std::string> &tokens);


std::vector<cv::Point3f> NonMaxSup(const cv::Mat &response);
