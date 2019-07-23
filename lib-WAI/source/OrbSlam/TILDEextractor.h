/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef TILDEEXTRACTOR_H
#define TILDEEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>
#include <KPextractor.h>


namespace ORB_SLAM2
{

class TILDEextractor : public KPextractor
{
public:

    TILDEextractor(std::string filterpath);

    ~TILDEextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray _imageRGB,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    std::vector<cv::Mat> mvImagePyramid;

protected:

    void computeKeyPoint(std::vector<cv::KeyPoint>& allKeypoints, cv::Mat image);
    std::vector<cv::Point> pattern;

    std::vector<float> param;
    std::vector<float> bias;
    std::vector<std::vector<float>> coeffs;
    std::vector<cv::Mat> filters;
    std::vector<std::string> tokens;
};

} //namespace ORB_SLAM

#endif

