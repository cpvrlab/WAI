#include <iostream>
#include <stdio.h>
#include "tools.h"
#include "ExtractKeypoints.h"
#include "orb_descriptor.h"
#include "matching.h"

void image_descriptors_and_keypoints(std::vector<cv::KeyPoint> &keypoints, std::vector<Descriptor> &desc, cv::Mat &image, PyramidParameters &p)
{
    cv::Mat grayscaleImg;
    std::vector<cv::Mat> image_pyramid;
    std::vector<std::vector<cv::KeyPoint>> all_keypoints;
    std::vector<std::vector<Descriptor>> all_desc;

    grayscaleImg = to_grayscale(image);
    build_pyramid(image_pyramid, grayscaleImg, p);

    KeyPointExtract(all_keypoints, image_pyramid, p, 20, 7);
    
    ComputeORBDescriptor(all_desc, image_pyramid, p, all_keypoints);

    flatten_keypoints(keypoints, all_keypoints, p);
    flatten_decriptors(desc, all_desc, p);
}

int main(int argc, char** argv)
{
    PyramidParameters pyramid_param;
    cv::Mat image1;
    cv::Mat image2;
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<Descriptor> desc1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<Descriptor> desc2;
    std::vector<int> indexes;

    if (argc < 3)
    {
        std::cout << "usage:" << std::endl << argv[0] << " image1 image2" << std::endl;
        exit(1);
    }

    init_pyramid_parameters(pyramid_param, 5, 1.2, 1000);
    image1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    image2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    image_descriptors_and_keypoints(keypoints1, desc1, image1, pyramid_param);
    image_descriptors_and_keypoints(keypoints2, desc2, image2, pyramid_param);

    match_keypoints_1(indexes, keypoints1, desc1, keypoints2, desc2, true);

    //Draw keypoints
    cv::drawKeypoints(image1, keypoints1, image1, cv::Scalar(255, 0, 0));
    cv::drawKeypoints(image2, keypoints2, image2, cv::Scalar(255, 0, 0));

    cv::Mat concatenated;
    cv::hconcat(image1, image2, concatenated);

    for (int i = 0; i < indexes.size(); i++)
    {
        if (indexes[i] >= 0)
        {
            cv::Scalar clr(rand() % 255, rand() % 255, rand() % 255); 
            cv::line(concatenated, keypoints2[i].pt + cv::Point2f(image1.cols, 0), keypoints1[indexes[i]].pt, clr);
        }
    }

    cv::imshow("orbextractor", concatenated);
    cv::waitKey(0);

    return 0;
}


