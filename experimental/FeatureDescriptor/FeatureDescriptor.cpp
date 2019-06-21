#include <iostream>
#include <stdio.h>
#include "tools.h"
#include "ExtractKeypoints.h"
#include "orb_descriptor.h"

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

void connect_keypoints(std::vector<int> &indexes,
                       std::vector<cv::KeyPoint> &kps1, std::vector<Descriptor> &desc1, 
                       std::vector<cv::KeyPoint> &kps2, std::vector<Descriptor> &desc2,
                       float thres = 30)
{
    std::vector<int> distances;
    for (int i = 0; i < kps2.size(); i++)
    {
        indexes.push_back(-1);
        distances.push_back(INT_MAX);
    }

    for (int i = 0; i < kps1.size(); i++)
    {
        int min_dist       = INT_MAX;
        int min_dist2      = INT_MAX;
        int min_dist_index = -1;
        int min_dist2_index = -1;
        
        for (int j = 0; j < kps2.size(); j++)
        {
            int dist = hamming_distance(desc1[i], desc2[j]);

            if (dist < min_dist)
            {
                min_dist       = dist;
                min_dist_index = j;
            }
            else if (dist < min_dist2)
            {
                min_dist2 = dist;
                min_dist2_index = j;
            }
        }

        if (min_dist > thres)
            continue;

        //If there is no match yet for this point or the current point is best than the previous, add the matching
        if (indexes[min_dist_index] == -1 || distances[min_dist_index] > min_dist)
        {
            indexes[min_dist_index] = i;
            distances[min_dist_index] = min_dist;
        }
        else 
        {
            //Try to add its second best distance
            if (indexes[min_dist2_index] == -1 || distances[min_dist2_index] > min_dist2)
            {
                indexes[min_dist2_index] = i;
                distances[min_dist2_index] = min_dist2;
            }
        }
    }
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

    init_pyramid_parameters(pyramid_param, 4, 1.2, 1000);
    image1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    image2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    image_descriptors_and_keypoints(keypoints1, desc1, image1, pyramid_param);
    image_descriptors_and_keypoints(keypoints2, desc2, image2, pyramid_param);

    connect_keypoints(indexes, keypoints1, desc1, keypoints2, desc2);

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


