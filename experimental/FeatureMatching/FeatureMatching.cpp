#include <iostream>
#include <stdio.h>
#include "tools.h"
#include "ExtractKeypoints.h"
#include "orb_descriptor.h"
#include "matching.h"
#include "app.h"
#include "gui.h"

void image_descriptors_and_keypoints(App &app)
{
    cv::Mat grayscaleImg;
    std::vector<std::vector<cv::KeyPoint>> all_keypoints;
    std::vector<std::vector<Descriptor>> all_desc;

    //Image 1
    grayscaleImg = to_grayscale(app.image1);

#if EQUALIZE_HIST == 1
    equalizeHist(grayscaleImg, grayscaleImg);
#endif

    build_pyramid(app.image1_pyramid, grayscaleImg, app.pyramid_param);
    KeyPointExtract(all_keypoints, app.image1_pyramid, app.pyramid_param, 20, 7);
    ComputeORBDescriptor(all_desc, app.image1_pyramid, app.pyramid_param, all_keypoints);

    flatten_keypoints(app.keypoints1, all_keypoints, app.pyramid_param);
    flatten_decriptors(app.descs1, all_desc, app.pyramid_param);


    //Image 2
    all_keypoints.clear();
    all_desc.clear();

    grayscaleImg = to_grayscale(app.image2);

#if EQUALIZE_HIST == 1
    equalizeHist(grayscaleImg, grayscaleImg);
#endif

    build_pyramid(app.image2_pyramid, grayscaleImg, app.pyramid_param);
    KeyPointExtract(all_keypoints, app.image2_pyramid, app.pyramid_param, 20, 7);
    ComputeORBDescriptor(all_desc, app.image2_pyramid, app.pyramid_param, all_keypoints);

    flatten_keypoints(app.keypoints2, all_keypoints, app.pyramid_param);
    flatten_decriptors(app.descs2, all_desc, app.pyramid_param);
}

int main(int argc, char** argv)
{
    App app;

    if (argc < 3)
    {
        std::cout << "Usage:" << std::endl << argv[0] << " image1 image2" << std::endl;
        exit(1);
    }

    app.name = "Best tool ever made!";
    app.closeup_left = "closeup left";
    app.closeup_right = "closeup right";
    app.select_radius = 10;
    app.local_idx = 0;
    app.right_idx = 0;
    app.left_idx = 0;

    app.image1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    app.image2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    if (!app.image1.data || !app.image2.data)
    {
        std::cout << "Usage:" << std::endl << argv[0] << " image1 image2" << std::endl;
        std::cout << "Can't open images" << std::endl;
        exit(1);
    }

    init_pyramid_parameters(app.pyramid_param, 3, 1.2, 1000);

    image_descriptors_and_keypoints(app);
    match_keypoints_1(app.matching_2_1, app.keypoints1, app.descs1, app.keypoints2, app.descs2, true);
    app.matching_1_2 = get_inverted_matching(app.matching_2_1, app.keypoints1.size());
    
    start_gui(app);

    return 0;
}


