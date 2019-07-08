#include <iostream>
#include <stdio.h>
#include "tools.h"
#include "ExtractKeypoints.h"
#include "orb_descriptor.h"
#include "matching.h"
#include "app.h"
#include "gui_tools.h"

void image_descriptors_and_keypoints(App &app)
{
    cv::Mat grayscaleImg;
    std::vector<std::vector<cv::KeyPoint>> all_keypoints;
    std::vector<std::vector<Descriptor>> all_desc;

    //Image 1
    grayscaleImg = to_grayscale(app.image1);
    build_pyramid(app.image1_pyramid, grayscaleImg, app.pyramid_param);
    KeyPointExtract(all_keypoints, app.image1_pyramid, app.pyramid_param, 20, 7);
    ComputeORBDescriptor(all_desc, app.image1_pyramid, app.pyramid_param, all_keypoints);

    flatten_keypoints(app.keypoints1, all_keypoints, app.pyramid_param);
    flatten_decriptors(app.descs1, all_desc, app.pyramid_param);


    //Image 2
    all_keypoints.clear();
    all_desc.clear();

    grayscaleImg = to_grayscale(app.image2);
    build_pyramid(app.image2_pyramid, grayscaleImg, app.pyramid_param);
    KeyPointExtract(all_keypoints, app.image2_pyramid, app.pyramid_param, 20, 7);
    ComputeORBDescriptor(all_desc, app.image2_pyramid, app.pyramid_param, all_keypoints);

    flatten_keypoints(app.keypoints2, all_keypoints, app.pyramid_param);
    flatten_decriptors(app.descs2, all_desc, app.pyramid_param);
}


bool sort_fct(cv::KeyPoint &p1, cv::KeyPoint &p2)
{
    return p1.response > p2.response;
}

std::pair<std::string, std::string> make_keypoint_pair_text(App &app, int idx1, int idx2)
{
    std::pair<std::string, std::string> texts;
    std::stringstream ss1;
    std::stringstream ss2;

    Descriptor d1 = app.descs1[idx1];
    Descriptor d2 = app.descs2[idx2];

    ss1 << "Index " << idx1 << std::endl;
    ss2 << "Index " << idx2 << std::endl;
    if (idx2 == app.matching_1_2[idx1])
    {
        ss1 << "Distance " << hamming_distance(d1, d2) << std::endl;
        ss2 << "Distance " << hamming_distance(d1, d2) << std::endl;
    }
    ss1 << "Angle " << keypoint_degree(app.keypoints1[idx1]) << std::endl;
    ss2 << "Angle " << keypoint_degree(app.keypoints2[idx2]) << std::endl;
    ss1 << "Octave " << app.keypoints1[idx1].octave << std::endl;
    ss2 << "Octave " << app.keypoints2[idx2].octave << std::endl;

    texts.first = ss1.str();
    texts.second = ss2.str();

    return texts;
}

void mouse_button_left(int x, int y, int flags, App * app)
{
    reset_color(app->kp1_colors, blue());
    reset_color(app->kp2_colors, blue());

    if (x > app->image1.cols)
    {
        int idx2 = select_closest_feature(app->keypoints2, app->matching_2_1, x - app->image1.cols, y);
        if (idx2 < 0) { return; }

        std::cout << "selected point idx " << idx2 << std::endl;
        int idx1 = app->matching_2_1[idx2];

        app->poi = app->keypoints2[idx2].pt;

        if ((flags & cv::EVENT_FLAG_CTRLKEY) && cv::EVENT_FLAG_CTRLKEY)
        {
            std::pair<std::string, std::string> text = make_keypoint_pair_text(*app, idx1, idx2);
            draw_closeup(app->image1_pyramid[app->keypoints1[idx1].octave], app->keypoints1[idx1], "close up left", text.first);
            draw_closeup(app->image2_pyramid[app->keypoints2[idx2].octave], app->keypoints2[idx2], "close up right", text.second);
        }
        app->kp1_colors[idx1] = red();
        app->kp2_colors[idx2] = red();
    }
    else
    {
        int idx1 = select_closest_feature(app->keypoints1, app->matching_1_2, x, y);
        if (idx1 < 0) { return; }

        std::cout << "selected point idx " << idx1 << std::endl;

        app->poi = app->keypoints1[idx1].pt;
        int idx2 = app->matching_1_2[idx1];

        if ((flags & cv::EVENT_FLAG_CTRLKEY) && cv::EVENT_FLAG_CTRLKEY)
        {
            std::pair<std::string, std::string> text = make_keypoint_pair_text(*app, idx1, idx2);
            draw_closeup(app->image1_pyramid[app->keypoints1[idx1].octave], app->keypoints1[idx1], "close up left", text.first);
            draw_closeup(app->image2_pyramid[app->keypoints2[idx2].octave], app->keypoints2[idx2], "close up right", text.second);
        }
        app->kp1_colors[idx1] = red();
        app->kp2_colors[idx2] = red();
    }
    draw_matches_lines(*app);
    draw_main(*app, "draw matches");
}

void mouse_button_right(int x, int y, int flags, App * app)
{
    reset_color(app->kp1_colors, blue());
    reset_color(app->kp2_colors, blue());

    app->ordered_keypoints1 = app->keypoints1;
    app->ordered_keypoints2 = app->keypoints2;

    std::stringstream ss;
    if (x > app->image1.cols)
    {
        std::vector<int> idxs2 = select_closest_features(app->ordered_keypoints2, app->select_radius, x - app->image1.cols, y);
        int idx2;

        if (idxs2.size() > 1)
        {
            if (app->local_idx >= idxs2.size())
                app->local_idx = 0;

            idx2 = idxs2[app->local_idx++];
        }
        else if (idxs2.size() == 1) { idx2 = idxs2[0]; }
        else { idx2 = select_closest_feature(app->ordered_keypoints2, x - app->image1.cols, y); }

        ss << "Point idx: " << idx2 << std::endl << "octave: " << app->keypoints2[idx2].octave << std::endl;

        if (app->matching_2_1[idx2] >= 0)
            ss << "Has matching to " << app->matching_2_1[idx2] << std::endl;
        draw_closeup(app->image2_pyramid[app->keypoints2[idx2].octave], app->keypoints2[idx2], "close up right", ss.str());
        app->kp2_colors[idx2] = red();

        compute_similarity(app->ordered_keypoints1, app->descs1, app->descs2[idx2]);
        sort(app->ordered_keypoints1.begin(), app->ordered_keypoints1.end(), sort_fct);
        set_color_by_value(app->kp1_colors, app->ordered_keypoints1);
    }
    else
    { 
        std::vector<int> idxs1 = select_closest_features(app->ordered_keypoints1, app->select_radius, x, y);
        int idx1;
        if (idxs1.size() > 1)
        {
            if (app->local_idx >= idxs1.size())
                app->local_idx = 0;

            idx1 = idxs1[app->local_idx++];
        }
        else if (idxs1.size() == 1) { idx1 = idxs1[0]; }
        else { idx1 = select_closest_feature(app->ordered_keypoints1, x, y); }

        ss << "Point idx: " << idx1 << std::endl << "octave: " << app->keypoints1[idx1].octave << std::endl;
        if (app->matching_1_2[idx1] >= 0)
            ss << "Has matching to " << app->matching_1_2[idx1] << std::endl;
        draw_closeup(app->image1_pyramid[app->keypoints1[idx1].octave], app->keypoints1[idx1], "close up left", ss.str());
        app->kp1_colors[idx1] = red();

        compute_similarity(app->ordered_keypoints2, app->descs2, app->descs1[idx1]);
        sort(app->ordered_keypoints2.begin(), app->ordered_keypoints2.end(), sort_fct);
        set_color_by_value(app->kp2_colors, app->ordered_keypoints2);
    }

    draw_by_similarity(*app);
    draw_main(*app, "point similarity");
}

void mouse_events(int event, int x, int y, int flags, void *userdata)
{
    App * app = (App*)userdata;

    switch(event)
    {
        case cv::EVENT_LBUTTONDOWN:
            mouse_button_left(x, y, flags, app);
            break;
        case cv::EVENT_RBUTTONDOWN:
            mouse_button_right(x, y, flags, app);
            break;
    }
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
    app.select_radius = 10;
    app.local_idx = 0;

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
    
    // GUI stuff
    init_color(app.kp1_colors, app.keypoints1.size());
    init_color(app.kp2_colors, app.keypoints2.size());


    cv::namedWindow(app.name, 1);
    cv::setMouseCallback(app.name, mouse_events, &app);

    for(;;)
    {
        int retval = cv::waitKey(0);
        if (retval != 227)
            break;
    }
    return 0;
}


