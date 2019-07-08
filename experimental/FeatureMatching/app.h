#ifndef APP
#define APP

typedef struct App
{
    std::string name;
    cv::Mat image1;
    cv::Mat image2;

    PyramidParameters pyramid_param;
    std::vector<cv::Mat> image1_pyramid;
    std::vector<cv::Mat> image2_pyramid;

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<Descriptor> descs1;
    std::vector<Descriptor> descs2;
    std::vector<int> matching_2_1;
    std::vector<int> matching_1_2;

    std::vector<cv::Scalar> kp1_colors;
    std::vector<cv::Scalar> kp2_colors;

    std::vector<cv::KeyPoint> ordered_keypoints1;
    std::vector<cv::KeyPoint> ordered_keypoints2;

    cv::Mat out_image;

    cv::Point poi;
    int local_idx;
    float select_radius;
}App;

#endif

