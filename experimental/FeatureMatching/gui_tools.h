#ifndef GUI_TOOLS
#define GUI_TOOLS
#include "tools.h"
#include "app.h"

//Color in BGR
cv::Scalar red();
cv::Scalar blue();
cv::Scalar green();

cv::Scalar color_interpolate(cv::Scalar c1, cv::Scalar c2, float v);

void reset_color(std::vector<cv::Scalar> &colors, cv::Scalar col);

void init_color(std::vector<cv::Scalar> &colors, int size);

void set_color_by_value(std::vector<cv::Scalar> &colors, std::vector<cv::KeyPoint> &kps);

void draw_matches_lines(App &app);

void draw_by_similarity(App &app);

void draw_closeup(cv::Mat &image, cv::KeyPoint &kp, std::string window_name, std::string text);

void draw_main(App &app, std::string text);

#endif

