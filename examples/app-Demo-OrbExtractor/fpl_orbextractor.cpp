#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

#include "wai_orb.cpp"

void cv_extractAndDrawKeyPoints(cv::Mat                    image,
                                std::vector<cv::KeyPoint>& keyPoints,
                                u8**                       descriptors)
{
    cv::Mat grayscaleImg = cv::Mat(image.rows,
                                   image.cols,
                                   CV_8UC1);

    int from_to[] = {0, 0};
    cv::mixChannels(&image, 1, &grayscaleImg, 1, from_to, 1);

    FrameBuffer grayscaleBuffer;
    grayscaleBuffer.memory        = grayscaleImg.data;
    grayscaleBuffer.width         = grayscaleImg.cols;
    grayscaleBuffer.height        = grayscaleImg.rows;
    grayscaleBuffer.bytesPerPixel = 1;
    grayscaleBuffer.pitch         = (i32)grayscaleImg.step;

    // add border to image where we do not want corners to be detected
    const int minBorderX = EDGE_THRESHOLD - 3;
    const int minBorderY = minBorderX;
    const int maxBorderX = grayscaleBuffer.width - EDGE_THRESHOLD + 3;
    const int maxBorderY = grayscaleBuffer.height - EDGE_THRESHOLD + 3;

    FrameBuffer fastBuffer;
    fastBuffer.width         = (maxBorderX - minBorderX);
    fastBuffer.height        = (maxBorderY - minBorderY);
    fastBuffer.bytesPerPixel = 1;
    fastBuffer.pitch         = grayscaleBuffer.pitch;
    fastBuffer.memory        = ((u8*)grayscaleBuffer.memory) +
                        minBorderY * fastBuffer.pitch +
                        minBorderX * fastBuffer.bytesPerPixel;

    keyPoints = detectFastCorners(&fastBuffer,
                                  20);

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    std::vector<int> umax;
    umax.resize(HALF_PATCH_SIZE + 1);

    int          v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int          vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2  = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    for (cv::KeyPoint& keyPoint : keyPoints)
    {
        // Bring point coordinates from fastBuffer space to
        // grayscaleBuffer space
        keyPoint.pt.x += minBorderX;
        keyPoint.pt.y += minBorderY;

        keyPoint.angle = computeKeypointAngle(&grayscaleBuffer,
                                              keyPoint.pt.x,
                                              keyPoint.pt.y,
                                              umax);
    }

    const cv::Point*       pattern0 = (cv::Point*)bit_pattern_31_;
    std::vector<cv::Point> pattern;
    i32                    pointCount = ORB_PATTERN_VALUE_COUNT / 2;
    std::copy(pattern0, pattern0 + pointCount, std::back_inserter(pattern));

    *descriptors = (u8*)malloc(ORB_DESCRIPTOR_COUNT * keyPoints.size() * sizeof(u8));

    u8* keyPointDescriptor = *descriptors;
    for (cv::KeyPoint keyPoint : keyPoints)
    {
        computeOrbDescriptor(&grayscaleBuffer,
                             &keyPoint,
                             &pattern[0],
                             keyPointDescriptor);

        keyPointDescriptor += ORB_DESCRIPTOR_COUNT;
    }

    for (cv::KeyPoint keyPoint : keyPoints)
    {
        cv::rectangle(image,
                      cv::Rect((int)keyPoint.pt.x - 3, (int)keyPoint.pt.y - 3, 7, 7),
                      cv::Scalar(0, 0, 255));
    }
}

int main(int argc, char** argv)
{
    cv::Mat image1, image2;
    image1 = cv::imread("/home/jdellsperger/projects/WAI/data/images/textures/Lena.tiff", CV_LOAD_IMAGE_COLOR);
    image2 = cv::imread("/home/jdellsperger/projects/WAI/data/images/textures/Lena_s.tiff", CV_LOAD_IMAGE_COLOR);

    if (!image1.data || !image2.data)
    {
        printf("Could not load image.\n");
        return -1;
    }

    std::vector<cv::KeyPoint> keyPointsImage1   = std::vector<cv::KeyPoint>();
    std::vector<cv::KeyPoint> keyPointsImage2   = std::vector<cv::KeyPoint>();
    u8*                       descriptorsImage1 = nullptr;
    u8*                       descriptorsImage2 = nullptr;
    cv_extractAndDrawKeyPoints(image1, keyPointsImage1, &descriptorsImage1);
    cv_extractAndDrawKeyPoints(image2, keyPointsImage2, &descriptorsImage2);

    cv::Mat concatenatedImage;
    cv::hconcat(image1, image2, concatenatedImage);

    for (int a = 0; a < keyPointsImage1.size(); a++)
    {
        int minDist      = INT_MAX;
        int minDistIndex = -1;
        for (int b = 0; b < keyPointsImage2.size(); b++)
        {
            int dist = descriptorDistance(&descriptorsImage1[a], &descriptorsImage2[b]);

            if (dist < minDist)
            {
                minDist      = dist;
                minDistIndex = b;
            }
        }

        cv::line(concatenatedImage,
                 keyPointsImage1[a].pt,
                 cv::Point((int)keyPointsImage2[minDistIndex].pt.x + image2.cols, (int)keyPointsImage2[minDistIndex].pt.y),
                 cv::Scalar(255, 0, 0));
    }

    cv::namedWindow("orbextractor", CV_WINDOW_AUTOSIZE);

    cv::imshow("orbextractor", concatenatedImage);

    cv::waitKey(0);

    return 0;
}
