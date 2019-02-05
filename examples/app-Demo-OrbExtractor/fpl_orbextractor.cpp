#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

int main(int argc, char** argv)
{
    cv::Mat image;
    image = cv::imread("/home/jdellsperger/projects/WAI/data/images/textures/Lena.tiff", CV_LOAD_IMAGE_COLOR);

    if (!image.data)
    {
        printf("Could not load image.\n");
        return -1;
    }

    cv::namedWindow("orbextractor", CV_WINDOW_AUTOSIZE);
    cv::imshow("orbextractor", image);

    cv::waitKey(0);

    return 0;
}
