#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

#include <WAIHelper.h>
#include <WAIMode.h>
#include <WAISensorCamera.h>
#include <WAIKeyFrameDB.h>
#include <WAIMap.h>
#include <WAIOrbVocabulary.h>
#include <WAIFrame.h>

#include <OrbSlam/LocalMapping.h>
#include <OrbSlam/LoopClosing.h>
#include <OrbSlam/Initializer.h>
#include <OrbSlam/ORBmatcher.h>
#include <OrbSlam/Optimizer.h>
#include <OrbSlam/PnPsolver.h>

int main()
{
    std::string dataRoot = std::string(WAI_ROOT) + "/experimental/Initialization/testdata/";

    std::string orbVocFile   = std::string(WAI_ROOT) + "/data/calibrations/ORBvoc.bin";
    int         nFeatures    = 1000;
    float       fScaleFactor = 1.2;
    int         nLevels      = 1;
    int         fIniThFAST   = 20;
    int         fMinThFAST   = 7;

    WAIOrbVocabulary::initialize(orbVocFile);
    ORB_SLAM2::ORBVocabulary* orbVoc = WAIOrbVocabulary::get();

    ORB_SLAM2::ORBextractor extractor = ORB_SLAM2::ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cv::FileStorage fs(dataRoot + "cam_calibration_main.xml", cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        return false;
    }

    cv::Mat cameraMat, distortionMat;

    fs["cameraMat"] >> cameraMat;
    fs["distortion"] >> distortionMat;

    int from_to[] = {0, 0};

    cv::Mat img1     = cv::imread(dataRoot + "chessboard_logitech_01.jpg");
    cv::Mat img1Gray = cv::Mat(img1.rows,
                               img1.cols,
                               CV_8UC1);
    cv::mixChannels(&img1, 1, &img1Gray, 1, from_to, 1);

    cv::Mat img2     = cv::imread(dataRoot + "chessboard_logitech_02.jpg");
    cv::Mat img2Gray = cv::Mat(img2.rows,
                               img2.cols,
                               CV_8UC1);
    cv::mixChannels(&img2, 1, &img2Gray, 1, from_to, 1);

    int flags =
      //CALIB_CB_ADAPTIVE_THRESH |
      //CALIB_CB_NORMALIZE_IMAGE |
      cv::CALIB_CB_FAST_CHECK;
    cv::Size chessboardSize(8, 5);

    std::vector<cv::Point2f> p2D1;
    bool                     found1 = cv::findChessboardCorners(img1Gray,
                                            chessboardSize,
                                            p2D1,
                                            flags);

    if (found1)
    {
        cv::drawChessboardCorners(img1, chessboardSize, p2D1, found1);
    }

    std::vector<cv::Point2f> p2D2;
    bool                     found2 = cv::findChessboardCorners(img2Gray,
                                            chessboardSize,
                                            p2D2,
                                            flags);

    if (found2)
    {
        cv::drawChessboardCorners(img2, chessboardSize, p2D2, found2);
    }

    if (found1 && found2)
    {
        WAIFrame frame1 = WAIFrame(img1Gray, 0, &extractor, cameraMat, distortionMat, orbVoc);
        WAIFrame frame2 = WAIFrame(img2Gray, 0, &extractor, cameraMat, distortionMat, orbVoc);

        std::vector<cv::Point3f> p3Dw;

        float chessboardWidthMM = 42.0f;
        for (int y = 0; y < chessboardSize.height; y++)
        {
            for (int x = 0; x < chessboardSize.width; x++)
            {
                p3Dw.push_back(cv::Point3f(y * chessboardWidthMM, x * chessboardWidthMM, 0.0f));
            }
        }

        cv::Mat r1, t1, r2, t2;
        bool    pose1Found = cv::solvePnP(p3Dw,
                                       p2D1,
                                       cameraMat,
                                       distortionMat,
                                       r1,
                                       t1,
                                       false,
                                       cv::SOLVEPNP_ITERATIVE);
        bool    pose2Found = cv::solvePnP(p3Dw,
                                       p2D2,
                                       cameraMat,
                                       distortionMat,
                                       r2,
                                       t2,
                                       false,
                                       cv::SOLVEPNP_ITERATIVE);

        if (pose1Found && pose2Found)
        {
            cv::Mat rotMat1, rotMat2;
            cv::Rodrigues(r1, rotMat1);

            cv::Mat pose1 = cv::Mat::eye(4, 4, CV_32F);
            rotMat1.copyTo(pose1.rowRange(0, 3).colRange(0, 3));
            t1.copyTo(pose1.rowRange(0, 3).col(3));

            frame1.SetPose(pose1);

            std::cout << frame1.mTcw << std::endl;

            cv::Rodrigues(r2, rotMat2);

            cv::Mat pose2 = cv::Mat::eye(4, 4, CV_32F);
            rotMat2.copyTo(pose2.rowRange(0, 3).colRange(0, 3));
            t2.copyTo(pose2.rowRange(0, 3).col(3));

            frame2.SetPose(pose2);

            std::cout << frame2.mTcw << std::endl;

            std::vector<int> matches;

            std::vector<cv::KeyPoint> kp1, kp2;
            for (int i = 0; i < p2D1.size(); i++)
            {
                kp1.push_back(cv::KeyPoint(p2D1[i], 2.0f));
            }

            for (int i = 0; i < p2D2.size(); i++)
            {
                kp2.push_back(cv::KeyPoint(p2D2[i], 2.0f));
            }

            for (int i = 0; i < p2D1.size(); i++)
            {
                matches.push_back(i);
            }

            cv::Mat                  r21, t21;
            std::vector<cv::Point3f> vP3De;
            std::vector<bool>        triangulated;
            ORB_SLAM2::Initializer   initializer(frame1, 1.0f, 200);
            initializer.InitializeWithKnownPose(kp1, kp2, frame1.mTcw, frame2.mTcw, frame1.mK, frame2.mK, matches, r21, t21, vP3De, triangulated);

            for (int i = 0; i < vP3De.size(); i++)
            {
                std::cout << vP3De[i] << std::endl;
            }
        }
    }

    cv::Mat imgConcat;
    cv::hconcat(img1, img2, imgConcat);

    cv::namedWindow("initialization", CV_WINDOW_AUTOSIZE);
    cv::imshow("initialization", imgConcat);

    cv::waitKey(0);

    return 0;
}
