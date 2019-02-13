#include <thread>

#include <DUtils/Random.h>

#include <WAIModeOrbSlam2DataOriented.h>
#include <WAIOrbExtraction.cpp>

#define ROTATION_HISTORY_LENGTH 30
#define MATCHER_DISTANCE_THRESHOLD_LOW 50
#define MATCHER_DISTANCE_THRESHOLD_HIGH 100

typedef std::pair<i32, i32> Match;

void decomposeE(const cv::Mat& E,
                cv::Mat&       R1,
                cv::Mat&       R2,
                cv::Mat&       t)
{
    cv::Mat u, w, vt;
    cv::SVD::compute(E, w, u, vt);

    u.col(2).copyTo(t);
    t = t / cv::norm(t);

    cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
    W.at<float>(0, 1) = -1;
    W.at<float>(1, 0) = 1;
    W.at<float>(2, 2) = 1;

    R1 = u * W * vt;
    if (cv::determinant(R1) < 0)
        R1 = -R1;

    R2 = u * W.t() * vt;
    if (cv::determinant(R2) < 0)
        R2 = -R2;
}

void triangulate(const cv::KeyPoint& kp1,
                 const cv::KeyPoint& kp2,
                 const cv::Mat&      P1,
                 const cv::Mat&      P2,
                 cv::Mat&            x3D)
{
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
    A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
    A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
    A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0, 3) / x3D.at<r32>(3);
}

i32 checkRT(const cv::Mat&                   R,
            const cv::Mat&                   t,
            const std::vector<cv::KeyPoint>& vKeys1,
            const std::vector<cv::KeyPoint>& vKeys2,
            const std::vector<Match>&        vMatches12,
            std::vector<bool32>&             vbMatchesInliers,
            const cv::Mat&                   K,
            std::vector<cv::Point3f>&        vP3D,
            r32                              th2,
            std::vector<bool32>&             vbGood,
            r32&                             parallax)
{
    // Calibration parameters
    const r32 fx = K.at<r32>(0, 0);
    const r32 fy = K.at<r32>(1, 1);
    const r32 cx = K.at<r32>(0, 2);
    const r32 cy = K.at<r32>(1, 2);

    vbGood = std::vector<bool32>(vKeys1.size(), false);
    vP3D.resize(vKeys1.size());

    std::vector<r32> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

    cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3, 4, CV_32F);
    R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    t.copyTo(P2.rowRange(0, 3).col(3));
    P2 = K * P2;

    cv::Mat O2 = -R.t() * t;

    i32 nGood = 0;

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
    {
        if (!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint& kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint& kp2 = vKeys2[vMatches12[i].second];
        cv::Mat             p3dC1;

        triangulate(kp1, kp2, P1, P2, p3dC1);

        if (!isfinite(p3dC1.at<r32>(0)) || !isfinite(p3dC1.at<r32>(1)) || !isfinite(p3dC1.at<r32>(2)))
        {
            vbGood[vMatches12[i].first] = false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        r32     dist1   = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        r32     dist2   = cv::norm(normal2);

        r32 cosParallax = normal1.dot(normal2) / (dist1 * dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if (p3dC1.at<r32>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R * p3dC1 + t;

        if (p3dC2.at<r32>(2) <= 0 && cosParallax < 0.99998)
            continue;

        // Check reprojection error in first image
        r32 im1x, im1y;
        r32 invZ1 = 1.0 / p3dC1.at<r32>(2);
        im1x      = fx * p3dC1.at<r32>(0) * invZ1 + cx;
        im1y      = fy * p3dC1.at<r32>(1) * invZ1 + cy;

        r32 squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

        if (squareError1 > th2)
            continue;

        // Check reprojection error in second image
        r32 im2x, im2y;
        r32 invZ2 = 1.0 / p3dC2.at<r32>(2);
        im2x      = fx * p3dC2.at<r32>(0) * invZ2 + cx;
        im2y      = fy * p3dC2.at<r32>(1) * invZ2 + cy;

        r32 squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

        if (squareError2 > th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<r32>(0), p3dC1.at<r32>(1), p3dC1.at<r32>(2));
        nGood++;

        if (cosParallax < 0.99998)
            vbGood[vMatches12[i].first] = true;
    }

    if (nGood > 0)
    {
        sort(vCosParallax.begin(), vCosParallax.end());

        size_t idx = std::min(50, i32(vCosParallax.size() - 1));
        parallax   = acos(vCosParallax[idx]) * 180 / CV_PI;
    }
    else
    {
        parallax = 0;
    }

    return nGood;
}

bool reconstructF(const std::vector<Match>&        matches,
                  const std::vector<cv::KeyPoint>& keyPoints1,
                  const std::vector<cv::KeyPoint>& keyPoints2,
                  const r32                        sigma,
                  std::vector<bool32>&             vbMatchesInliers,
                  cv::Mat&                         F21,
                  cv::Mat&                         K,
                  cv::Mat&                         R21,
                  cv::Mat&                         t21,
                  std::vector<cv::Point3f>&        vP3D,
                  std::vector<bool32>&             vbTriangulated,
                  float                            minParallax,
                  int                              minTriangulated)
{
    int N = 0;
    for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
        if (vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t() * F21 * K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    decomposeE(E21, R1, R2, t);

    cv::Mat t1 = t;
    cv::Mat t2 = -t;

    // Reconstruct with the 4 hyphoteses and check
    std::vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    std::vector<bool32>      vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
    float                    parallax1, parallax2, parallax3, parallax4;

    r32 sigmaSq = sigma * sigma;
    int nGood1  = checkRT(R1, t1, keyPoints1, keyPoints2, matches, vbMatchesInliers, K, vP3D1, 4.0 * sigmaSq, vbTriangulated1, parallax1);
    int nGood2  = checkRT(R2, t1, keyPoints1, keyPoints2, matches, vbMatchesInliers, K, vP3D2, 4.0 * sigmaSq, vbTriangulated2, parallax2);
    int nGood3  = checkRT(R1, t2, keyPoints1, keyPoints2, matches, vbMatchesInliers, K, vP3D3, 4.0 * sigmaSq, vbTriangulated3, parallax3);
    int nGood4  = checkRT(R2, t2, keyPoints1, keyPoints2, matches, vbMatchesInliers, K, vP3D4, 4.0 * sigmaSq, vbTriangulated4, parallax4);

    int maxGood = std::max(nGood1, std::max(nGood2, std::max(nGood3, nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = std::max(static_cast<int>(0.9 * N), minTriangulated);

    int nsimilar = 0;
    if (nGood1 > 0.7 * maxGood)
        nsimilar++;
    if (nGood2 > 0.7 * maxGood)
        nsimilar++;
    if (nGood3 > 0.7 * maxGood)
        nsimilar++;
    if (nGood4 > 0.7 * maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if (maxGood < nMinGood || nsimilar > 1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if (maxGood == nGood1)
    {
        if (parallax1 > minParallax)
        {
            vP3D           = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood2)
    {
        if (parallax2 > minParallax)
        {
            vP3D           = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood3)
    {
        if (parallax3 > minParallax)
        {
            vP3D           = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood4)
    {
        if (parallax4 > minParallax)
        {
            vP3D           = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

bool32 reconstructH(const std::vector<Match>&        matches,
                    const std::vector<cv::KeyPoint>& keyPoints1,
                    const std::vector<cv::KeyPoint>& keyPoints2,
                    const r32                        sigma,
                    std::vector<bool32>&             vbMatchesInliers,
                    cv::Mat&                         H21,
                    cv::Mat&                         K,
                    cv::Mat&                         R21,
                    cv::Mat&                         t21,
                    std::vector<cv::Point3f>&        vP3D,
                    std::vector<bool32>&             vbTriangulated,
                    r32                              minParallax,
                    i32                              minTriangulated)
{
    i32 N = 0;
    for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
    {
        if (vbMatchesInliers[i])
        {
            N++;
        }
    }

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    cv::Mat invK = K.inv();
    cv::Mat A    = invK * H21 * K;

    cv::Mat U, w, Vt, V;
    cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
    V = Vt.t();

    r32 s = cv::determinant(U) * cv::determinant(Vt);

    r32 d1 = w.at<r32>(0);
    r32 d2 = w.at<r32>(1);
    r32 d3 = w.at<r32>(2);

    if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001)
    {
        return false;
    }

    std::vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    r32 aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
    r32 aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
    r32 x1[] = {aux1, aux1, -aux1, -aux1};
    r32 x3[] = {aux3, -aux3, aux3, -aux3};

    //case d'=d2
    r32 aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

    r32 ctheta   = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    r32 stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for (i32 i = 0; i < 4; i++)
    {
        cv::Mat Rp       = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<r32>(0, 0) = ctheta;
        Rp.at<r32>(0, 2) = -stheta[i];
        Rp.at<r32>(2, 0) = stheta[i];
        Rp.at<r32>(2, 2) = ctheta;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<r32>(0) = x1[i];
        tp.at<r32>(1) = 0;
        tp.at<r32>(2) = -x3[i];
        tp *= d1 - d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<r32>(0) = x1[i];
        np.at<r32>(1) = 0;
        np.at<r32>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<r32>(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    //case d'=-d2
    r32 aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

    r32 cphi   = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    r32 sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for (i32 i = 0; i < 4; i++)
    {
        cv::Mat Rp       = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<r32>(0, 0) = cphi;
        Rp.at<r32>(0, 2) = sphi[i];
        Rp.at<r32>(1, 1) = -1;
        Rp.at<r32>(2, 0) = sphi[i];
        Rp.at<r32>(2, 2) = -cphi;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<r32>(0) = x1[i];
        tp.at<r32>(1) = 0;
        tp.at<r32>(2) = x3[i];
        tp *= d1 + d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<r32>(0) = x1[i];
        np.at<r32>(1) = 0;
        np.at<r32>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<r32>(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    i32                      bestGood        = 0;
    i32                      secondBestGood  = 0;
    i32                      bestSolutionIdx = -1;
    r32                      bestParallax    = -1;
    std::vector<cv::Point3f> bestP3D;
    std::vector<bool32>      bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for (size_t i = 0; i < 8; i++)
    {
        r32                      parallaxi;
        std::vector<cv::Point3f> vP3Di;
        std::vector<bool32>      vbTriangulatedi;

        i32 nGood =
          checkRT(vR[i],
                  vt[i],
                  keyPoints1,
                  keyPoints2,
                  matches,
                  vbMatchesInliers,
                  K,
                  vP3Di,
                  4.0 * (sigma * sigma),
                  vbTriangulatedi,
                  parallaxi);

        if (nGood > bestGood)
        {
            secondBestGood   = bestGood;
            bestGood         = nGood;
            bestSolutionIdx  = i;
            bestParallax     = parallaxi;
            bestP3D          = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if (nGood > secondBestGood)
        {
            secondBestGood = nGood;
        }
    }

    if (secondBestGood < 0.75 * bestGood && bestParallax >= minParallax && bestGood > minTriangulated && bestGood > 0.9 * N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D           = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

r32 checkFundamental(const std::vector<Match>&        matches,
                     const std::vector<cv::KeyPoint>& keyPoints1,
                     const std::vector<cv::KeyPoint>& keyPoints2,
                     const cv::Mat&                   F21,
                     const r32                        sigma,
                     std::vector<bool32>&             vbMatchesInliers)
{
    const i32 N = matches.size();

    const r32 f11 = F21.at<r32>(0, 0);
    const r32 f12 = F21.at<r32>(0, 1);
    const r32 f13 = F21.at<r32>(0, 2);
    const r32 f21 = F21.at<r32>(1, 0);
    const r32 f22 = F21.at<r32>(1, 1);
    const r32 f23 = F21.at<r32>(1, 2);
    const r32 f31 = F21.at<r32>(2, 0);
    const r32 f32 = F21.at<r32>(2, 1);
    const r32 f33 = F21.at<r32>(2, 2);

    vbMatchesInliers.resize(N);

    r32 score = 0;

    const r32 th      = 3.841;
    const r32 thScore = 5.991;

    const r32 invSigmaSquare = 1.0 / (sigma * sigma);

    for (i32 i = 0; i < N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint& kp1 = keyPoints1[matches[i].first];
        const cv::KeyPoint& kp2 = keyPoints2[matches[i].second];

        const r32 u1 = kp1.pt.x;
        const r32 v1 = kp1.pt.y;
        const r32 u2 = kp2.pt.x;
        const r32 v2 = kp2.pt.y;

        const r32 a2 = f11 * u1 + f12 * v1 + f13;
        const r32 b2 = f21 * u1 + f22 * v1 + f23;
        const r32 c2 = f31 * u1 + f32 * v1 + f33;

        const r32 num2 = a2 * u2 + b2 * v2 + c2;

        const r32 squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

        const r32 chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        const r32 a1 = f11 * u2 + f21 * v2 + f31;
        const r32 b1 = f12 * u2 + f22 * v2 + f32;
        const r32 c1 = f13 * u2 + f23 * v2 + f33;

        const r32 num1 = a1 * u1 + b1 * v1 + c1;

        const r32 squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

        const r32 chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}

r32 checkHomography(const std::vector<Match>&        matches,
                    const std::vector<cv::KeyPoint>& keyPoints1,
                    const std::vector<cv::KeyPoint>& keyPoints2,
                    const cv::Mat&                   H21,
                    const cv::Mat&                   H12,
                    std::vector<bool32>&             vbMatchesInliers,
                    r32                              sigma)
{
    const i32 N = matches.size();

    const r32 h11 = H21.at<r32>(0, 0);
    const r32 h12 = H21.at<r32>(0, 1);
    const r32 h13 = H21.at<r32>(0, 2);
    const r32 h21 = H21.at<r32>(1, 0);
    const r32 h22 = H21.at<r32>(1, 1);
    const r32 h23 = H21.at<r32>(1, 2);
    const r32 h31 = H21.at<r32>(2, 0);
    const r32 h32 = H21.at<r32>(2, 1);
    const r32 h33 = H21.at<r32>(2, 2);

    const r32 h11inv = H12.at<r32>(0, 0);
    const r32 h12inv = H12.at<r32>(0, 1);
    const r32 h13inv = H12.at<r32>(0, 2);
    const r32 h21inv = H12.at<r32>(1, 0);
    const r32 h22inv = H12.at<r32>(1, 1);
    const r32 h23inv = H12.at<r32>(1, 2);
    const r32 h31inv = H12.at<r32>(2, 0);
    const r32 h32inv = H12.at<r32>(2, 1);
    const r32 h33inv = H12.at<r32>(2, 2);

    vbMatchesInliers.resize(N);

    r32 score = 0;

    const r32 th = 5.991;

    const r32 invSigmaSquare = 1.0 / (sigma * sigma);

    for (i32 i = 0; i < N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint& kp1 = keyPoints1[matches[i].first];
        const cv::KeyPoint& kp2 = keyPoints2[matches[i].second];

        const r32 u1 = kp1.pt.x;
        const r32 v1 = kp1.pt.y;
        const r32 u2 = kp2.pt.x;
        const r32 v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        const r32 w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
        const r32 u2in1    = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
        const r32 v2in1    = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

        const r32 squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

        const r32 chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        const r32 w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
        const r32 u1in2    = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
        const r32 v1in2    = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

        const r32 squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

        const r32 chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += th - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}

cv::Mat computeF21(const std::vector<cv::Point2f>& vP1,
                   const std::vector<cv::Point2f>& vP2)
{
    const i32 N = vP1.size();

    cv::Mat A(N, 9, CV_32F);

    for (i32 i = 0; i < N; i++)
    {
        const r32 u1 = vP1[i].x;
        const r32 v1 = vP1[i].y;
        const r32 u2 = vP2[i].x;
        const r32 v2 = vP2[i].y;

        A.at<r32>(i, 0) = u2 * u1;
        A.at<r32>(i, 1) = u2 * v1;
        A.at<r32>(i, 2) = u2;
        A.at<r32>(i, 3) = v2 * u1;
        A.at<r32>(i, 4) = v2 * v1;
        A.at<r32>(i, 5) = v2;
        A.at<r32>(i, 6) = u1;
        A.at<r32>(i, 7) = v1;
        A.at<r32>(i, 8) = 1;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<r32>(2) = 0;

    cv::Mat result = u * cv::Mat::diag(w) * vt;
    return result;
}

cv::Mat computeH21(const std::vector<cv::Point2f>& vP1,
                   const std::vector<cv::Point2f>& vP2)
{
    const i32 N = vP1.size();

    cv::Mat A(2 * N, 9, CV_32F);

    for (i32 i = 0; i < N; i++)
    {
        const r32 u1 = vP1[i].x;
        const r32 v1 = vP1[i].y;
        const r32 u2 = vP2[i].x;
        const r32 v2 = vP2[i].y;

        A.at<r32>(2 * i, 0) = 0.0;
        A.at<r32>(2 * i, 1) = 0.0;
        A.at<r32>(2 * i, 2) = 0.0;
        A.at<r32>(2 * i, 3) = -u1;
        A.at<r32>(2 * i, 4) = -v1;
        A.at<r32>(2 * i, 5) = -1;
        A.at<r32>(2 * i, 6) = v2 * u1;
        A.at<r32>(2 * i, 7) = v2 * v1;
        A.at<r32>(2 * i, 8) = v2;

        A.at<r32>(2 * i + 1, 0) = u1;
        A.at<r32>(2 * i + 1, 1) = v1;
        A.at<r32>(2 * i + 1, 2) = 1;
        A.at<r32>(2 * i + 1, 3) = 0.0;
        A.at<r32>(2 * i + 1, 4) = 0.0;
        A.at<r32>(2 * i + 1, 5) = 0.0;
        A.at<r32>(2 * i + 1, 6) = -u2 * u1;
        A.at<r32>(2 * i + 1, 7) = -u2 * v1;
        A.at<r32>(2 * i + 1, 8) = -u2;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat result = vt.row(8).reshape(0, 3);
    return result;
}

void normalizeKeyPoints(const std::vector<cv::KeyPoint>& vKeys,
                        std::vector<cv::Point2f>&        vNormalizedPoints,
                        cv::Mat&                         T)
{
    r32       meanX = 0;
    r32       meanY = 0;
    const i32 N     = vKeys.size();

    vNormalizedPoints.resize(N);

    for (i32 i = 0; i < N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX / N;
    meanY = meanY / N;

    r32 meanDevX = 0;
    r32 meanDevY = 0;

    for (i32 i = 0; i < N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX / N;
    meanDevY = meanDevY / N;

    r32 sX = 1.0 / meanDevX;
    r32 sY = 1.0 / meanDevY;

    for (i32 i = 0; i < N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T               = cv::Mat::eye(3, 3, CV_32F);
    T.at<r32>(0, 0) = sX;
    T.at<r32>(1, 1) = sY;
    T.at<r32>(0, 2) = -meanX * sX;
    T.at<r32>(1, 2) = -meanY * sY;
}

void findFundamental(const std::vector<Match>&               matches,
                     const std::vector<cv::KeyPoint>&        keyPoints1,
                     const std::vector<cv::KeyPoint>&        keyPoints2,
                     const i32                               maxRansacIterations,
                     const std::vector<std::vector<size_t>>& ransacSets,
                     const r32                               sigma,
                     r32&                                    score,
                     std::vector<bool32>&                    inlierMatchesFlags,
                     cv::Mat&                                F21)
{
    // Number of putative matches
    const i32 N = inlierMatchesFlags.size();

    // Normalize coordinates
    std::vector<cv::Point2f> vPn1, vPn2;
    cv::Mat                  T1, T2;
    normalizeKeyPoints(keyPoints1, vPn1, T1);
    normalizeKeyPoints(keyPoints2, vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score              = 0.0;
    inlierMatchesFlags = std::vector<bool32>(N, false);

    // Iteration variables
    std::vector<cv::Point2f> vPn1i(8);
    std::vector<cv::Point2f> vPn2i(8);
    cv::Mat                  F21i;
    std::vector<bool32>      vbCurrentInliers(N, false);
    r32                      currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for (i32 it = 0; it < maxRansacIterations; it++)
    {
        // Select a minimum set
        for (i32 j = 0; j < 8; j++)
        {
            i32 idx = ransacSets[it][j];

            vPn1i[j] = vPn1[matches[idx].first];
            vPn2i[j] = vPn2[matches[idx].second];
        }

        cv::Mat Fn = computeF21(vPn1i, vPn2i);

        F21i = T2t * Fn * T1;

        currentScore = checkFundamental(matches,
                                        keyPoints1,
                                        keyPoints2,
                                        F21i,
                                        sigma,
                                        vbCurrentInliers);

        if (currentScore > score)
        {
            F21                = F21i.clone();
            inlierMatchesFlags = vbCurrentInliers;
            score              = currentScore;
        }
    }
}

void findHomography(const std::vector<Match>&               matches,
                    const std::vector<cv::KeyPoint>&        keyPoints1,
                    const std::vector<cv::KeyPoint>&        keyPoints2,
                    const i32                               maxRansacIterations,
                    const std::vector<std::vector<size_t>>& ransacSets,
                    const r32                               sigma,
                    r32&                                    score,
                    std::vector<bool32>&                    inlierMatchesFlags,
                    cv::Mat&                                H21)
{
    // Number of putative matches
    const i32 N = matches.size();

    // Normalize coordinates
    std::vector<cv::Point2f> vPn1, vPn2;
    cv::Mat                  T1, T2;
    normalizeKeyPoints(keyPoints1, vPn1, T1);
    normalizeKeyPoints(keyPoints2, vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    score              = 0.0;
    inlierMatchesFlags = std::vector<bool32>(N, false);

    // Iteration variables
    std::vector<cv::Point2f> vPn1i(8);
    std::vector<cv::Point2f> vPn2i(8);
    cv::Mat                  H21i, H12i;
    std::vector<bool32>      vbCurrentInliers(N, false);
    r32                      currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for (i32 it = 0; it < maxRansacIterations; it++)
    {
        // Select a minimum set
        for (size_t j = 0; j < 8; j++)
        {
            i32 idx = ransacSets[it][j];

            vPn1i[j] = vPn1[matches[idx].first];
            vPn2i[j] = vPn2[matches[idx].second];
        }

        cv::Mat Hn = computeH21(vPn1i, vPn2i);
        H21i       = T2inv * Hn * T1;
        H12i       = H21i.inv();

        currentScore = checkHomography(matches,
                                       keyPoints1,
                                       keyPoints2,
                                       H21i,
                                       H12i,
                                       vbCurrentInliers,
                                       sigma);

        if (currentScore > score)
        {
            H21                = H21i.clone();
            inlierMatchesFlags = vbCurrentInliers;
            score              = currentScore;
        }
    }
}

void computeThreeMaxima(std::vector<i32>* rotationHistory,
                        const i32         historyLength,
                        i32&              ind1,
                        i32&              ind2,
                        i32&              ind3)
{
    i32 max1 = 0;
    i32 max2 = 0;
    i32 max3 = 0;

    for (i32 i = 0; i < historyLength; i++)
    {
        const i32 s = rotationHistory[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (r32)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (r32)max1)
    {
        ind3 = -1;
    }
}

i32 descriptorDistance(const cv::Mat& a,
                       const cv::Mat& b)
{
    const i32* pa = a.ptr<int32_t>();
    const i32* pb = b.ptr<int32_t>();

    i32 dist = 0;

    for (i32 i = 0; i < 8; i++, pa++, pb++)
    {
        u32 v = *pa ^ *pb;
        v     = v - ((v >> 1) & 0x55555555);
        v     = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

bool32 calculateKeyPointGridCell(const cv::KeyPoint& keyPoint,
                                 const r32           minX,
                                 const r32           minY,
                                 const r32           invGridElementWidth,
                                 const r32           invGridElementHeight,
                                 i32*                posX,
                                 i32*                posY)
{
    bool32 result = false;

    i32 x = (i32)round((keyPoint.pt.x - minX) * invGridElementWidth);
    i32 y = (i32)round((keyPoint.pt.y - minY) * invGridElementHeight);

    // Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (x < 0 || x >= FRAME_GRID_COLS || y < 0 || y >= FRAME_GRID_ROWS)
    {
        result = false;
    }
    else
    {
        *posX  = x;
        *posY  = y;
        result = true;
    }

    return result;
}

void undistortKeyPoints(const cv::Mat                   cameraMat,
                        const cv::Mat                   distortionCoefficients,
                        const std::vector<cv::KeyPoint> keyPoints,
                        const i32                       numberOfKeyPoints,
                        std::vector<cv::KeyPoint>&      undistortedKeyPoints)
{
    if (distortionCoefficients.at<r32>(0) == 0.0f)
    {
        undistortedKeyPoints = keyPoints;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(numberOfKeyPoints, 2, CV_32F);
    for (i32 i = 0; i < numberOfKeyPoints; i++)
    {
        mat.at<r32>(i, 0) = keyPoints[i].pt.x;
        mat.at<r32>(i, 1) = keyPoints[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, cameraMat, distortionCoefficients, cv::Mat(), cameraMat);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    undistortedKeyPoints.resize(numberOfKeyPoints);
    for (i32 i = 0; i < numberOfKeyPoints; i++)
    {
        cv::KeyPoint kp         = keyPoints[i];
        kp.pt.x                 = mat.at<r32>(i, 0);
        kp.pt.y                 = mat.at<r32>(i, 1);
        undistortedKeyPoints[i] = kp;
    }
}

void computeScalePyramid(const cv::Mat          image,
                         const i32              numberOfLevels,
                         const std::vector<r32> inverseScaleFactors,
                         const i32              edgeThreshold,
                         std::vector<cv::Mat>&  imagePyramid)
{
    for (i32 level = 0; level < numberOfLevels; level++)
    {
        r32      scale = inverseScaleFactors[level];
        cv::Size sz(cvRound((r32)image.cols * scale), cvRound((r32)image.rows * scale));
        cv::Size wholeSize(sz.width + edgeThreshold * 2, sz.height + edgeThreshold * 2);
        cv::Mat  temp(wholeSize, image.type()), masktemp;
        imagePyramid[level] = temp(cv::Rect(edgeThreshold, edgeThreshold, sz.width, sz.height));

        if (level)
        {
            resize(imagePyramid[level - 1],
                   imagePyramid[level],
                   sz,
                   0,
                   0,
                   CV_INTER_LINEAR);

            copyMakeBorder(imagePyramid[level],
                           temp,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image,
                           temp,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           edgeThreshold,
                           cv::BORDER_REFLECT_101);
        }
    }
}

WAI::ModeOrbSlam2DataOriented::ModeOrbSlam2DataOriented(SensorCamera* camera)
  : _camera(camera)
{
    r32 scaleFactor        = 1.2f;
    i32 pyramidScaleLevels = 8;
    i32 numberOfFeatures   = 2000; // TODO(jan): 2000 for initialization, 1000 otherwise
    i32 orbPatchSize       = 31;
    i32 orbHalfPatchSize   = 15;

    _state.status                  = OrbSlamStatus_Initializing;
    _state.pyramidScaleLevels      = pyramidScaleLevels;
    _state.numberOfFeatures        = numberOfFeatures;
    _state.orbOctTreePatchSize     = orbPatchSize;
    _state.orbOctTreeHalfPatchSize = orbHalfPatchSize;
    _state.initialFastThreshold    = 20;
    _state.minimalFastThreshold    = 7;
    _state.edgeThreshold           = 19;

    const i32        npoints  = 512;
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(_state.pattern));

    _state.pyramidScaleFactors.resize(pyramidScaleLevels);
    _state.inversePyramidScaleFactors.resize(pyramidScaleLevels);
    _state.numberOfFeaturesPerScaleLevel.resize(pyramidScaleLevels);

    _state.pyramidScaleFactors[0]        = 1.0f;
    _state.inversePyramidScaleFactors[0] = 1.0f;

    for (i32 i = 1; i < _state.pyramidScaleLevels; i++)
    {
        _state.pyramidScaleFactors[i]        = _state.pyramidScaleFactors[i - 1] * scaleFactor;
        _state.inversePyramidScaleFactors[i] = 1.0f / _state.pyramidScaleFactors[i];
    }

    r32 inverseScaleFactor            = 1.0f / scaleFactor;
    r32 numberOfFeaturesPerScaleLevel = numberOfFeatures * (1.0f - inverseScaleFactor) / (1.0f - pow((r64)inverseScaleFactor, (r64)pyramidScaleLevels));
    i32 sumFeatures                   = 0;
    for (i32 level = 0; level < pyramidScaleLevels - 1; level++)
    {
        _state.numberOfFeaturesPerScaleLevel[level] = cvRound(numberOfFeaturesPerScaleLevel);
        sumFeatures += _state.numberOfFeaturesPerScaleLevel[level];
        numberOfFeaturesPerScaleLevel *= inverseScaleFactor;
    }
    _state.numberOfFeaturesPerScaleLevel[pyramidScaleLevels - 1] = std::max(numberOfFeatures - sumFeatures, 0);

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    _state.umax.resize(orbHalfPatchSize + 1);
    i32       v, v0;
    i32       vmax = cvFloor(orbHalfPatchSize * sqrt(2.f) / 2 + 1);
    i32       vmin = cvCeil(orbHalfPatchSize * sqrt(2.f) / 2);
    const r64 hp2  = orbHalfPatchSize * orbHalfPatchSize;
    for (v = 0; v <= vmax; v++)
    {
        _state.umax[v] = cvRound(sqrt(hp2 - v * v));
    }

    // Make sure we are symmetric
    for (v = orbHalfPatchSize, v0 = 0; v >= vmin; v--)
    {
        while (_state.umax[v0] == _state.umax[v0 + 1])
        {
            v0++;
        }

        _state.umax[v] = v0;
        v0++;
    }

    _camera->subscribeToUpdate(this);
}

static void initializeKeyFrame(const OrbSlamState*        state,
                               const cv::Mat&             cameraFrame,
                               const cv::Mat&             cameraMat,
                               const cv::Mat&             distortionMat,
                               i32&                       numberOfKeyPoints,
                               std::vector<cv::KeyPoint>& keyPoints,
                               std::vector<cv::KeyPoint>& undistortedKeyPoints,
                               std::vector<size_t>        keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                               cv::Mat&                   descriptors)
{
    numberOfKeyPoints = 0;
    keyPoints.clear();
    undistortedKeyPoints.clear();

    for (i32 i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (i32 j = 0; j < FRAME_GRID_ROWS; j++)
        {
            keyPointIndexGrid[i][j].clear();
        }
    }

    std::vector<cv::Mat> imagePyramid;
    imagePyramid.resize(state->pyramidScaleLevels);

    // Compute scaled images according to scale factors
    computeScalePyramid(cameraFrame,
                        state->pyramidScaleLevels,
                        state->inversePyramidScaleFactors,
                        state->edgeThreshold,
                        imagePyramid);

    // Compute key points, distributed in an evenly spaced grid
    // on every scale level
    std::vector<std::vector<cv::KeyPoint>> allKeyPoints;
    computeKeyPointsInOctTree(state->pyramidScaleLevels,
                              imagePyramid,
                              state->edgeThreshold,
                              state->numberOfFeatures,
                              state->numberOfFeaturesPerScaleLevel,
                              state->initialFastThreshold,
                              state->minimalFastThreshold,
                              state->orbOctTreePatchSize,
                              state->orbOctTreeHalfPatchSize,
                              state->pyramidScaleFactors,
                              state->umax,
                              allKeyPoints);

    for (i32 level = 0; level < state->pyramidScaleLevels; level++)
    {
        numberOfKeyPoints += (i32)allKeyPoints[level].size();
    }

    if (numberOfKeyPoints)
    {
        descriptors.create(numberOfKeyPoints, 32, CV_8U);
    }

    keyPoints.reserve(numberOfKeyPoints);

    i32 offset = 0;
    for (i32 level = 0; level < state->pyramidScaleLevels; ++level)
    {
        i32                        tOffset           = level * 3;
        std::vector<cv::KeyPoint>& keyPointsForLevel = allKeyPoints[level];
        i32                        nkeypointsLevel   = (i32)keyPointsForLevel.size();

        if (nkeypointsLevel == 0) continue;

        // Preprocess the resized image
        cv::Mat workingMat = imagePyramid[level].clone();
        cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        // Compute the descriptors
        cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);

        for (size_t i = 0; i < keyPointsForLevel.size(); i++)
        {
            computeOrbDescriptor(keyPointsForLevel[i], cameraFrame, &state->pattern[0], desc.ptr((i32)i));
        }
        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0)
        {
            r32 scale = state->pyramidScaleFactors[level]; //getScale(level, firstLevel, scaleFactor);
            for (std::vector<cv::KeyPoint>::iterator keypoint    = keyPointsForLevel.begin(),
                                                     keypointEnd = keyPointsForLevel.end();
                 keypoint != keypointEnd;
                 keypoint++)
                keypoint->pt *= scale;
        }

        // Add the keypoints to the output
        keyPoints.insert(keyPoints.end(), keyPointsForLevel.begin(), keyPointsForLevel.end());
    }

    if (!numberOfKeyPoints)
    {
        return;
    }

    undistortKeyPoints(cameraMat,
                       distortionMat,
                       keyPoints,
                       numberOfKeyPoints,
                       undistortedKeyPoints);

    i32 nReserve = 0.5f * numberOfKeyPoints / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (u32 i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (u32 j = 0; j < FRAME_GRID_ROWS; j++)
        {
            keyPointIndexGrid[i][j].reserve(nReserve);
        }
    }

    for (i32 i = 0; i < numberOfKeyPoints; i++)
    {
        const cv::KeyPoint& kp = undistortedKeyPoints[i];

        i32    xPos, yPos;
        bool32 keyPointIsInGrid = calculateKeyPointGridCell(kp, state->minX, state->minY, state->invGridElementWidth, state->invGridElementWidth, &xPos, &yPos);
        if (keyPointIsInGrid)
        {
            keyPointIndexGrid[xPos][yPos].push_back(i);
        }
    }

    // TODO(jan): 'retain image' functionality
}

static std::vector<size_t> getFeatureIndicesForArea(const i32                       numberOfKeyPoints,
                                                    const r32                       searchWindowSize,
                                                    const r32                       x,
                                                    const r32                       y,
                                                    const r32                       minX,
                                                    const r32                       minY,
                                                    const r32                       invGridElementWidth,
                                                    const r32                       invGridElementHeight,
                                                    const i32                       minLevel,
                                                    const i32                       maxLevel,
                                                    const std::vector<size_t>       keyPointIndexGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                                                    const std::vector<cv::KeyPoint> undistortedKeyPoints)
{
    std::vector<size_t> result;

    result.reserve(numberOfKeyPoints);

    const i32 nMinCellX = std::max(0, (i32)floor((x - minX - searchWindowSize) * invGridElementWidth));
    if (nMinCellX >= FRAME_GRID_COLS)
        return result;

    const i32 nMaxCellX = std::min((i32)FRAME_GRID_COLS - 1, (i32)ceil((x - minX + searchWindowSize) * invGridElementWidth));
    if (nMaxCellX < 0)
        return result;

    const i32 nMinCellY = std::max(0, (i32)floor((y - minY - searchWindowSize) * invGridElementHeight));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return result;

    const i32 nMaxCellY = std::min((i32)FRAME_GRID_ROWS - 1, (i32)ceil((y - minY + searchWindowSize) * invGridElementHeight));
    if (nMaxCellY < 0)
        return result;

    const bool32 checkLevels = (minLevel > 0) || (maxLevel >= 0);

    for (i32 ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (i32 iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const std::vector<size_t> vCell = keyPointIndexGrid[ix][iy];

            if (vCell.empty()) continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint& kpUn = undistortedKeyPoints[vCell[j]];
                if (checkLevels)
                {
                    if (kpUn.octave < minLevel) continue;
                    if (maxLevel >= 0 && kpUn.octave > maxLevel) continue;
                }

                const r32 distx = kpUn.pt.x - x;
                const r32 disty = kpUn.pt.y - y;

                if (fabs(distx) < searchWindowSize && fabs(disty) < searchWindowSize)
                {
                    result.push_back(vCell[j]);
                }
            }
        }
    }

    return result;
}

void WAI::ModeOrbSlam2DataOriented::notifyUpdate()
{
    switch (_state.status)
    {
        case OrbSlamStatus_Initializing:
        {
            cv::Mat cameraMat     = _camera->getCameraMatrix();
            cv::Mat distortionMat = _camera->getDistortionMatrix();
            cv::Mat cameraFrame   = _camera->getImageGray();

            if (!_state.referenceKeyFrame)
            {
                if (distortionMat.at<r32>(0) != 0.0)
                {
                    cv::Mat mat(4, 2, CV_32F);
                    mat.at<r32>(0, 0) = 0.0;
                    mat.at<r32>(0, 1) = 0.0;
                    mat.at<r32>(1, 0) = cameraFrame.cols;
                    mat.at<r32>(1, 1) = 0.0;
                    mat.at<r32>(2, 0) = 0.0;
                    mat.at<r32>(2, 1) = cameraFrame.rows;
                    mat.at<r32>(3, 0) = cameraFrame.cols;
                    mat.at<r32>(3, 1) = cameraFrame.rows;

                    // Undistort corners
                    mat = mat.reshape(2);
                    cv::undistortPoints(mat, mat, cameraMat, distortionMat, cv::Mat(), cameraMat);
                    mat = mat.reshape(1);

                    _state.minX = (r32)std::min(mat.at<r32>(0, 0), mat.at<r32>(2, 0));
                    _state.maxX = (r32)std::max(mat.at<r32>(1, 0), mat.at<r32>(3, 0));
                    _state.minY = (r32)std::min(mat.at<r32>(0, 1), mat.at<r32>(1, 1));
                    _state.maxY = (r32)std::max(mat.at<r32>(2, 1), mat.at<r32>(3, 1));
                }
                else
                {
                    _state.minX = 0.0f;
                    _state.maxX = cameraFrame.cols;
                    _state.minY = 0.0f;
                    _state.maxY = cameraFrame.rows;
                }

                _state.invGridElementWidth  = static_cast<r32>(FRAME_GRID_COLS) / static_cast<r32>(_state.maxX - _state.minX);
                _state.invGridElementHeight = static_cast<r32>(FRAME_GRID_ROWS) / static_cast<r32>(_state.maxY - _state.minY);

                _state.fx    = cameraMat.at<r32>(0, 0);
                _state.fy    = cameraMat.at<r32>(1, 1);
                _state.cx    = cameraMat.at<r32>(0, 2);
                _state.cy    = cameraMat.at<r32>(1, 2);
                _state.invfx = 1.0f / _state.fx;
                _state.invfy = 1.0f / _state.fy;

                _state.referenceKeyFrame = new KeyFrame();

                initializeKeyFrame(&_state,
                                   cameraFrame,
                                   cameraMat,
                                   distortionMat,
                                   _state.referenceKeyFrame->numberOfKeyPoints,
                                   _state.referenceKeyFrame->keyPoints,
                                   _state.referenceKeyFrame->undistortedKeyPoints,
                                   _state.referenceKeyFrame->keyPointIndexGrid,
                                   _state.referenceKeyFrame->descriptors);

                if (_state.referenceKeyFrame->numberOfKeyPoints <= 100)
                {
                    delete _state.referenceKeyFrame;
                    _state.referenceKeyFrame = nullptr;
                }
                else
                {
                    _state.previouslyMatchedKeyPoints.resize(_state.referenceKeyFrame->numberOfKeyPoints);
                    for (i32 i = 0; i < _state.referenceKeyFrame->numberOfKeyPoints; i++)
                    {
                        _state.previouslyMatchedKeyPoints[i] = _state.referenceKeyFrame->undistortedKeyPoints[i].pt;
                    }

                    std::fill(_state.initializationMatches.begin(), _state.initializationMatches.end(), -1);
                }
            }
            else
            {
                KeyFrame currentKeyFrame = {};

                initializeKeyFrame(&_state,
                                   cameraFrame,
                                   cameraMat,
                                   distortionMat,
                                   currentKeyFrame.numberOfKeyPoints,
                                   currentKeyFrame.keyPoints,
                                   currentKeyFrame.undistortedKeyPoints,
                                   currentKeyFrame.keyPointIndexGrid,
                                   currentKeyFrame.descriptors);

                if (currentKeyFrame.numberOfKeyPoints > 100)
                {
                    _state.initializationMatches                 = std::vector<i32>(_state.referenceKeyFrame->numberOfKeyPoints, -1);
                    bool32 checkOrientation                      = true;
                    r32    shortestToSecondShortestDistanceRatio = 0.9f;

                    i32 numberOfMatches = 0;

                    std::vector<i32> rotHist[ROTATION_HISTORY_LENGTH];
                    for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
                    {
                        rotHist[i].reserve(500);
                    }

                    const r32 factor = 1.0f / ROTATION_HISTORY_LENGTH;

                    std::vector<i32> matchesDistances(currentKeyFrame.numberOfKeyPoints, INT_MAX);
                    std::vector<i32> matchesKeyPointIndices(currentKeyFrame.numberOfKeyPoints, -1);

                    for (size_t i1 = 0, iend1 = _state.referenceKeyFrame->numberOfKeyPoints;
                         i1 < iend1;
                         i1++)
                    {
                        cv::KeyPoint keyPointReferenceKeyFrame = _state.referenceKeyFrame->undistortedKeyPoints[i1];

                        i32 level1 = keyPointReferenceKeyFrame.octave;
                        if (level1 > 0) continue;

                        std::vector<size_t> keyPointIndicesCurrentFrame =
                          getFeatureIndicesForArea(currentKeyFrame.numberOfKeyPoints,
                                                   100,
                                                   _state.previouslyMatchedKeyPoints[i1].x,
                                                   _state.previouslyMatchedKeyPoints[i1].y,
                                                   _state.minX,
                                                   _state.minY,
                                                   _state.invGridElementWidth,
                                                   _state.invGridElementHeight,
                                                   level1,
                                                   level1,
                                                   currentKeyFrame.keyPointIndexGrid,
                                                   currentKeyFrame.undistortedKeyPoints);

                        if (keyPointIndicesCurrentFrame.empty()) continue;

                        cv::Mat d1 = _state.referenceKeyFrame->descriptors.row(i1);

                        // smaller is better
                        i32 shortestDist       = INT_MAX;
                        i32 secondShortestDist = INT_MAX;
                        i32 shortestDistId     = -1;

                        for (std::vector<size_t>::iterator vit = keyPointIndicesCurrentFrame.begin();
                             vit != keyPointIndicesCurrentFrame.end();
                             vit++)
                        {
                            size_t i2 = *vit;

                            cv::Mat d2 = currentKeyFrame.descriptors.row(i2);

                            i32 dist = descriptorDistance(d1, d2);

                            if (matchesDistances[i2] <= dist) continue;

                            if (dist < shortestDist)
                            {
                                secondShortestDist = shortestDist;
                                shortestDist       = dist;
                                shortestDistId     = i2;
                            }
                            else if (dist < secondShortestDist)
                            {
                                secondShortestDist = dist;
                            }
                        }

                        if (shortestDist <= MATCHER_DISTANCE_THRESHOLD_LOW)
                        {
                            // test that shortest distance is unambiguous
                            if (shortestDist < shortestToSecondShortestDistanceRatio * (r32)secondShortestDist)
                            {
                                // delete previous match, if it exists
                                if (matchesKeyPointIndices[shortestDistId] >= 0)
                                {
                                    i32 previouslyMatchedKeyPointId                           = matchesKeyPointIndices[shortestDistId];
                                    _state.initializationMatches[previouslyMatchedKeyPointId] = -1;
                                    numberOfMatches--;
                                }

                                _state.initializationMatches[i1]       = shortestDistId;
                                matchesKeyPointIndices[shortestDistId] = i1;
                                matchesDistances[shortestDistId]       = shortestDist;
                                numberOfMatches++;

                                if (checkOrientation)
                                {
                                    r32 rot = _state.referenceKeyFrame->undistortedKeyPoints[i1].angle - currentKeyFrame.undistortedKeyPoints[shortestDistId].angle;
                                    if (rot < 0.0) rot += 360.0f;

                                    i32 bin = round(rot * factor);
                                    if (bin == ROTATION_HISTORY_LENGTH) bin = 0;

                                    assert(bin >= 0 && bin < ROTATION_HISTORY_LENGTH);

                                    rotHist[bin].push_back(i1);
                                }
                            }
                        }
                    }

                    if (checkOrientation)
                    {
                        i32 ind1 = -1;
                        i32 ind2 = -1;
                        i32 ind3 = -1;

                        computeThreeMaxima(rotHist, ROTATION_HISTORY_LENGTH, ind1, ind2, ind3);

                        for (i32 i = 0; i < ROTATION_HISTORY_LENGTH; i++)
                        {
                            if (i == ind1 || i == ind2 || i == ind3) continue;

                            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                            {
                                i32 idx1 = rotHist[i][j];
                                if (_state.initializationMatches[idx1] >= 0)
                                {
                                    _state.initializationMatches[idx1] = -1;
                                    numberOfMatches--;
                                }
                            }
                        }
                    }

                    // update prev matched
                    for (size_t i1 = 0, iend1 = _state.initializationMatches.size();
                         i1 < iend1;
                         i1++)
                    {
                        if (_state.initializationMatches[i1] >= 0)
                        {
                            _state.previouslyMatchedKeyPoints[i1] = currentKeyFrame.undistortedKeyPoints[_state.initializationMatches[i1]].pt;
                        }
                    }

                    // Check if there are enough matches
                    if (numberOfMatches >= 100)
                    {
                        printf("Enough matches found\n");
                    }
                    else
                    {
                        delete _state.referenceKeyFrame;
                        _state.referenceKeyFrame = nullptr;
                    }

                    for (u32 i = 0; i < _state.referenceKeyFrame->keyPoints.size(); i++)
                    {
                        cv::rectangle(_camera->getImageRGB(),
                                      _state.referenceKeyFrame->keyPoints[i].pt,
                                      cv::Point(_state.referenceKeyFrame->keyPoints[i].pt.x + 3, _state.referenceKeyFrame->keyPoints[i].pt.y + 3),
                                      cv::Scalar(0, 0, 255));
                    }

                    //ghm1: decorate image with tracked matches
                    for (u32 i = 0; i < _state.initializationMatches.size(); i++)
                    {
                        if (_state.initializationMatches[i] >= 0)
                        {
                            cv::line(_camera->getImageRGB(),
                                     _state.referenceKeyFrame->keyPoints[i].pt,
                                     currentKeyFrame.keyPoints[_state.initializationMatches[i]].pt,
                                     cv::Scalar(0, 255, 0));
                        }
                    }

                    cv::Mat                  rcw;            // Current Camera Rotation
                    cv::Mat                  tcw;            // Current Camera Translation
                    std::vector<bool32>      vbTriangulated; // Triangulated Correspondences (mvIniMatches)
                    std::vector<cv::Point3f> initialPoints;

                    {
                        const i32 maxRansacIterations = 200;
                        const r32 sigma               = 1.0f;

                        std::vector<Match>  matches;
                        std::vector<bool32> matched;

                        matches.reserve(currentKeyFrame.undistortedKeyPoints.size());
                        matched.resize(_state.referenceKeyFrame->undistortedKeyPoints.size());
                        for (size_t i = 0, iend = _state.initializationMatches.size(); i < iend; i++)
                        {
                            if (_state.initializationMatches[i] >= 0)
                            {
                                matches.push_back(std::make_pair(i, _state.initializationMatches[i]));
                                matched[i] = true;
                            }
                            else
                                matched[i] = false;
                        }

                        const i32 N = matches.size();

                        // Indices for minimum set selection
                        std::vector<size_t> vAllIndices;
                        vAllIndices.reserve(N);
                        std::vector<size_t> vAvailableIndices;

                        for (i32 i = 0; i < N; i++)
                        {
                            vAllIndices.push_back(i);
                        }

                        // Generate sets of 8 points for each RANSAC iteration
                        std::vector<std::vector<size_t>> ransacSets = std::vector<std::vector<size_t>>(maxRansacIterations, std::vector<size_t>(8, 0));

                        DUtils::Random::SeedRandOnce(0);

                        for (i32 it = 0; it < maxRansacIterations; it++)
                        {
                            vAvailableIndices = vAllIndices;

                            // Select a minimum set
                            for (size_t j = 0; j < 8; j++)
                            {
                                i32 randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
                                i32 idx   = vAvailableIndices[randi];

                                ransacSets[it][j] = idx;

                                vAvailableIndices[randi] = vAvailableIndices.back();
                                vAvailableIndices.pop_back();
                            }
                        }

                        // Launch threads to compute in parallel a fundamental matrix and a homography
                        std::vector<bool32> vbMatchesInliersH, vbMatchesInliersF;
                        r32                 scoreHomography, scoreFundamental;
                        cv::Mat             H, F;

                        std::thread threadH(&findHomography,
                                            std::ref(matches),
                                            std::ref(_state.referenceKeyFrame->undistortedKeyPoints),
                                            std::ref(currentKeyFrame.undistortedKeyPoints),
                                            maxRansacIterations,
                                            std::ref(ransacSets),
                                            sigma,
                                            std::ref(scoreHomography),
                                            std::ref(vbMatchesInliersH),
                                            std ::ref(H));
                        std::thread threadF(&findFundamental,
                                            std::ref(matches),
                                            std::ref(_state.referenceKeyFrame->undistortedKeyPoints),
                                            std::ref(currentKeyFrame.undistortedKeyPoints),
                                            maxRansacIterations,
                                            std::ref(ransacSets),
                                            sigma,
                                            std::ref(scoreFundamental),
                                            std::ref(vbMatchesInliersF),
                                            std::ref(F));

                        // Wait until both threads have finished
                        threadH.join();
                        threadF.join();

                        // Compute ratio of scores
                        r32 ratioHomography = scoreHomography / (scoreHomography + scoreFundamental);

                        // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
                        bool32 reconstructed = false;
                        if (ratioHomography > 0.40)
                        {
                            reconstructed = reconstructH(matches,
                                                         _state.referenceKeyFrame->undistortedKeyPoints,
                                                         currentKeyFrame.undistortedKeyPoints,
                                                         sigma,
                                                         matched,
                                                         H,
                                                         cameraMat,
                                                         rcw,
                                                         tcw,
                                                         initialPoints,
                                                         vbTriangulated,
                                                         1.0,
                                                         50);
                        }
                        else
                        {
                            reconstructed = reconstructF(matches,
                                                         _state.referenceKeyFrame->undistortedKeyPoints,
                                                         currentKeyFrame.undistortedKeyPoints,
                                                         sigma,
                                                         vbMatchesInliersF,
                                                         F,
                                                         cameraMat,
                                                         rcw,
                                                         tcw,
                                                         initialPoints,
                                                         vbTriangulated,
                                                         1.0,
                                                         50);
                        }

                        if (reconstructed)
                        {
                            printf("Model valid!!!\n");
                        }
                    }
                }
                else
                {
                    delete _state.referenceKeyFrame;
                    _state.referenceKeyFrame = nullptr;
                }
            }
        }
        break;
    }
}
