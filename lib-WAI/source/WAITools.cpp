#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <WAIHelper.h>
#include <WAI.h>

//Return gx gy magnitude
std::vector<cv::Mat> image_gradient(const cv::Mat &input_rgb_image)
{
    //the output
    std::vector<cv::Mat> gradImage(3);
    std::vector<cv::Mat> color_channels(3);
    std::vector<cv::Mat> gx(3);
    std::vector<cv::Mat> gy(3);

    // The derivative5 kernels
    cv::Mat d1 = (cv::Mat_ <float>(1, 5) << 0.109604, 0.276691, 0.000000, -0.276691, -0.109604);
    cv::Mat d1T = (cv::Mat_ <float>(5, 1) << 0.109604, 0.276691, 0.000000, -0.276691, -0.109604);
    cv::Mat p = (cv::Mat_ <float>(1, 5) << 0.037659, 0.249153, 0.426375, 0.249153, 0.037659);
    cv::Mat pT = (cv::Mat_ <float>(5, 1) << 0.037659, 0.249153, 0.426375, 0.249153, 0.037659);

    // split the channels into each color channel
    cv::split(input_rgb_image, color_channels);
    // prepare output
    for (int idx = 0; idx < 3; ++idx) {
        gradImage[idx].create(color_channels[0].rows, color_channels[0].cols, CV_32F);
    }
    //	return gradImage;

    // for each channel do the derivative 5
    for (int idxC = 0; idxC < 3; ++idxC) {
        cv::sepFilter2D(color_channels[idxC], gx[idxC], CV_32F, d1, p, cv::Point(-1, -1), 0,
                cv::BORDER_REFLECT);
        cv::sepFilter2D(color_channels[idxC], gy[idxC], CV_32F, p, d1, cv::Point(-1, -1), 0,
                cv::BORDER_REFLECT);
        // since we do the other direction, just flip signs
        gx[idxC] = -gx[idxC];
        gy[idxC] = -gy[idxC];
    }

    // the magnitude image
    std::vector<cv::Mat> mag(3);
    for (int idxC = 0; idxC < 3; ++idxC) {
        cv::sqrt(gx[idxC].mul(gx[idxC]) + gy[idxC].mul(gy[idxC]), mag[idxC]);
    }

    // Keep only max from each color component (based on magnitude)
    float curVal, maxVal; int maxIdx;
    for (int i = 0; i < mag[0].rows; i++)
    {
        float* pixelin1[3];
        float* pixelin2[3];
        float* pixelin3[3];

        for (int idxC = 0; idxC < 3; ++idxC)
        {
            pixelin1[idxC] = gx[idxC].ptr<float>(i);  // point to first color in row
            pixelin2[idxC] = gy[idxC].ptr<float>(i);  // point to first color in row
            pixelin3[idxC] = mag[idxC].ptr<float>(i);  // point to first color in row
        }

        float* pixelout1 = gradImage[0].ptr<float>(i);  // point to first color in row
        float* pixelout2 = gradImage[1].ptr<float>(i);  // point to first color in row
        float* pixelout3 = gradImage[2].ptr<float>(i);  // point to first color in row

        for (int j = 0; j < mag[0].cols; j++)
        {
            maxIdx = 0;
            maxVal = 0;
            for (int idxC = 0; idxC < 3; ++idxC)
            {
                curVal = *pixelin3[idxC];
                if (maxVal < curVal) {
                    maxIdx = idxC;
                    maxVal = curVal;
                }
            }
            *pixelout1++ = *pixelin1[maxIdx] * 0.5 + 128.0;
            *pixelout2++ = *pixelin2[maxIdx] * 0.5 + 128.0;
            *pixelout3++ = *pixelin3[maxIdx];

            //next in
            for (int idxC = 0; idxC < 3; ++idxC)
            {
                pixelin1[idxC]++;
                pixelin2[idxC]++;
                pixelin3[idxC]++;
            }
        }
    }

    return gradImage;
}

// Return L U V
std::vector<cv::Mat> rgb_to_luv(const cv::Mat &input_color_image)
{
    std::vector<cv::Mat> luvImage(3);
    for (int idxC = 0; idxC < 3; ++idxC) {
        luvImage[idxC].create(input_color_image.rows, input_color_image.cols, CV_32F);
    }

    //init
    const float y0 =(float) ((6.0/29)*(6.0/29)*(6.0/29));
    const float a = (float) ((29.0/3)*(29.0/3)*(29.0/3));
    const double XYZ[3][3] = {  {  0.430574,  0.341550,  0.178325 },
                                {  0.222015,  0.706655,  0.071330 },
                                {  0.020183,  0.129553,  0.939180 }   };

    const double Un_prime = 0.197833;
    const double Vn_prime = 0.468331;
    const double maxi     = 1.0/270;
    const double minu     = -88*maxi;
    const double minv     = -134*maxi;
    const double Lt       = 0.008856;
    static float lTable[1064];
    for(int i=0; i<1025; i++)
    {
        float y = (float) (i/1024.0);
        float l = y>y0 ? 116*(float)pow((double)y,1.0/3.0)-16 : y*a;
        lTable[i] = l*maxi;
    }

    cv::Mat in(input_color_image);

    cv::Mat out1(luvImage[0]);
    cv::Mat out2(luvImage[1]);
    cv::Mat out3(luvImage[2]);

    for (int i = 0; i < in.rows; i++)
    {
        uchar* pixelin = in.ptr<uchar>(i);  // point to first color in row
        float* pixelout1 = out1.ptr<float>(i);  // point to first color in row
        float* pixelout2 = out2.ptr<float>(i);  // point to first color in row
        float* pixelout3 = out3.ptr<float>(i);  // point to first color in row
        for (int j = 0; j < in.cols; j++)//row
        {
            //cv::Vec3b rgb = in.at<cv::Vec3b>(j,i);
            float b = *pixelin++ / 255.0f;
            float g = *pixelin++ / 255.0f;
            float r = *pixelin++ / 255.0f;

            //RGB to LUV conversion

            //delcare variables
            float  x, y, z, u_prime, v_prime, constant, L, u, v;

            //convert RGB to XYZ...
            x       = XYZ[0][0]*r + XYZ[0][1]*g + XYZ[0][2]*b;
            y       = XYZ[1][0]*r + XYZ[1][1]*g + XYZ[1][2]*b;
            z       = XYZ[2][0]*r + XYZ[2][1]*g + XYZ[2][2]*b;

            //convert XYZ to LUV...

            //compute ltable(y*1024)
            L = lTable[(int)(y*1024)];

            //compute u_prime and v_prime
            constant    = 1/(x + 15 * y + 3 * z + 1e-35);   //=z

            u_prime = (4 * x) * constant;   //4*x*z
            v_prime = (9 * y) * constant;

            //compute u* and v*
            u = (float) (13 * L * (u_prime - Un_prime)) - minu;
            v = (float) (13 * L * (v_prime - Vn_prime)) - minv;

            *pixelout1++ = L*270*2.55;
            *pixelout2++ = ((u*270-88)+ 134.0)* 255.0 / 354.0;
            *pixelout3++ = ((v*270-134)+ 140.0)* 255.0 / 256.0;
        }
    }
    return luvImage;
}

static void Tokenize(const std::string &mystring, std::vector<std::string> &tok,
              const std::string &sep = " ", int lp = 0, int p = 0)
{
    lp = mystring.find_first_not_of(sep, p);
    p = mystring.find_first_of(sep, lp);
    if (std::string::npos != p || std::string::npos != lp) {
        tok.push_back(mystring.substr(lp, p - lp));
        Tokenize(mystring, tok, sep, lp, p);
    }
}

static std::string delSpaces(std::string & str)
{
    std::stringstream trim;
    trim << str;
    trim >> str;
    return str;
}

void filters_open(std::string path, std::vector<float> &param, std::vector<float> &bias, std::vector<std::vector<float>> &coeffs, std::vector<cv::Mat> &filters, std::vector<std::string> &tokens)
{
    //std::ifstream fic(path, std::ios::in);
    std::ifstream fic(path);
    std::string lineread;

    if (!fic.is_open()) {

       WAI_LOG("AAAAAAAAAA %s", path);
    }

    getline(fic, lineread);

    param.clear();
    tokens.clear();
    Tokenize(lineread, tokens, " ");

    for (int i = 0; i < tokens.size(); i++) {
        param.push_back(stof(delSpaces(tokens[i])));
    }

    getline(fic, lineread);

    tokens.clear();
    Tokenize(lineread, tokens);

    if (tokens.size() != 5) {
        std::cout << "Cannot open filter " << path << std::endl;
    }
    int nbMax = stoi(delSpaces(tokens[0]));
    int nbSum = stoi(delSpaces(tokens[1]));
    int nbOriginalFilters = nbMax * nbSum;
    int nbApproximatedFilters = stoi(delSpaces(tokens[2]));
    int nbChannels = stoi(delSpaces(tokens[3]));
    int sizeFilters = stoi(delSpaces(tokens[4]));

    param.push_back(nbMax);
    param.push_back(nbSum);
    param.push_back(nbApproximatedFilters);
    param.push_back(nbChannels);
    param.push_back(sizeFilters);

    //get bias
    getline(fic, lineread);
    tokens.clear();
    Tokenize(lineread, tokens);
    if (tokens.size() != nbOriginalFilters) {
        std::cout << "Wrong number of cascades" << std::endl;
    }
    //bias
    bias.resize(nbOriginalFilters);
    for (int i = 0; i < tokens.size(); i++)
        bias[i] = stof(delSpaces(tokens[i]));

    //coeffs
    coeffs = std::vector<std::vector<float>>(nbOriginalFilters, std::vector<float>(nbApproximatedFilters * nbChannels));
    int row = 0;
    while (getline(fic, lineread)) {
        tokens.clear();
        Tokenize(lineread, tokens);
        for (int i = 0; i < nbApproximatedFilters * nbChannels; i++)
            coeffs[row][i] = stof(delSpaces(tokens[i]));

        if (++row == nbOriginalFilters)
            break;
    }

    filters = std::vector<cv::Mat> (nbApproximatedFilters * nbChannels * 2, cv::Mat(1, sizeFilters, CV_32FC1));
    row = 0;
    while (getline(fic, lineread))
    {
        tokens.clear();
        Tokenize(lineread, tokens);

        std::vector<float>r(sizeFilters);
        for (int i = 0; i < sizeFilters; i++)
            r[i] = stof(delSpaces(tokens[i]));

        filters[row] = cv::Mat(r).clone();

        if (++row == nbApproximatedFilters * nbChannels * 2)
            break;
    }
}

std::vector<cv::Point3f> NonMaxSup(const cv::Mat &response)
{
    std::vector<cv::Point3f> res;

    for(int i=1; i < response.rows-1; ++i)
    {
        for(int j=1; j < response.cols-1; ++j)
        {
            bool bMax = true;

            for(int ii=-1; ii <= +1; ++ii)
            {
                for(int jj=-1; jj <= +1; ++jj)
                {
                    if (ii == 0 && jj == 0)
                        continue;
                    bMax &= response.at<float>(i,j) > response.at<float>(i+ii,j+jj);
                }
            }

            if (bMax)
            {
                res.push_back(cv::Point3f(j,i, response.at<float>(i,j)));
            }
        }
    }

    return res;
}

