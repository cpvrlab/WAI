#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using namespace std;

class Calib
{
    public:
        Matrix3f intrinsic_matrix;
        int width;
        int height;
        Vector2f K;
        Vector2f P;
        std::vector<std::vector<cv::Vec3f>> _points3d;
        std::vector<std::vector<cv::Vec2f>> _points2d;

        Calib(int width, int height)
        {
            this->width = width;
            this->height = height;
            reset(50, width, height);
        }

        void reset_calibration(float fov, int width, int height)
        { 
            float cx  = width * 0.5f;
            float cy  = height * 0.5f;
            float fy  = cy / tanf(fov * 0.5f * M_PI / 180.0);
            float fx  = fy;

            intrinsic_matrix << fx, 0, cx, 0, fy, cy, 0, 0, 1;
            K << 0, 0;
            P << 0, 0;
        }

        void reset(float fov, int width, int height)
        {
            _points2d.clear();
            _points3d.clear();
            reset_calibration(fov, width, height);
        }

        void feed(std::vector<Vector3f> position, std::vector<Vector2f> projected)
        {
            std::vector<cv::Vec3f> cv_position;
            std::vector<cv::Vec2f> cv_projected;

            for (Vector3f v : position)
                cv_position.push_back(cv::Vec3f(v[0], v[1], v[2]));
            
            for (Vector2f v : projected)
                cv_projected.push_back(cv::Vec2f(v[0], v[1]));

            _points3d.push_back(cv_position);
            _points2d.push_back(cv_projected);
        }

        double reprojection_errors(const std::vector<std::vector<cv::Vec3f>>& objectPoints,
                const std::vector<std::vector<cv::Vec2f>>& imagePoints,
                const std::vector<cv::Mat>&                rvecs,
                const std::vector<cv::Mat>&                tvecs,
                const cv::Mat&                             intrinsic_matrix,
                const cv::Mat&                             distCoeffs)
        {
            std::vector<cv::Point2f> imagePoints2;
            size_t       totalPoints = 0;
            double       totalErr    = 0, err;

            for (size_t i = 0; i < objectPoints.size(); ++i)
            {
                cv::projectPoints(objectPoints[i],
                        rvecs[i],
                        tvecs[i],
                        intrinsic_matrix,
                        distCoeffs,
                        imagePoints2);

                err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);

                size_t n = objectPoints[i].size();
                totalErr += err * err;
                totalPoints += n;
            }

            return std::sqrt(totalErr / totalPoints);
        }

        float get_fovy()
        {
            float fy     = intrinsic_matrix(4);
            float cy     = intrinsic_matrix(7);
            float fov    = 2.0 * atan2(cy, fy);
            return fov * 180.0 / M_PI;
        }

        cv::Mat eigen2cv(Matrix3f m)
        {
            cv::Mat r;
            r = (cv::Mat_<double>(3, 3) << m(0), m(3), m(6), m(1), m(4), m(7), m(2), m(5), m(8));
            return r;
        }

        Matrix3f cv2eigen(cv::Mat m)
        {
            Matrix3f r;

            r << m.at<double>(0, 0), m.at<double>(0,1), m.at<double>(0,2), m.at<double>(1,0), m.at<double>(1,1), m.at<double>(1,2), m.at<double>(2,0), m.at<double>(2,1), m.at<double>(2,2);
            return r;
        }

        float try_calibrate()
        {
            cv::Mat matrix = eigen2cv (intrinsic_matrix);

            cv::Size img_size(width, height);
            cv::Mat distortion = (cv::Mat_<float>(5, 1) << K[0], K[1], P[0], P[1], 0);

            std::vector<cv::Mat> rvecs, tvecs;
            double rms = cv::calibrateCamera(_points3d, _points2d, img_size, matrix, distortion, rvecs, tvecs, cv::CALIB_USE_INTRINSIC_GUESS);

            float totalAvgErr = reprojection_errors(_points3d, _points2d, rvecs, tvecs, matrix, distortion);

            intrinsic_matrix = cv2eigen(matrix);

            K[0] = distortion.at<double>(0,0);
            K[1] = distortion.at<double>(1,0);
            P[0] = distortion.at<double>(2,0);
            P[1] = distortion.at<double>(3,0);

            return totalAvgErr;
        }
};

class Camera
{
    public:
        Matrix4f extrinsic_matrix;
        Matrix3f intrinsic_matrix;
        Vector2f c;
        Vector2f K;
        Vector2f P;
        Vector3f pos;
        float fov;
        int width;
        int height;
        Camera() { }

        void look_at(Vector3f pos, Vector3f up, Vector3f point)
        {
            this->pos = pos;
            Vector3f dir = (point - pos).normalized();
            Vector3f right = dir.cross(up).normalized();
            up = right.cross(dir).normalized();

            extrinsic_matrix << right[0], right[1], right[2], -right.dot(pos),
                                up[0], up[1], up[2], -up.dot(pos),
                                -dir[0], -dir[1], -dir[2], dir.dot(pos),
                                0, 0, 0, 1;
        }

        void set_intrinsic(float fovy, int img_width, int img_height, float c0, float c1)
        {
            fov = fovy;
            float fov_rad = fovy * M_PI / 180.0;
            c[0] = c0;
            c[1] = c1;
            float fy = c[1] / tanf(fov_rad * 0.5f);
            float fx = fy;
            width = img_width;
            height = img_height;

            intrinsic_matrix << fx, 0 , c[0],
                                0,  fy, c[1],
                                0,  0 , 1;
        }

        void set_intrinsic(float fovy, int img_width, int img_height)
        {
            c[0] = img_width * 0.5f;
            c[1] = img_height * 0.5f;
            set_intrinsic(fovy, img_width, img_height, c[0], c[1]);
        }

        void set_distortion(float k1, float k2, float p1, float p2)
        {
            K << k1, k2;
            P << p1, p2;
        }
};

Vector3f distord(Camera * c, Vector3f p)
{
    Vector3f d;
    float r2 = p[0] * p[0] + p[1] * p[1];
    float r4 = r2 * r2;
    d = p * (1 + c->K[0] * r2 + c->K[1] * r4);
    d[0] += 2 * c->P[0] * p[0]*p[1] + c->P[1] * (r2 + 2 * p[0]*p[0]);
    d[1] += c->P[0] * (r2 + 2 * p[1]*p[1]) + 2 * c->P[1] * p[0]*p[1];
    d[2] = 1.0;
    return d;
}

Vector2f project_point(Camera * c, Vector3f p)
{
    Vector3f v = Vector3f (p[0] / p[2], p[1] / p[2], 1.0);
    Vector3f d = distord(c, v);
    Vector3f w = c->intrinsic_matrix * d;

    return Vector2f(w[0], w[1]);
}


std::vector<Vector3f> generate_cloud(int nb_points, float w, float d, float h)
{
    std::vector<Vector3f> cloud;

    for (int i = 0; i < nb_points; i++)
    {
        Vector3f p;
        p[0] = w * ((float)rand() / (float)RAND_MAX - 0.5);
        p[1] = h * ((float)rand() / (float)RAND_MAX - 0.5);
        p[2] = d * ((float)rand() / (float)RAND_MAX - 0.5);
        cloud.push_back(p);
    }
    return cloud;
}

std::vector<Vector3f> get_visible_points(Camera * c, std::vector<Vector3f> cloud)
{
    std::vector<Vector3f> visible_points;

    for (Vector3f p : cloud)
    {
        Vector4f v = (c->extrinsic_matrix * Vector4f(p[0], p[1], p[2], 1.0));
        Vector3f w = Vector3f(v[0], v[1], v[2]);

        if (v[2] <= 0.0000001)
        {
            Vector2f u = project_point(c, w);
            if (u[0] > 0 && u[1] > 0 && u[0] < c->width && u[1] < c->height)
                visible_points.push_back(p);
        }
    }
    return visible_points;
}

std::vector<Vector3f> camera_transform(Camera * c, std::vector<Vector3f> cloud)
{
    std::vector<Vector3f> transformed_points;

    for (Vector3f p : cloud)
    {
        Vector4f v = (c->extrinsic_matrix * Vector4f(p[0], p[1], p[2], 1.0));
        Vector3f w = Vector3f(v[0], v[1], v[2]);
        transformed_points.push_back(w);
    }
    return transformed_points;
}

std::vector<Vector2f> project_no_distortion(Camera * c, std::vector<Vector3f> visible_points)
{
    std::vector<Vector2f> projected;

    for (Vector3f p : visible_points)
    {
        Vector3f v = Vector3f (p[0] / p[2], p[1] / p[2], 1.0);
        Vector3f w = c->intrinsic_matrix * v;
        projected.push_back(Vector2f(w[0], w[1]));
    }
    return projected;
}

std::vector<Vector2f> project_and_distord(Camera * c, std::vector<Vector3f> visible_points)
{
    std::vector<Vector2f> projected;

    for (Vector3f p : visible_points)
    {
        Vector3f v = Vector3f (p[0] / p[2], p[1] / p[2], 1.0);
        Vector3f d = distord(c, v);
        Vector3f w = c->intrinsic_matrix * d;

        projected.push_back(Vector2f(w[0], w[1]));
    }
    return projected;
}

std::vector<Vector2f> noise_projection(std::vector<Vector2f> projected, float d)
{
    std::vector<Vector2f> noised;
    for (Vector2f p : projected)
    {
        float radius = 2. * d * ((float)rand() / (float)RAND_MAX - 0.5);
        float angle = 2. * M_PI * ((float)rand() / (float)RAND_MAX - 0.5); 
        Vector2f n = p + radius * Vector2f(cos(angle), sin(angle));
        noised.push_back(n);
    }
    return noised;
}

std::vector<int> find_neighbors(std::vector<Vector2f> &projected, Vector2f point, float r)
{
    std::vector<int> neighbors;

    for (int i = 0; i < projected.size(); i++)
    {
        Vector2f n = projected[i];
        if ((n - point).norm() <= r)
            neighbors.push_back(i);
    }
    return neighbors;
}

void make_false_association(std::vector<Vector2f> &projected, std::vector<Vector3f> &points3d, float probability, float radius)
{
    for (int i = 0; i < projected.size(); i++)
    {
        if (((float)rand() / (float)RAND_MAX) > probability)
        {
            Vector2f p = projected.at(i);
            std::vector<int> neighbors = find_neighbors(projected, p, radius);
            if (neighbors.size() == 0)
                continue;

            // Get a random idx from the neighbors
            int idx = (rand() * neighbors.size()) / RAND_MAX; 

            //Swap point at i and idx
            Vector2f v2 = projected[i];
            projected[i] = projected[idx];
            projected[idx] = v2;

            Vector3f v3 = points3d[i];
            points3d[i] = points3d[idx];
            points3d[idx] = v3;
        }
    }
}

void save_to(std::string name, std::vector<Vector2f> data)
{
    ofstream file;
    file.open(name);

    for (Vector2f p : data)
        file << p[0] << " " << p[1] << endl;
    file.close();
}

void save_to(std::string name, std::vector<Vector3f> data)
{
    ofstream file;
    file.open(name);

    for (Vector3f p : data)
        file << p[0] << " " << p[1] << " " << p[2] << endl;
    file.close();
}

int main ()
{
    float noise_radius = 7; //in px
    float false_association_radius = 5;
    float false_association_probability = 0.5;
    int nb_points = 200;

    srand(time(NULL));
    Calib calib(640, 360);
    Camera c;
    std::vector<Vector3f> camera_positions = {Vector3f(0, 0, -1), Vector3f(3, 0, 0), Vector3f(2, 1, 0), Vector3f(1, 0, 0) };

    std::vector<Vector3f> cloud;
    std::vector<Vector3f> visible_points;
    std::vector<Vector3f> transformed_points;
    std::vector<Vector2f> projected;
    std::vector<Vector2f> noised;

    cloud = generate_cloud(nb_points, 1, 1, 1);
    c.set_intrinsic(30, 640, 360);
    c.set_distortion(0.5, 0.4, 0., 0.);

    for (int i = 0; i < camera_positions.size(); i++)
    {
        c.look_at(camera_positions.at(i), Vector3f(0, 1, 0), Vector3f(0, 0, 0));
        visible_points = get_visible_points(&c, cloud);
        transformed_points = camera_transform(&c, visible_points);
        projected = project_and_distord(&c, transformed_points);
        noised = noise_projection(projected, 7);
        make_false_association(noised, visible_points, 0.5, 5);
        calib.feed(visible_points, noised);
    }

    /* Try to guess the results */

    cout << "== True camera parameters ==" << endl << endl;
    cout << "intrinsic matrix " << endl << c.intrinsic_matrix << endl;
    cout << "fovy : " << c.fov << endl;
    cout << "k1, k2 : " << c.K[0] << " " << c.K[1] << endl;
    cout << "p1, p2 : " << c.P[0] << " " << c.P[1] << endl << endl;


    cout << "nb points : " << nb_points << endl;
    cout << "nb views : " << camera_positions.size() << endl;
    cout << "noise radius : " << noise_radius << endl;
    cout << "false association radius : " << false_association_radius << endl;
    cout << "false assiciation probability : " << false_association_probability << endl << endl << endl;


    cout << "== Initial guessed camera paramters ==" << endl << endl;
    cout << "intrinsic matrix " << endl << calib.intrinsic_matrix << endl;
    cout << "fovy : " << calib.get_fovy() << endl;
    cout << "k1, k2 : " << calib.K[0] << " " << calib.K[1] << endl;
    cout << "p1, p2 : " << calib.P[0] << " " << calib.P[1] << endl << endl;


    float error = calib.try_calibrate();

    
    cout << "== Final guessed camera paramters ==" << endl << endl;
    cout << "intrinsic matrix " << endl << calib.intrinsic_matrix << endl;
    cout << "fovy : " << calib.get_fovy() << endl;
    cout << "k1, k2 : " << calib.K[0] << " " << calib.K[1] << endl;
    cout << "p1, p2 : " << calib.P[0] << " " << calib.P[1] << endl << endl;
    cout << "error = " << error << endl;

    //save_to("projected.txt", projected);

    return 0;
}

