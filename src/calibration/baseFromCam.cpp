#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace Eigen;

struct DHParameter {
    double a;
    double alpha;
    double d;
    double theta;
};

Matrix4d computeDHMatrix(const DHParameter& dh) {
    double c_theta = cos(dh.theta);
    double s_theta = sin(dh.theta);
    double c_alpha = cos(dh.alpha);
    double s_alpha = sin(dh.alpha);
    
    Matrix4d T = Matrix4d::Identity();
    T(0,0) = c_theta;
    T(0,1) = -s_theta * c_alpha;
    T(0,2) = s_theta * s_alpha;
    T(0,3) = dh.a * c_theta;
    
    T(1,0) = s_theta;
    T(1,1) = c_theta * c_alpha;
    T(1,2) = -c_theta * s_alpha;
    T(1,3) = dh.a * s_theta;
    
    T(2,1) = s_alpha;
    T(2,2) = c_alpha;
    T(2,3) = dh.d;
    
    return T;
}

Matrix4d getBaseFromCam(const cv::Mat& cameraFrame, 
                       const std::vector<DHParameter>& dhTable,
                       const Matrix4d& T_cam_marker) {
    Matrix4d T_base_eef = Matrix4d::Identity();
    
    for (const auto& dh : dhTable) {
        T_base_eef *= computeDHMatrix(dh);
    }
    
    Matrix4d T_base_world = T_base_eef * T_cam_marker;
    return T_base_world;
}

int main() {
    // Define DH parameters (6-DOF robot)
    // theta, d , a, alpha
    std::vector<DHParameter> dhTable = {
        {0.0, 0.0, 0.5, 0.0},      // Joint 1
        {0.3, 0.0, 0.0, 0.5},      // Joint 2
        {0.3, 0.0, 0.0, 0.2},      // Joint 3
        {0.0, 1.57, 0.2, 0.0},     // Joint 4
        {0.0, -1.57, 0.0, 0.0},    // Joint 5
        {0.0, 0.0, 0.1, 0.0}       // Joint 6
    };


    
    // Known transformation from camera to end-effector
    Matrix4d T_eef_cam = Matrix4d::Identity();
    T_eef_cam.block<3,3>(0,0) = AngleAxisd(0.1, Vector3d::UnitZ()).toRotationMatrix();
    T_eef_cam.block<3,1>(0,3) = Vector3d(0.05, 0.05, 0.1);
    
    // Dummy camera frame (can be replaced with actual frame data)
    cv::Mat cameraFrame(480, 640, CV_8UC3);
    
    // Compute base to world transformation
    Matrix4d T_base_world = getBaseFromCam(cameraFrame, dhTable, T_eef_cam);
    
    std::cout << "Base to World Transformation:\n" << T_base_world << std::endl;
    
    return 0;
}