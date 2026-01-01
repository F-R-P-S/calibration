#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

// ============================================================================
// CORE HAND-EYE CALIBRATION ALGORITHM (AX = XB)
// ============================================================================
// Solves: gripper2base * X = X * target2cam
// Where X is the unknown camera-to-gripper (end-effector) transform
// ============================================================================

Eigen::Matrix4d handEyeCalibration(
    const vector<Eigen::Matrix4d>& T_gripper2base,  // Robot end-effector poses in base frame
    const vector<Eigen::Matrix4d>& T_target2cam)    // Calibration target poses in camera frame
{
    if (T_gripper2base.size() != T_target2cam.size()) {
        throw runtime_error("Mismatched number of poses");
    }
    
    if (T_gripper2base.size() < 3) {
        throw runtime_error("Need at least 3 pose pairs");
    }
    
    size_t n = T_gripper2base.size();
    
    // Convert Eigen to OpenCV format
    vector<Mat> R_gripper2base, t_gripper2base;
    vector<Mat> R_target2cam, t_target2cam;
    
    for (size_t i = 0; i < n; i++) {
        // Extract rotation and translation from gripper2base
        Mat R_g2b(3, 3, CV_64F);
        Mat t_g2b(3, 1, CV_64F);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                R_g2b.at<double>(r, c) = T_gripper2base[i](r, c);
            }
            t_g2b.at<double>(r, 0) = T_gripper2base[i](r, 3);
        }
        
        // Extract rotation and translation from target2cam
        Mat R_t2c(3, 3, CV_64F);
        Mat t_t2c(3, 1, CV_64F);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                R_t2c.at<double>(r, c) = T_target2cam[i](r, c);
            }
            t_t2c.at<double>(r, 0) = T_target2cam[i](r, 3);
        }
        
        // Convert rotation matrices to rotation vectors
        Mat rvec_g2b, rvec_t2c;
        Rodrigues(R_g2b, rvec_g2b);
        Rodrigues(R_t2c, rvec_t2c);
        
        R_gripper2base.push_back(rvec_g2b);
        t_gripper2base.push_back(t_g2b);
        R_target2cam.push_back(rvec_t2c);
        t_target2cam.push_back(t_t2c);
    }
    
    // Solve AX = XB using OpenCV's calibrateHandEye
    Mat R_cam2gripper, t_cam2gripper;
    calibrateHandEye(R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    R_cam2gripper, t_cam2gripper,
                    CALIB_HAND_EYE_TSAI);
    
    // Convert result back to Eigen 4x4 matrix
    Mat R_result;
    Rodrigues(R_cam2gripper, R_result);
    
    Eigen::Matrix4d T_cam2gripper = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            T_cam2gripper(r, c) = R_result.at<double>(r, c);
        }
        T_cam2gripper(r, 3) = t_cam2gripper.at<double>(r, 0);
    }
    
    return T_cam2gripper;
}

// ============================================================================
// EXAMPLE USAGE
// ===================================================

struct CameraIntrinsics {
    Mat camera_matrix;  // 3x3 intrinsic matrix
    Mat dist_coeffs;    // Distortion coefficients
};

#define dataset_size 35

Eigen::Matrix4d cvPoseToEigen(const Mat& rvec, const Mat& tvec) {
    Mat R;
    Rodrigues(rvec, R);  // Convert rotation vector to matrix

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            T(i, j) = R.at<double>(i, j);
        }
        T(i, 3) = tvec.at<double>(i);
    }
    return T;
}


bool detectCheckerboardPose(const Mat& image,
                            const CameraIntrinsics& cam_params,
                            Size board_size,
                            float square_size,
                            Eigen::Matrix4d& T_target2cam) {

    Mat vis_img = image.clone(); // for visualization
    vector<Point2f> corners;

    // Detect inner corners
    bool found = findChessboardCorners(vis_img, board_size, corners,
                                       CALIB_CB_ADAPTIVE_THRESH |
                                       CALIB_CB_NORMALIZE_IMAGE);

    if (!found) {
        cerr << "Checkerboard not detected" << endl;
        return false;
    }

    // Refine corner accuracy
    Mat gray;
    cvtColor(vis_img, gray, COLOR_BGR2GRAY);
    cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

    // Draw checkerboard for visualization
    // drawChessboardCorners(vis_img, board_size, corners, found);

    // Prepare 3D points for solvePnP
    vector<Point3f> object_points;
    for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
            object_points.push_back(Point3f(j * square_size, i * square_size, 0));
        }
    }

    // Solve PnP
    Mat rvec, tvec;
    solvePnP(object_points, corners,
             cam_params.camera_matrix, cam_params.dist_coeffs,
             rvec, tvec);

    T_target2cam = cvPoseToEigen(rvec, tvec);

    cout << "Checkerboard detected at "
         << tvec.at<double>(2) * 1000 << " mm" << endl;

    // Draw coordinate axis
    // drawFrameAxes(vis_img, cam_params.camera_matrix, cam_params.dist_coeffs,
    //               rvec, tvec, square_size * 3);

    // Show visualization window
    // imshow("Detected Pose", vis_img);
    // waitKey(0);

    return true;
}

// ============================================================================
// Load image and detect the target type requested
// ============================================================================
bool getTargetPoseFromImage(const string& image_path,
                            const CameraIntrinsics& cam_params,
                            Eigen::Matrix4d& T_target2cam,
                            const string& target_type,
                            float marker_size) {

    Mat image = imread(image_path);
    if (image.empty()) {
        cerr << "Failed to load image: " << image_path << endl;
        return false;
    }

    cout << "Processing image: " << image_path << endl;

    // if (target_type == "aruco") {
    //     return detectSingleMarkerPose(image, cam_params, marker_size, T_target2cam);
    // }
    // else if (target_type == "checkerboard") {
        Size board_size(11, 8);      // INNER corners (not squares)
        float square_size = marker_size;
        return detectCheckerboardPose(image, cam_params, board_size, square_size, T_target2cam);
    // }

    cerr << "Unknown target type: " << target_type << endl;
    return false;
}



int main() {
    cout << "Hand-Eye Calibration - Core Algorithm\n" << endl;
    
    // Example: You would collect these from your robot and camera
    vector<Eigen::Matrix4d> T_gripper2base;  // End-effector poses (from robot FK)
    vector<Eigen::Matrix4d> T_target2cam;    // Target poses (from camera detection)
    
    // Example data (in practice, you get these from measurements)
    // Pose 1
    string poseIndex;
    for (int i = 0; i < dataset_size; i++) {
        //read from txt file for T_Gripper2base        
        if (i < 10){
            poseIndex = "0" + to_string(i);            
        }else poseIndex = to_string(i);

        string file_path = "/home/allen/capstone/calibration/src/calibration/eyeInHand/examples/calib1211/poses/poses" + poseIndex + ".txt";

        ifstream infile(file_path);
        Eigen::Matrix4d gripper_pose = Eigen::Matrix4d::Identity();
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                infile >> gripper_pose(r, c);
            }
        }

        cout<< "Loaded pose is " << gripper_pose << endl;

        T_gripper2base.push_back(gripper_pose);
        infile.close(); 
    }

    CameraIntrinsics cam_params;
    cam_params.camera_matrix = (Mat_<double>(3,3) <<
        800, 0, 320,
        0, 800, 240,
        0, 0, 1);

    cam_params.dist_coeffs = (Mat_<double>(5,1) <<
        0, 0, 0, 0, 0);

    // Data for T_target2cam from taken images
    for (int i = 0; i < dataset_size; i++) {

        //read from txt file for T_Target2cam        
        if (i < 10){
            poseIndex = "0" + to_string(i);            
        }else poseIndex = to_string(i);

        string file_path = "/home/allen/capstone/calibration/src/calibration/eyeInHand/eyeInHand/examples/calib1211/Images/img_raw" + poseIndex + ".png";
        
        Eigen::Matrix4d target_pose = Eigen::Matrix4d::Identity();

        getTargetPoseFromImage(
            file_path,
            cam_params,
            target_pose,
            "checkerboard",
            0.0275); //marker_size in meters
        
        T_target2cam.push_back(target_pose);
    }
    
    // Run calibration
    try {
        Eigen::Matrix4d T_cam2gripper = handEyeCalibration(T_gripper2base, T_target2cam);
        
        cout << "Calibrated Camera-to-EndEffector Transform:" << endl;
        cout << T_cam2gripper << endl;
        
        cout << "\nTranslation (mm): [" 
             << T_cam2gripper(0, 3) * 1000 << ", "
             << T_cam2gripper(1, 3) * 1000 << ", "
             << T_cam2gripper(2, 3) * 1000 << "]" << endl;
        
    } catch (const exception& e) {
        cerr << "Calibration failed: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}