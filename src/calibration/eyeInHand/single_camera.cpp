#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

// ============================================================================
// CAMERA INTRINSICS
// ============================================================================
struct CameraIntrinsics {
    Mat camera_matrix;  // 3x3 intrinsic matrix
    Mat dist_coeffs;    // Distortion coefficients
};

// ============================================================================
// Convert OpenCV pose (rvec + tvec) into Eigen 4x4 transformation matrix
// ============================================================================
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

// ============================================================================
// Detect a single ArUco marker and compute T_target2cam
// Also draw the marker outline + axis on the image
// ============================================================================
bool detectSingleMarkerPose(const Mat& image,
                            const CameraIntrinsics& cam_params,
                            float marker_size,
                            Eigen::Matrix4d& T_target2cam,
                            int dict_id = 10) {

    // Clone image for visualization
    Mat vis_img = image.clone();

    // Setup ArUco detector
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(dict_id);
    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();

    // Detect markers
    vector<int> marker_ids;
    vector<vector<Point2f>> marker_corners;
    aruco::detectMarkers(vis_img, dictionary, marker_corners, marker_ids, params);

    if (marker_ids.empty()) {
        cerr << "No ArUco markers detected" << endl;
        return false;
    }

    // Estimate pose
    vector<Vec3d> rvecs, tvecs;
    aruco::estimatePoseSingleMarkers(marker_corners, marker_size,
                                     cam_params.camera_matrix,
                                     cam_params.dist_coeffs,
                                     rvecs, tvecs);

    // Convert to Eigen transform
    Mat rvec = (Mat_<double>(3,1) << rvecs[0][0], rvecs[0][1], rvecs[0][2]);
    Mat tvec = (Mat_<double>(3,1) << tvecs[0][0], tvecs[0][1], tvecs[0][2]);
    T_target2cam = cvPoseToEigen(rvec, tvec);

    cout << "Marker ID " << marker_ids[0] << " detected at "
         << tvecs[0][2] * 1000 << " mm" << endl;

    // Draw detected marker and coordinate axes
    aruco::drawDetectedMarkers(vis_img, marker_corners, marker_ids);
    aruco::drawAxis(vis_img, cam_params.camera_matrix, cam_params.dist_coeffs,
                    rvecs[0], tvecs[0], marker_size * 0.5);

    // Show visualization
    imshow("Detected Pose", vis_img);
    waitKey(0);

    return true;
}

// ============================================================================
// Detect checkerboard and compute T_target2cam
// Also draw corners on the image
// ============================================================================
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
    drawChessboardCorners(vis_img, board_size, corners, found);

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
    drawFrameAxes(vis_img, cam_params.camera_matrix, cam_params.dist_coeffs,
                  rvec, tvec, square_size * 3);

    // Show visualization window
    imshow("Detected Pose", vis_img);
    waitKey(0);

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

    if (target_type == "aruco") {
        return detectSingleMarkerPose(image, cam_params, marker_size, T_target2cam);
    }
    else if (target_type == "checkerboard") {
        Size board_size(11, 8);      // INNER corners (not squares)
        float square_size = marker_size;
        return detectCheckerboardPose(image, cam_params, board_size, square_size, T_target2cam);
    }

    cerr << "Unknown target type: " << target_type << endl;
    return false;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    cout << "Camera Pose Detection Tool\n" << endl;

    // Camera calibration parameters (replace with real calibration)
    CameraIntrinsics cam_params;
    cam_params.camera_matrix = (Mat_<double>(3,3) <<
        800, 0, 320,
        0, 800, 240,
        0, 0, 1);

    cam_params.dist_coeffs = (Mat_<double>(5,1) <<
        0, 0, 0, 0, 0);

    Eigen::Matrix4d T_target2cam;

    string image_path =
        "/home/allen/capstone/calibration/src/calibration/examples/calib1211/Images/img_raw00.png";

    bool ok = getTargetPoseFromImage(
        image_path,
        cam_params,
        T_target2cam,
        "checkerboard",
        0.0275); //maker_size in meters

    if (ok) {
        cout << "\nTranslation (mm): "
             << T_target2cam(0,3) * 1000 << ", "
             << T_target2cam(1,3) * 1000 << ", "
             << T_target2cam(2,3) * 1000 << "\n";
    }

    return 0;
}
