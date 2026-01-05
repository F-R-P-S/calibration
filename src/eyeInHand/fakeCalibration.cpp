#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;

// ============================================================
// 1. UR10 FORWARD KINEMATICS (Standard DH)
// ============================================================
Matrix4d getUR10FK(const VectorXd& q) {
    // UR10 DH Parameters
    double d[]     = {0.1273, 0.0, 0.0, 0.1639, 0.1157, 0.0922};
    double a[]     = {0.0, -0.612, -0.5723, 0.0, 0.0, 0.0};
    double alpha[] = {M_PI/2.0, 0.0, 0.0, M_PI/2.0, -M_PI/2.0, 0.0};

    Matrix4d T = Matrix4d::Identity();
    for (int i = 0; i < 6; i++) {
        double ct = cos(q(i));
        double st = sin(q(i));
        double ca = cos(alpha[i]);
        double sa = sin(alpha[i]);

        Matrix4d Ti;
        Ti << ct, -st*ca,  st*sa, a[i]*ct,
              st,  ct*ca, -ct*sa, a[i]*st,
              0,       sa,     ca,     d[i],
              0,        0,      0,        1;
        T *= Ti;
    }
    return T;
}

// ============================================================
// 2. HAND–EYE CALIBRATION (Tsai–Lenz, Eigen)
// ============================================================
Matrix4d handEyeCalibration(const vector<Matrix4d>& T_base_ee,
                            const vector<Matrix4d>& T_cam_target)
{
    size_t n = T_base_ee.size() - 1;

    // ---------- 1. Solve for Rotation (R_ee_cam) ----------
    MatrixXd A_rot(3*n, 3);
    VectorXd b_rot(3*n);

    for (size_t i = 0; i < n; i++) {
        // A: Relative motion of gripper (base frame)
        Matrix4d A = T_base_ee[i].inverse() * T_base_ee[i+1];
        // B: Relative motion of camera (target frame)
        // CRITICAL FIX: B must be T_ci_t * T_ci+1_t.inverse()
        Matrix4d B = T_cam_target[i] * T_cam_target[i+1].inverse();

        AngleAxisd aaA(A.block<3,3>(0,0));
        AngleAxisd aaB(B.block<3,3>(0,0));

        // Modified Rodrigues Parameter vectors
        Vector3d Pg = 2.0 * sin(aaA.angle()/2.0) * aaA.axis();
        Vector3d Pc = 2.0 * sin(aaB.angle()/2.0) * aaB.axis();

        Vector3d s = Pg + Pc;
        Matrix3d skew;
        skew <<  0,   -s(2),  s(1),
                s(2),    0,  -s(0),
               -s(1),  s(0),    0;

        A_rot.block<3,3>(3*i,0) = skew;
        b_rot.segment<3>(3*i)   = Pg - Pc;
    }

    // Solve least squares for rotation vector P'
    Vector3d Pp = A_rot.colPivHouseholderQr().solve(b_rot);
    Vector3d P  = 2.0 * Pp / sqrt(1.0 + Pp.squaredNorm());

    // Rodrigues rotation matrix construction
    Matrix3d K;
    K <<  0,  -P(2),  P(1),
         P(2),    0, -P(0),
        -P(1),  P(0),    0;

    Matrix3d R = Matrix3d::Identity() + 
                 sqrt(1.0 - 0.25 * P.squaredNorm()) * K + 
                 0.5 * K * K;

    // SVD Orthogonalization (Ensures R is a valid SO(3) matrix)
    JacobiSVD<Matrix3d> svd(R, ComputeFullU | ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();

    // ---------- 2. Solve for Translation (t_ee_cam) ----------
    MatrixXd A_t(3*n, 3);
    VectorXd b_t(3*n);

    for (size_t i = 0; i < n; i++) {
        Matrix4d A = T_base_ee[i].inverse() * T_base_ee[i+1];
        Matrix4d B = T_cam_target[i] * T_cam_target[i+1].inverse();

        A_t.block<3,3>(3*i,0) = A.block<3,3>(0,0) - Matrix3d::Identity();
        b_t.segment<3>(3*i)   = R * B.topRightCorner<3,1>() - A.topRightCorner<3,1>();
    }

    Vector3d t = A_t.colPivHouseholderQr().solve(b_t);

    Matrix4d X = Matrix4d::Identity();
    X.block<3,3>(0,0) = R;
    X.topRightCorner<3,1>() = t;

    return X;
}

// ============================================================
// 3. MAIN
// ============================================================
// unit: meters
int main() {
    // 1. Setup Ground Truth
    Matrix4d T_ee_cam_GT = Matrix4d::Identity();
    T_ee_cam_GT.topRightCorner<3,1>() = Vector3d(0.15, -0.32, 0.8);

    Matrix4d T_world_target = Matrix4d::Identity();
    T_world_target.topRightCorner<3,1>() = Vector3d(0.8, 0.5, 0.3);

    vector<Matrix4d> T_base_ee, T_cam_target;

    // 2. Generate Synthetic Samples
    for (int i = 0; i < 30; i++) {
        // Joints must be diverse (rotation is key!)
        VectorXd q = VectorXd::Random(6) * M_PI;

        Matrix4d T_be = getUR10FK(q);
        // T_ct (Camera to Target) = T_c_ee * T_ee_b * T_b_target
        Matrix4d T_ct = (T_be * T_ee_cam_GT).inverse() * T_world_target;

        T_base_ee.push_back(T_be);
        T_cam_target.push_back(T_ct);
    }

    // 3. Run Calibration
    Matrix4d T_ee_cam_est = handEyeCalibration(T_base_ee, T_cam_target);

    // 4. Verification
    cout << "--- Hand-Eye Calibration Result ---" << endl;
    cout << "Estimated Matrix:\n" << T_ee_cam_est << endl;

    cout << "\nComparison (meters):" << endl;
    cout << "GT Trans:  " << T_ee_cam_GT.topRightCorner<3,1>().transpose() << endl;
    cout << "Est Trans: " << T_ee_cam_est.topRightCorner<3,1>().transpose() << endl;

    double err = (T_ee_cam_GT.topRightCorner<3,1>() - T_ee_cam_est.topRightCorner<3,1>()).norm();
    cout << "\nTranslation Error: " << err * 1000.0 << " mm" << endl;

    return 0;
}