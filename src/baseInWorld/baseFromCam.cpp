#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

using namespace Eigen;
using namespace std;

/* =========================
   Utility: invert transform
   ========================= */
Matrix4d invertTF(const Matrix4d& T) {
    Matrix4d inv = Matrix4d::Identity();
    inv.block<3,3>(0,0) = T.block<3,3>(0,0).transpose();
    inv.block<3,1>(0,3) =
        -inv.block<3,3>(0,0) * T.block<3,1>(0,3);
    return inv;
}

/* =========================
   DH + Forward Kinematics
   ========================= */
struct DH {
    double a, alpha, d, theta;
};

Matrix4d dhMatrix(const DH& p) {
    Matrix4d T = Matrix4d::Identity();
    double ct = cos(p.theta), st = sin(p.theta);
    double ca = cos(p.alpha), sa = sin(p.alpha);

    T << ct, -st*ca,  st*sa, p.a*ct,
         st,  ct*ca, -ct*sa, p.a*st,
          0,      sa,     ca,     p.d,
          0,       0,      0,      1;
    return T;
}

Matrix4d forwardKinematics(const vector<DH>& dh) {
    Matrix4d T = Matrix4d::Identity();
    for (const auto& j : dh)
        T *= dhMatrix(j);
    return T;
}

/* =========================
   Main
   ========================= */
int main() {

    /* -------------------------
       TRUE base pose in world
       (this is the "idea" truth)
       ------------------------- */
    Matrix4d T_W_B_true = Matrix4d::Identity();
    T_W_B_true.block<3,3>(0,0) =
        AngleAxisd(0.4, Vector3d::UnitZ()).toRotationMatrix();
    T_W_B_true.block<3,1>(0,3) << 1.0, 0.5, 0.0;

    /* -------------------------
       Joint angles (known)
       ------------------------- */
    vector<DH> dh = {
        {0.0,  M_PI/2, 0.4,  0.3},
        {0.3,  0.0,    0.0, -0.6},
        {0.2,  0.0,    0.0,  0.5},
        {0.0,  M_PI/2, 0.2,  1.0},
        {0.0, -M_PI/2, 0.0, -0.7},
        {0.0,  0.0,    0.1,  0.4}
    };

    Matrix4d T_B_E = forwardKinematics(dh);

    /* -------------------------
       Hand-eye calibration
       E → C
       ------------------------- */
    Matrix4d T_E_C = Matrix4d::Identity();
    T_E_C.block<3,3>(0,0) =
        AngleAxisd(0.2, Vector3d::UnitY()).toRotationMatrix();
    T_E_C.block<3,1>(0,3) << 0.05, 0.03, 0.08;

    /* -------------------------
       What camera observes
       (simulated measurement)
       ------------------------- */
    Matrix4d T_W_C =
        T_W_B_true * T_B_E * T_E_C;

    /* -------------------------
       Reconstruct base from camera
       ------------------------- */
    Matrix4d T_C_E = invertTF(T_E_C);
    Matrix4d T_E_B = invertTF(T_B_E);

    Matrix4d T_W_B_calc =
        T_W_C * T_C_E * T_E_B;

    /* -------------------------
       Results
       ------------------------- */
    cout << "\n=== TRUE World → Base ===\n";
    cout << T_W_B_true << "\n";

    cout << "\n=== CALCULATED World → Base ===\n";
    cout << T_W_B_calc << "\n";

    cout << "\n=== ERROR (calc - true) ===\n";
    cout << T_W_B_calc - T_W_B_true << "\n";

    return 0;
}
