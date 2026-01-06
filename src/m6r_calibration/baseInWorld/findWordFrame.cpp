#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <Eigen/Dense>

#include <vector>
#include <thread>
#include <cmath>
#include <iostream>
#include <cstdlib>

// =====================================================
// Degree-based trig (MATCHES YOUR FK)
// =====================================================
inline double COSD(double x) { return cos(x * M_PI / 180.0); }
inline double SIND(double x) { return sin(x * M_PI / 180.0); }

// =====================================================
// DH definition
// =====================================================
struct DHParam {
  double a;      // mm
  double alpha;  // deg
  double d;      // mm
  double theta_offset;
};

// =====================================================
// Standard DH Transform (degrees, mm)
// =====================================================
Eigen::Matrix4d dhTransform(const DHParam& p, double theta_deg)
{
  double ct = COSD(theta_deg);
  double st = SIND(theta_deg);
  double ca = COSD(p.alpha);
  double sa = SIND(p.alpha);

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

  T(0,0) = ct;        T(0,1) = -st * ca;   T(0,2) =  st * sa;   T(0,3) = p.a * ct;
  T(1,0) = st;        T(1,1) =  ct * ca;   T(1,2) = -ct * sa;   T(1,3) = p.a * st;
  T(2,1) = sa;        T(2,2) =  ca;        T(2,3) = p.d;

  return T;
}

// =====================================================
// Forward Kinematics using YOUR DH
// =====================================================
Eigen::Matrix4d forwardKinematics(
    const std::vector<double>& joint_deg,
    const std::vector<DHParam>& dh)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for (size_t i = 0; i < dh.size(); ++i) {
    T *= dhTransform(dh[i], joint_deg[i] + dh[i].theta_offset);
  }
  return T;
}

// =====================================================
// Fake AprilTag detection
// =====================================================
bool fakeAprilTagDetected()
{
  return true;
}

// =====================================================
// Simple noise helper (mm)
// =====================================================
double noise_mm(double stddev = 1.0)
{
  return ((double)rand() / RAND_MAX - 0.5) * 2.0 * stddev;
}

// =====================================================
// MAIN
// =====================================================
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("test_moveit");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  auto arm = moveit::planning_interface::MoveGroupInterface(node, "arm");
  arm.setMaxVelocityScalingFactor(1.0);
  arm.setMaxAccelerationScalingFactor(1.0);

  // -----------------------------------------------------
  // Initial joint move (UNCHANGED)
  // -----------------------------------------------------
  std::vector<double> joints_init = {0.0, 0.75, 2.2, 0.0, 0.0, 0.0};
  arm.setStartStateToCurrentState();
  arm.setJointValueTarget(joints_init);

  moveit::planning_interface::MoveGroupInterface::Plan plan1;
  if (arm.plan(plan1) == moveit::planning_interface::MoveItErrorCode::SUCCESS) {
    arm.execute(plan1);
  }

  // -----------------------------------------------------
  // SEARCH MODE (UNCHANGED)
  // -----------------------------------------------------
  std::vector<geometry_msgs::msg::Pose> waypoints;
  geometry_msgs::msg::Pose pose1 = arm.getCurrentPose().pose;
  pose1.position.x += 0.2;
  waypoints.push_back(pose1);

  geometry_msgs::msg::Pose pose2 = pose1;
  pose2.position.y += 0.2;
  waypoints.push_back(pose2);

  geometry_msgs::msg::Pose pose3 = pose2;
  pose3.position.y -= 0.2;
  pose3.position.x -= 0.2;
  waypoints.push_back(pose3);

  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm.computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

  if (fraction > 0.95) {
    moveit::planning_interface::MoveGroupInterface::Plan search_plan;
    search_plan.trajectory_ = trajectory;
    arm.execute(search_plan);
  }

  // =====================================================
  // TRUE WORLD (SIMULATION-ONLY, HIDDEN FROM ALGORITHM)
  // =====================================================
  Eigen::Matrix4d T_world_true_base = Eigen::Matrix4d::Identity();
  T_world_true_base(0,3) = 400.0;
  T_world_true_base(1,3) = -200.0;

  // unit: mm
  Eigen::Matrix4d T_world_tag = Eigen::Matrix4d::Identity();
  T_world_tag(0,3) = 1200.0;
  T_world_tag(1,3) = 500.0;
  T_world_tag(2,3) = 300.0;

  Eigen::Matrix4d T_ee_cam = Eigen::Matrix4d::Identity();
  T_ee_cam(2,3) = 80.0;

  // =====================================================
  // TAG DETECTION + BASE CALIBRATION
  // =====================================================
  std::vector<Eigen::Matrix4d> world_base_estimates;

  for (int sample = 0; sample < 3; ++sample)
  {
    if (!fakeAprilTagDetected())
      continue;

    RCLCPP_INFO(node->get_logger(),
                "AprilTag detected, capturing joint state (sample %d)",
                sample);

    // Joint state from MoveIt
    std::vector<double> joint_rad = arm.getCurrentJointValues();

    std::vector<double> joint_deg(6);
    for (int i = 0; i < 6; ++i)
      joint_deg[i] = joint_rad[i] * 180.0 / M_PI;

    // DH table (YOUR VALUES)
    std::vector<DHParam> dh_table = {
        {  0.0,  90.0, 120.0, 0.0 },
        {450.0,   0.0,   0.0, 0.0 },
        {  0.0,   0.0,   0.0, 0.0 },
        {  0.0,  90.0, 450.0, 0.0 },
        {  0.0, -90.0,   0.0, 0.0 },
        {  0.0,   0.0,  80.0, 0.0 }
    };

    Eigen::Matrix4d T_base_ee = forwardKinematics(joint_deg, dh_table);

    // Simulated camera measurement
    // what the camera "sees" the tag as
    Eigen::Matrix4d T_cam_tag =
        T_ee_cam.inverse() *
        T_base_ee.inverse() *
        T_world_true_base.inverse() *
        T_world_tag;

    T_cam_tag(0,3) += noise_mm(1.0);
    T_cam_tag(1,3) += noise_mm(1.0);
    T_cam_tag(2,3) += noise_mm(1.0);

    // Calibration equation
    Eigen::Matrix4d T_world_base_est =
        T_world_tag *
        T_cam_tag.inverse() *
        T_ee_cam.inverse() *
        T_base_ee.inverse();

    world_base_estimates.push_back(T_world_base_est);
  }

  // =====================================================
  // AVERAGE + ERROR
  // =====================================================
  Eigen::Vector3d t_sum = Eigen::Vector3d::Zero();
  for (auto& T : world_base_estimates)
    t_sum += T.block<3,1>(0,3);

  Eigen::Vector3d t_est = t_sum / world_base_estimates.size();
  Eigen::Vector3d t_true = T_world_true_base.block<3,1>(0,3);

  std::cout << "\nTRUE T_world_base (mm): "
            << t_true.transpose() << std::endl;

  std::cout << "EST  T_world_base (mm): "
            << t_est.transpose() << std::endl;

  std::cout << "POSITION ERROR (mm): "
            << (t_est - t_true).norm() << std::endl;

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
