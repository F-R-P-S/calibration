#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <iomanip>
#include <chrono>


void record_high_frequency_trajectory(
    const moveit::planning_interface::MoveGroupInterface::Plan& plan,
    const std::string& filename,
    double target_delta_ms,
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_pub,
    rclcpp::Logger logger)
{
    // Generate timestamped filename
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    std::string folder = "output_logs/";
    
    std::filesystem::create_directories(folder);
    
    ss << folder << filename << "_"
       << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S") << ".txt"; 
    
    std::ofstream log_file(ss.str(), std::ios::out);
    if (!log_file.is_open()) {
        RCLCPP_ERROR(logger, "Failed to open %s for writing!", ss.str().c_str());
        return;
    }

    const auto& trajectory = plan.trajectory_.joint_trajectory;
    size_t num_joints = trajectory.joint_names.size();
    
    // Write header with format description
    log_file << "# High-frequency interpolated trajectory for motor control\n";
    log_file << "# Format: time(s) delta_time(ms) joint1(rad) joint2(rad) ... joint6(rad)\n";
    log_file << "# Target update rate: " << (1000.0/target_delta_ms) << " Hz (" << target_delta_ms << " ms)\n";
    log_file << "# Joint names: ";
    for (const auto& name : trajectory.joint_names) {
        log_file << name << " ";
    }
    log_file << "\n#\n";

    if (trajectory.points.size() < 2) {
        RCLCPP_WARN(logger, "Trajectory has fewer than 2 points, cannot interpolate");
        log_file.close();
        return;
    }

    // Calculate trajectory parameters
    double total_duration = rclcpp::Duration(trajectory.points.back().time_from_start).seconds();
    double target_delta_s = target_delta_ms / 1000.0;
    
    size_t interpolated_count = 0;
    double current_time = 0.0;
    double previous_time = 0.0;
    
    // Interpolate trajectory at fixed time intervals
    while (current_time <= total_duration) {
        // Find which two waypoints we're currently between
        size_t idx = 0;
        for (size_t i = 0; i < trajectory.points.size() - 1; ++i) {
            double t1 = rclcpp::Duration(trajectory.points[i].time_from_start).seconds();
            double t2 = rclcpp::Duration(trajectory.points[i + 1].time_from_start).seconds();
            
            if (current_time >= t1 && current_time <= t2) {
                idx = i;
                break;
            }
        }
        
        // Handle edge case at end of trajectory
        if (current_time >= total_duration) {
            idx = trajectory.points.size() - 2;
        }
        
        // Get the two waypoints we're interpolating between
        const auto& p1 = trajectory.points[idx];
        const auto& p2 = trajectory.points[idx + 1];
        
        double t1 = rclcpp::Duration(p1.time_from_start).seconds();
        double t2 = rclcpp::Duration(p2.time_from_start).seconds();
        
        // Calculate interpolation factor (0.0 to 1.0)
        // alpha = 0.0 means we're at p1, alpha = 1.0 means we're at p2
        double alpha = (t2 - t1) > 0 ? (current_time - t1) / (t2 - t1) : 0.0;
        alpha = std::max(0.0, std::min(1.0, alpha));
        
        // Calculate time delta since last update
        double delta_time_ms = (current_time - previous_time) * 1000.0;
        
        // Write: timestamp, delta_time, joint positions
        // log_file << std::fixed << std::setprecision(6) << current_time << " ";
        // log_file << std::fixed << std::setprecision(2) << delta_time_ms << " ";
        
        // Linear interpolation for each joint: pos = p1 + alpha * (p2 - p1)
        for (size_t j = 0; j < num_joints; ++j) {
            double interpolated = p1.positions[j] + alpha * (p2.positions[j] - p1.positions[j]);
            log_file << std::fixed << std::setprecision(6) << interpolated;

            if (j < num_joints - 1) log_file << " ";
        }
        log_file << "\n";
        
        interpolated_count++;
        previous_time = current_time;
        current_time += target_delta_s;
    }

    log_file.close();
    
}


int main(int argc, char **argv)
{
    // initialize ROS2
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("test_moveit");
    auto joint_pub = node->create_publisher<std_msgs::msg::Float64MultiArray>("/joint_angles", 10);

    //change robot to search mode
    double j1 = node->declare_parameter<double>("joint_1", 0.0);
    double j2 = node->declare_parameter<double>("joint_2", 0.75);
    double j3 = node->declare_parameter<double>("joint_3", 2.2);
    double j4 = node->declare_parameter<double>("joint_4", 0.0);
    double j5 = node->declare_parameter<double>("joint_5", 0.0);
    double j6 = node->declare_parameter<double>("joint_6", 0.0);

    bool log = node->declare_parameter<bool>("log", true);

    rclcpp::executors::SingleThreadedExecutor executor; // create executor
    executor.add_node(node); // add node to executor
    auto spinner = std::thread([&executor]() { executor.spin(); });// create thread and make it spin

    // Initialize MoveGroupInterface for the "arm" planning group
    auto arm = moveit::planning_interface::MoveGroupInterface(node, "arm");
    arm.setMaxVelocityScalingFactor(1.0);
    arm.setMaxAccelerationScalingFactor(1.0);

    // set search mode
    //Joint Goal
    // std::vector<double> joints = {1.5, 0.5, -1.0, 0.0, 1.0, 0.0};
    std::vector<double> joints = {j1, j2, j3, j4, j5, j6};

    arm.setStartStateToCurrentState();
    arm.setJointValueTarget(joints);
    moveit::planning_interface::MoveGroupInterface::Plan plan1;
    // Plan to the joint target
    bool success1 = (arm.plan(plan1) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    //execute the plan if successful
    if (success1){ 
        arm.execute(plan1);
        // double target_hz = 50.0;
        // double target_delta_ms = 1000.0 / target_hz;
        // if (log) record_high_frequency_trajectory(plan1, "trajectory_high_freq", target_delta_ms, joint_pub, node->get_logger());
    }


    // ===============================
    // SEARCH MODE: Cartesian scanning
    // ===============================

    // define search path
    std::vector<geometry_msgs::msg::Pose> waypoints;
    geometry_msgs::msg::Pose pose1 = arm.getCurrentPose().pose;
    pose1.position.x += 0.2;
    waypoints.push_back(pose1);
    geometry_msgs::msg::Pose pose2 = pose1;
    pose2.position.y += 0.2;
    waypoints.push_back(pose2); 
    geometry_msgs::msg::Pose pose3 = pose2;
    pose3.position.y += -0.2;
    pose3.position.x += -0.2;
    waypoints.push_back(pose3);


    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = arm.computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
    RCLCPP_INFO(node->get_logger(), "searching path computed (%.2f%% achieved)", fraction * 100.0);

    // execute the search path if successful
    if (fraction > 0.95) {
        RCLCPP_INFO(node->get_logger(), "start searching...");

        moveit::planning_interface::MoveGroupInterface::Plan search_plan;
        search_plan.trajectory_ = trajectory;

        arm.execute(search_plan); 
    }



    // std::vector<double> tag = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // for now random decide if the location is determined or not


    // calculate the robot base in the world

    rclcpp::shutdown();
    spinner.join();
    return 0;
}