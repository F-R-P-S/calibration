#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <filesystem>

// ============================================================================
// OBSTACLE LOADER: Load obstacles from YAML configuration
// ============================================================================
namespace obstacle_loader {

struct ObstacleConfig {
    std::string package_name = "m6r_moveit_config";  // Change to your package name
    std::string config_file = "obstacles.yaml";
    std::string default_frame = "world";
    bool wait_for_scene_update = true;
    double scene_update_timeout = 2.0;
};

moveit_msgs::msg::CollisionObject createCollisionObject(
    const YAML::Node& obs_config,
    const rclcpp::Time& timestamp,
    const std::string& default_frame,
    const rclcpp::Logger& logger)
{
    moveit_msgs::msg::CollisionObject collision_object;
    
    collision_object.id = obs_config["id"].as<std::string>();
    collision_object.header.frame_id = obs_config["frame"].as<std::string>(default_frame);
    collision_object.header.stamp = timestamp;
    collision_object.operation = moveit_msgs::msg::CollisionObject::ADD;
    
    std::string type = obs_config["type"].as<std::string>("box");
    collision_object.primitives.resize(1);
    collision_object.primitive_poses.resize(1);
    
    if (type == "box") {
        collision_object.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
        auto dims = obs_config["dimensions"].as<std::vector<double>>();
        if (dims.size() != 3) {
            throw std::runtime_error("Box requires exactly 3 dimensions [x, y, z]");
        }
        collision_object.primitives[0].dimensions.clear();
        for (const auto& d : dims) {
            collision_object.primitives[0].dimensions.push_back(d);
        }
    } 
    else if (type == "sphere") {
        collision_object.primitives[0].type = shape_msgs::msg::SolidPrimitive::SPHERE;
        collision_object.primitives[0].dimensions.push_back(obs_config["radius"].as<double>());
    } 
    else if (type == "cylinder") {
        collision_object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
        collision_object.primitives[0].dimensions.push_back(obs_config["height"].as<double>());
        collision_object.primitives[0].dimensions.push_back(obs_config["radius"].as<double>());
    } 
    else {
        throw std::runtime_error("Unknown obstacle type: " + type);
    }
    
    collision_object.primitive_poses[0].position.x = obs_config["position"]["x"].as<double>();
    collision_object.primitive_poses[0].position.y = obs_config["position"]["y"].as<double>();
    collision_object.primitive_poses[0].position.z = obs_config["position"]["z"].as<double>();
    
    tf2::Quaternion q;
    q.setRPY(
        obs_config["orientation"]["roll"].as<double>(0.0),
        obs_config["orientation"]["pitch"].as<double>(0.0),
        obs_config["orientation"]["yaw"].as<double>(0.0)
    );
    collision_object.primitive_poses[0].orientation = tf2::toMsg(q);
    
    RCLCPP_INFO(logger, "  Created [%s] '%s' at (%.2f, %.2f, %.2f)",
        type.c_str(), collision_object.id.c_str(),
        collision_object.primitive_poses[0].position.x,
        collision_object.primitive_poses[0].position.y,
        collision_object.primitive_poses[0].position.z);
    
    return collision_object;
}

bool loadObstacles(
    moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
    const rclcpp::Node::SharedPtr& node,
    const ObstacleConfig& config = ObstacleConfig())
{
    auto logger = node->get_logger();
    
    try {
        std::string package_path = 
            ament_index_cpp::get_package_share_directory(config.package_name);
        std::string full_path = package_path + "/config/" + config.config_file;
        
        if (!std::filesystem::exists(full_path)) {
            RCLCPP_ERROR(logger, "Config file not found: %s", full_path.c_str());
            return false;
        }
        
        RCLCPP_INFO(logger, "Loading obstacles from: %s", full_path.c_str());
        YAML::Node yaml_config = YAML::LoadFile(full_path);
        
        if (!yaml_config["obstacles"]) {
            RCLCPP_WARN(logger, "No 'obstacles' key found in config file");
            return true;
        }
        
        std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
        auto current_time = node->now();
        
        for (const auto& obs : yaml_config["obstacles"]) {
            try {
                auto collision_obj = createCollisionObject(
                    obs, current_time, config.default_frame, logger);
                collision_objects.push_back(collision_obj);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(logger, "Failed to create obstacle: %s", e.what());
                continue;
            }
        }
        
        if (collision_objects.empty()) {
            RCLCPP_WARN(logger, "No valid obstacles found in config file");
            return true;
        }
        
        RCLCPP_INFO(logger, "Adding %zu obstacles to planning scene...", 
                    collision_objects.size());
        planning_scene_interface.addCollisionObjects(collision_objects);
        
        if (config.wait_for_scene_update) {
            rclcpp::sleep_for(
                std::chrono::milliseconds(
                    static_cast<int>(config.scene_update_timeout * 1000)));
        }
        
        RCLCPP_INFO(logger, "✓ Successfully added %zu obstacles to RViz2", 
                    collision_objects.size());
        return true;
        
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(logger, "YAML parsing error: %s", e.what());
        return false;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Failed to load obstacles: %s", e.what());
        return false;
    }
}

}  // namespace obstacle_loader

enum TargetState { PENDING = 0, SUCCESS = 1, FAILURE = -1 };

void visualize_target(
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub,
    const geometry_msgs::msg::Pose& pose,
    const std::string& label,
    int state,
    rclcpp::Node::SharedPtr node,
    rclcpp::Logger logger)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = node->now();
    marker.ns = "target_poses";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = pose;
    marker.scale.x = 0.15;
    marker.scale.y = 0.02;
    marker.scale.z = 0.02;
    
    if (state == TargetState::PENDING) {
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
    } else if (state == TargetState::SUCCESS) {
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
    } else { // FAILURE
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
    }

    marker.color.a = 0.8; // Semi-transparent
    marker.lifetime = rclcpp::Duration::from_seconds(0);
    marker_pub->publish(marker);
    
    visualization_msgs::msg::Marker text_marker;
    text_marker.header = marker.header;
    text_marker.ns = "target_labels";
    text_marker.id = 1;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.pose = pose;
    text_marker.pose.position.z += 0.1;
    text_marker.text = label;
    text_marker.scale.z = 0.05;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    marker_pub->publish(text_marker);
    
    RCLCPP_INFO(logger, "Target visualized: %s", label.c_str());
}

// ============================================================================
// HELPER FUNCTION: Log High-Frequency Trajectory for PWM Control
// ============================================================================
// 
// PURPOSE:
//   This function takes a MoveIt trajectory (which typically has sparse waypoints)
//   and interpolates it at a high frequency suitable for real-time control systems.
//   For example, MoveIt might generate 20 waypoints over 10 seconds, but your
//   motor controller needs position updates every 10ms (100 Hz).
//
// HOW IT WORKS:
//   1. Takes MoveIt's planned trajectory with N waypoints
//   2. Interpolates between waypoints at fixed time intervals (e.g., 10ms)
//   3. Uses linear interpolation to calculate joint positions at each interval
//   4. Outputs: timestamp, joint positions, and time delta between updates
//
// OUTPUT FORMAT:
//   time(s) delta_time(ms) joint1(rad) joint2(rad) ... joint6(rad)
//
// EXAMPLE OUTPUT:
//   0.000000 10.00 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
// ============================================================================
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
        log_file << std::fixed << std::setprecision(6) << current_time << " ";
        log_file << std::fixed << std::setprecision(2) << delta_time_ms << " ";
        
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
    
    // Log summary statistics
    RCLCPP_INFO(logger, "✓ High-frequency trajectory logged to: %s", ss.str().c_str());
    RCLCPP_INFO(logger, "  Original waypoints: %zu", trajectory.points.size());
    RCLCPP_INFO(logger, "  Interpolated waypoints: %zu", interpolated_count);
    RCLCPP_INFO(logger, "  Total duration: %.3f seconds", total_duration);
    RCLCPP_INFO(logger, "  Target update rate: %.1f Hz (%.1f ms)", 1000.0/target_delta_ms, target_delta_ms);
}


// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("m6r_improved_demo");
    auto logger = node->get_logger();
    
    // Declare ROS parameters for target position
    double px = node->declare_parameter<double>("target_x", 0.3);
    double py = node->declare_parameter<double>("target_y", 0.2);
    double pz = node->declare_parameter<double>("target_z", 0.25);
    
    // Start executor in background thread
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    auto spinner = std::thread([&executor]() { executor.spin(); });

    // Initialize MoveIt interfaces
    auto arm = moveit::planning_interface::MoveGroupInterface(node, "arm");
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // Publishers
    auto marker_pub = node->create_publisher<visualization_msgs::msg::Marker>("/target_marker", 10);
    auto joint_pub = node->create_publisher<std_msgs::msg::Float64MultiArray>("/joint_angles", 10);

    // ========================================
    // Configure Planning Parameters
    // ========================================
    arm.setPlannerId("RRTstar");
    arm.setGoalPositionTolerance(0.01);
    arm.setGoalOrientationTolerance(0.05);
    arm.setGoalJointTolerance(0.01);
    arm.setMaxVelocityScalingFactor(0.15);
    arm.setMaxAccelerationScalingFactor(0.15);
    arm.setPlanningTime(60.0);
    arm.setNumPlanningAttempts(100);
    arm.allowReplanning(true);
    arm.setReplanAttempts(10);
        
    RCLCPP_INFO(logger, "Planning Configuration:");
    RCLCPP_INFO(logger, "  Planner:        RRTstar");
    RCLCPP_INFO(logger, "  Position tol:   10mm");
    RCLCPP_INFO(logger, "  Orientation:    ~3°");
    RCLCPP_INFO(logger, "  Attempts:       100");
    RCLCPP_INFO(logger, "  Planning time:  60s");
    RCLCPP_INFO(logger, "  Velocity scale: 15%%");
    RCLCPP_INFO(logger, "Target Position:");
    RCLCPP_INFO(logger, "  X: %.3f  Y: %.3f  Z: %.3f\n", px, py, pz);

    // ========================================
    // Move to Start Position
    // ========================================
    RCLCPP_INFO(logger, "=== Step 1: Moving to Start Position ===");
    std::vector<double> start_joints = {3.14, 1.0, 0.5, 0.0, 0.5, 0.0};
    arm.setJointValueTarget(start_joints);
    
    moveit::planning_interface::MoveGroupInterface::Plan start_plan;
    if (arm.plan(start_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(logger, "✓ Start position plan successful");
        arm.execute(start_plan);
        RCLCPP_INFO(logger, "✓ Moved to start position");
    } else {
        RCLCPP_ERROR(logger, "✗ Failed to plan to start position");
    }
    rclcpp::sleep_for(std::chrono::seconds(2));

    // ========================================
    // Load Obstacles from YAML
    // ========================================
    RCLCPP_INFO(logger, "\n=== Step 2: Loading Obstacles from YAML ===");
    
    obstacle_loader::ObstacleConfig obs_config;
    obs_config.package_name = "m6r_bringup";  // CHANGE THIS to your package name
    obs_config.config_file = "workcell1.yaml";
    obs_config.default_frame = "world";
    obs_config.wait_for_scene_update = true;
    obs_config.scene_update_timeout = 2.0;
    
    if (!obstacle_loader::loadObstacles(planning_scene_interface, node, obs_config)) {
        RCLCPP_ERROR(logger, "Failed to load obstacles from YAML");
        // Continue anyway for demonstration
    }

    // ========================================
    // Plan and Execute to Target
    // ========================================
    RCLCPP_INFO(logger, "\n=== Step 3: Planning to Target ===");
    
    // geometry_msgs::msg::Pose target;
    // geometry_msgs::msg::PoseStamped target;
    
    // Define target pose in WORLD frame
    geometry_msgs::msg::PoseStamped target;
    target.header.frame_id = "world";
    target.header.stamp = node->now();

    // Set target position from parameters
    target.pose.position.x = px;
    target.pose.position.y = py;
    target.pose.position.z = pz;

    // Set orientation (facing downwards)
    tf2::Quaternion q;
    q.setRPY(0.0, 1.57, 0.0);  // Roll=0, Pitch=90deg (facing down), Yaw=0
    target.pose.orientation = tf2::toMsg(q);

    // Set the pose target with the stamped pose
    arm.setPoseTarget(target);

    visualize_target(marker_pub, target.pose, "Target Pending", 0, node, logger);
    // Optional: Add small delay to ensure RViz updates
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // Plan to target
    arm.setStartStateToCurrentState();
    arm.setPoseTarget(target);
    
    RCLCPP_INFO(logger, "Planning path around obstacle...");
    auto plan_start = std::chrono::high_resolution_clock::now();
    
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (arm.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    
    auto plan_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(plan_end - plan_start);
    
    if (success) {
        RCLCPP_INFO(logger, "✓ Planning SUCCESS in %ld ms!", duration.count());
        RCLCPP_INFO(logger, "  Waypoints: %zu", plan.trajectory_.joint_trajectory.points.size());
        RCLCPP_INFO(logger, "  Duration: %.2f seconds", 
        rclcpp::Duration(plan.trajectory_.joint_trajectory.points.back().time_from_start).seconds());
        
        visualize_target(marker_pub, target.pose, "Target ✓", 1, node, logger);
        
        RCLCPP_INFO(logger, "\n=== Step 4: Logging Trajectory ===");
        double target_hz = 50.0;
        double target_delta_ms = 1000.0 / target_hz;
        record_high_frequency_trajectory(plan, "trajectory_high_freq", target_delta_ms, joint_pub, logger);
        
        RCLCPP_INFO(logger, "\n=== Step 5: Executing Trajectory ===");
        auto execute_result = arm.execute(plan);
        
        if (execute_result == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(logger, "✓ Execution complete!");
        } else {
            RCLCPP_ERROR(logger, "✗ Execution failed!");
        }
        
    } else {
        RCLCPP_ERROR(logger, "✗ Planning FAILED after %ld ms", duration.count());
        visualize_target(marker_pub, target.pose, "Target ✗ FAILED", -1, node, logger);
    }
    
    rclcpp::sleep_for(std::chrono::seconds(2));

    // ========================================
    // Clean Up and Return Home
    // ========================================
    RCLCPP_INFO(logger, "\n=== Step 6: Cleanup and Return Home ===");
    
    // Get all known collision objects and remove them
    auto known_objects = planning_scene_interface.getKnownObjectNames();
    if (!known_objects.empty()) {
        planning_scene_interface.removeCollisionObjects(known_objects);
        RCLCPP_INFO(logger, "✓ Removed %zu obstacles", known_objects.size());
    }
    rclcpp::sleep_for(std::chrono::seconds(1));

    std::vector<double> home_joints = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    arm.setJointValueTarget(home_joints);
    
    moveit::planning_interface::MoveGroupInterface::Plan home_plan;
    if (arm.plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(logger, "✓ Returning home...");
        arm.execute(home_plan);
    }

    rclcpp::shutdown();
    spinner.join();
    return 0;
}