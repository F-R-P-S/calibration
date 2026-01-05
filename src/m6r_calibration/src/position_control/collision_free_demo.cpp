#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>

// ============================================================================
// HELPER FUNCTION: Visualize Target in RViz
// ============================================================================
void visualize_target(
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub,
    const geometry_msgs::msg::Pose& pose,
    const std::string& label,
    bool is_reachable,
    rclcpp::Node::SharedPtr node,
    rclcpp::Logger logger)
{
    // Arrow marker showing target pose
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp = node->now();
    marker.ns = "target_poses";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = pose;
    marker.scale.x = 0.15;  // Length
    marker.scale.y = 0.02;  // Width
    marker.scale.z = 0.02;  // Height
    
    // Green = success, Red = failed
    marker.color.r = is_reachable ? 0.0 : 1.0;
    marker.color.g = is_reachable ? 1.0 : 0.0;
    marker.color.b = 0.0;
    marker.color.a = 0.8;
    marker.lifetime = rclcpp::Duration::from_seconds(0);
    marker_pub->publish(marker);
    
    // Text label
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
// HELPER FUNCTION: Log Trajectory to File with Time Deltas for PWM Control
// ============================================================================
void record_trajectory_with_timing(
    const moveit::planning_interface::MoveGroupInterface::Plan& plan,
    const std::string& filename,
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_pub,
    rclcpp::Logger logger)
{
    // Generate timestamped filename
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    std::string folder = "output_logs/";
    ss << folder << filename << "_"
       << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S") << ".txt"; 
    
    std::ofstream log_file(ss.str(), std::ios::out);
    if (!log_file.is_open()) {
        RCLCPP_ERROR(logger, "Failed to open %s for writing!", ss.str().c_str());
        return;
    }

    // Write header
    log_file << "# Format: time_delta(ms) joint1(rad) joint2(rad) joint3(rad) joint4(rad) joint5(rad) joint6(rad)\n";
    log_file << "# time_delta = milliseconds to wait before moving to this position\n";
    log_file << "# Total waypoints: " << plan.trajectory_.joint_trajectory.points.size() << "\n";
    log_file << "# Joint names: ";
    for (const auto& name : plan.trajectory_.joint_trajectory.joint_names) {
        log_file << name << " ";
    }
    log_file << "\n";
    
    // Calculate statistics
    if (plan.trajectory_.joint_trajectory.points.size() > 1) {
        double total_time = rclcpp::Duration(plan.trajectory_.joint_trajectory.points.back().time_from_start).seconds();
        double avg_rate = (plan.trajectory_.joint_trajectory.points.size() - 1) / total_time;
        double avg_delta = 1000.0 / avg_rate;  // milliseconds
        log_file << "# Total duration: " << std::fixed << std::setprecision(3) << total_time << " seconds\n";
        log_file << "# Average time delta: " << std::fixed << std::setprecision(2) << avg_delta << " ms\n";
    }
    log_file << "#\n";
    log_file << "# Usage for PWM control:\n";
    log_file << "# for each line:\n";
    log_file << "#   1. Wait for time_delta milliseconds\n";
    log_file << "#   2. Move joints to the specified positions\n";
    log_file << "#   3. Repeat for next line\n";
    log_file << "#\n";

    // Write trajectory data with time deltas
    double previous_time = 0.0;
    
    for (size_t idx = 0; idx < plan.trajectory_.joint_trajectory.points.size(); ++idx) {
        const auto &point = plan.trajectory_.joint_trajectory.points[idx];
        
        // Calculate time from previous waypoint (in milliseconds)
        double current_time = rclcpp::Duration(point.time_from_start).seconds();
        double time_delta_ms = (current_time - previous_time) * 1000.0;  // Convert to ms
        
        // Write time delta (ms)
        log_file << std::fixed << std::setprecision(3) << time_delta_ms;
        
        // Write joint positions (radians)
        for (size_t i = 0; i < point.positions.size(); ++i) {
            log_file << " " << std::fixed << std::setprecision(6) << point.positions[i];
        }
        log_file << "\n";
        
        // Update previous time
        previous_time = current_time;
        
        // Publish to topic for real-time monitoring
        std_msgs::msg::Float64MultiArray msg;
        msg.data = point.positions;
        joint_pub->publish(msg);
    }

    log_file.close();
    
    // Print summary
    RCLCPP_INFO(logger, "✓ Trajectory logged to: %s", ss.str().c_str());
    RCLCPP_INFO(logger, "  Total waypoints: %zu", plan.trajectory_.joint_trajectory.points.size());
    RCLCPP_INFO(logger, "  Total duration: %.3f seconds", 
                rclcpp::Duration(plan.trajectory_.joint_trajectory.points.back().time_from_start).seconds());
    
    // Calculate min/max time deltas
    if (plan.trajectory_.joint_trajectory.points.size() > 1) {
        double min_delta = std::numeric_limits<double>::max();
        double max_delta = 0.0;
        double prev_t = 0.0;
        
        for (const auto& point : plan.trajectory_.joint_trajectory.points) {
            double curr_t = rclcpp::Duration(point.time_from_start).seconds();
            double delta = (curr_t - prev_t) * 1000.0;
            if (delta > 0) {
                min_delta = std::min(min_delta, delta);
                max_delta = std::max(max_delta, delta);
            }
            prev_t = curr_t;
        }
        
        RCLCPP_INFO(logger, "  Time delta range: %.2f - %.2f ms", min_delta, max_delta);
        RCLCPP_INFO(logger, "  Suggested PWM update rate: %.1f Hz", 1000.0 / max_delta);
    }
}

// ============================================================================
// HELPER FUNCTION: Log High-Frequency Trajectory for PWM Control
// ============================================================================
void record_high_frequency_trajectory(
    const moveit::planning_interface::MoveGroupInterface::Plan& plan,
    const std::string& filename,
    double target_delta_ms,  // Target time between waypoints (e.g., 20ms for 50Hz)
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_pub,
    rclcpp::Logger logger)
{
    // Generate timestamped filename
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    std::string folder = "output_logs/";
    ss << folder << filename << "_"
       << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S") << ".txt"; 
    
    std::ofstream log_file(ss.str(), std::ios::out);
    if (!log_file.is_open()) {
        RCLCPP_ERROR(logger, "Failed to open %s for writing!", ss.str().c_str());
        return;
    }

    const auto& trajectory = plan.trajectory_.joint_trajectory;
    size_t num_joints = trajectory.joint_names.size();
    
    // Write header
    log_file << "# High-frequency interpolated trajectory\n";
    log_file << "# Format: joint1(rad) joint2(rad) ... joint6(rad)\n";
    log_file << "# Target delta time: " << target_delta_ms << " ms\n";
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

    // Calculate total duration
    double total_duration = rclcpp::Duration(trajectory.points.back().time_from_start).seconds();
    double target_delta_s = target_delta_ms / 1000.0;
    
    // Generate interpolated waypoints
    size_t interpolated_count = 0;
    double current_time = 0.0;
    
    while (current_time <= total_duration) {
        // Find the two waypoints to interpolate between
        size_t idx = 0;
        for (size_t i = 0; i < trajectory.points.size() - 1; ++i) {
            double t1 = rclcpp::Duration(trajectory.points[i].time_from_start).seconds();
            double t2 = rclcpp::Duration(trajectory.points[i + 1].time_from_start).seconds();
            
            if (current_time >= t1 && current_time <= t2) {
                idx = i;
                break;
            }
        }
        
        // Handle edge case for last point
        if (current_time >= total_duration) {
            idx = trajectory.points.size() - 2;
        }
        
        const auto& p1 = trajectory.points[idx];
        const auto& p2 = trajectory.points[idx + 1];
        
        double t1 = rclcpp::Duration(p1.time_from_start).seconds();
        double t2 = rclcpp::Duration(p2.time_from_start).seconds();
        
        // Linear interpolation ratio
        double alpha = (t2 - t1) > 0 ? (current_time - t1) / (t2 - t1) : 0.0;
        alpha = std::max(0.0, std::min(1.0, alpha));  // Clamp to [0, 1]
        
        // Interpolate joint positions
        for (size_t j = 0; j < num_joints; ++j) {
            double interpolated = p1.positions[j] + alpha * (p2.positions[j] - p1.positions[j]);
            log_file << std::fixed << std::setprecision(6) << interpolated;
            if (j < num_joints - 1) log_file << " ";
        }
        log_file << "\n";
        
        interpolated_count++;
        current_time += target_delta_s;
    }

    log_file.close();
    
    // Print summary
    RCLCPP_INFO(logger, "✓ High-frequency trajectory logged to: %s", ss.str().c_str());
    RCLCPP_INFO(logger, "  Original waypoints: %zu", trajectory.points.size());
    RCLCPP_INFO(logger, "  Interpolated waypoints: %zu", interpolated_count);
    RCLCPP_INFO(logger, "  Total duration: %.3f seconds", total_duration);
    RCLCPP_INFO(logger, "  Target update rate: %.1f Hz (%.1f ms)", 1000.0/target_delta_ms, target_delta_ms);
}


// create obstacle function
void add_l_shaped_obstacle(
    moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
    rclcpp::Node::SharedPtr node,
    rclcpp::Logger logger)
{
    std::vector<moveit_msgs::msg::CollisionObject> l_shape;
    l_shape.resize(3);

    // Horizontal part of L
    l_shape[0].id =  "l_horizontal";
    l_shape[0].header.frame_id = "base_link";
    l_shape[0].header.stamp = node->now();
    l_shape[0].primitives.resize(1);
    l_shape[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    l_shape[0].primitives[0].dimensions = {0.20, 0.05, 0.70};
    l_shape[0].primitive_poses.resize(1);
    l_shape[0].primitive_poses[0].position.x = 0.30;
    l_shape[0].primitive_poses[0].position.y = 0.40;
    l_shape[0].primitive_poses[0].position.z = 0.35;
    l_shape[0].primitive_poses[0].orientation.w = 1.0;
    l_shape[0].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Vertical part of L
    l_shape[1].id = "l_vertical";
    l_shape[1].header.frame_id = "base_link";
    l_shape[1].header.stamp = node->now();
    l_shape[1].primitives.resize(1);
    l_shape[1].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    l_shape[1].primitives[0].dimensions = {0.05, 0.30, 0.3};
    l_shape[1].primitive_poses.resize(1);
    l_shape[1].primitive_poses[0].position.x = 0.38;
    l_shape[1].primitive_poses[0].position.y = 0.025;
    l_shape[1].primitive_poses[0].position.z = 0.15;
    l_shape[1].primitive_poses[0].orientation.w = 1.0;
    l_shape[1].operation = moveit_msgs::msg::CollisionObject::ADD;

    l_shape[2].id = "l_vertical2";
    l_shape[2].header.frame_id = "base_link";
    l_shape[2].header.stamp = node->now();
    l_shape[2].primitives.resize(1);
    l_shape[2].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    l_shape[2].primitives[0].dimensions = {0.05, 0.30, 0.3};
    l_shape[2].primitive_poses.resize(1);
    l_shape[2].primitive_poses[0].position.x = 0.68;
    l_shape[2].primitive_poses[0].position.y = 0.025;
    l_shape[2].primitive_poses[0].position.z = 0.15;
    l_shape[2].primitive_poses[0].orientation.w = 1.0;
    l_shape[2].operation = moveit_msgs::msg::CollisionObject::ADD;


    planning_scene_interface.addCollisionObjects(l_shape);
    rclcpp::sleep_for(std::chrono::seconds(2));
    RCLCPP_INFO(logger, "✓ L-shaped obstacle added to scene");
}

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
    arm.setGoalPositionTolerance(0.01);           // 10mm
    arm.setGoalOrientationTolerance(0.05);        // ~3°
    arm.setGoalJointTolerance(0.01);
    arm.setMaxVelocityScalingFactor(0.15);        // 15% max velocity
    arm.setMaxAccelerationScalingFactor(0.15);    // 15% max acceleration
    arm.setPlanningTime(60.0);                    // max planning time
    arm.setNumPlanningAttempts(100);               // attempts limit
    arm.allowReplanning(true);
    arm.setReplanAttempts(10);
    
    RCLCPP_INFO(logger, "\n╔══════════════════════════════════════════╗");
    RCLCPP_INFO(logger, "║      M6R Obstacle Avoidance Demo        ║");
    RCLCPP_INFO(logger, "╚══════════════════════════════════════════╝");
    RCLCPP_INFO(logger, "Planning Configuration:");
    RCLCPP_INFO(logger, "  Planner:        RRTstar");
    RCLCPP_INFO(logger, "  Position tol:   10mm");
    RCLCPP_INFO(logger, "  Orientation:    ~3°");
    RCLCPP_INFO(logger, "  Attempts:       50");
    RCLCPP_INFO(logger, "  Planning time:  20s");
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
    // Add Obstacle
    // ========================================
    RCLCPP_INFO(logger, "\n=== Step 2: Adding L-Shaped Obstacle ===");
    add_l_shaped_obstacle(planning_scene_interface, node, logger);

    // ========================================
    // Plan and Execute to Target
    // ========================================
    RCLCPP_INFO(logger, "\n=== Step 3: Planning to Target ===");
    
    // Create target pose
    geometry_msgs::msg::Pose target;
    target.position.x = px;
    target.position.y = py;
    target.position.z = pz;
    
    tf2::Quaternion q;
    q.setRPY(0.0, 1.57, 0.0);  // Pointing down
    target.orientation.x = q.x();
    target.orientation.y = q.y();
    target.orientation.z = q.z();
    target.orientation.w = q.w();

    // Visualize target
    visualize_target(marker_pub, target, "Target Position", true, node, logger);
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // Plan trajectory
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
        
        // Update visualization
        visualize_target(marker_pub, target, "Target ✓", true, node, logger);
        
        // Log trajectory to file
        RCLCPP_INFO(logger, "\n=== Step 4: Logging Trajectory ===");
        // record_trajectory_with_timing(plan, "trajectory_log", joint_pub, logger);

        // High-frequency version for direct joint control
        double target_hz = 100.0;  // 50Hz update rate
        double target_delta_ms = 1000.0 / target_hz;
        record_high_frequency_trajectory(plan, "trajectory_high_freq", target_delta_ms, joint_pub, logger);
        
        // Execute trajectory
        RCLCPP_INFO(logger, "\n=== Step 5: Executing Trajectory ===");
        auto execute_result = arm.execute(plan);
        
        if (execute_result == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(logger, "✓ Execution complete!");
        } else {
            RCLCPP_ERROR(logger, "✗ Execution failed!");
        }
        
    } else {
        RCLCPP_ERROR(logger, "✗ Planning FAILED after %ld ms", duration.count());
        visualize_target(marker_pub, target, "Target ✗ FAILED", false, node, logger);
    }
    
    rclcpp::sleep_for(std::chrono::seconds(2));

    // ========================================
    // Clean Up and Return Home
    // ========================================
    RCLCPP_INFO(logger, "\n=== Step 6: Cleanup and Return Home ===");
    planning_scene_interface.removeCollisionObjects({"l_horizontal", "l_vertical", "l_vertical2"});
    RCLCPP_INFO(logger, "✓ Obstacles removed");
    rclcpp::sleep_for(std::chrono::seconds(1));

    std::vector<double> home_joints = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    arm.setJointValueTarget(home_joints);
    
    moveit::planning_interface::MoveGroupInterface::Plan home_plan;
    if (arm.plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(logger, "✓ Returning home...");
        arm.execute(home_plan);
    }

    RCLCPP_INFO(logger, "\n╔══════════════════════════════════════════╗");
    RCLCPP_INFO(logger, "║           Demo Complete!                 ║");
    RCLCPP_INFO(logger, "╚══════════════════════════════════════════╝\n");

    rclcpp::shutdown();
    spinner.join();
    return 0;
}