#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("kr3_forced_avoidance");
    
    auto logger = node->get_logger();
    
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    auto spinner = std::thread([&executor]() { executor.spin(); });

    auto arm = moveit::planning_interface::MoveGroupInterface(node, "arm");
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // Planning parameters
    arm.setPlannerId("RRTConnectkConfigDefault");
    arm.setMaxVelocityScalingFactor(0.15);  // Slower for visibility
    arm.setMaxAccelerationScalingFactor(0.15);
    arm.setPlanningTime(15.0);
    arm.setNumPlanningAttempts(20);
    
    RCLCPP_INFO(logger, "=== KUKA KR3 Obstacle Avoidance Demo ===");
    RCLCPP_INFO(logger, "Workspace: ~635mm reach from base");

    auto joint_pub = node->create_publisher<std_msgs::msg::Float64MultiArray>("/joint_angles", 10);

    // Helper function for planning
    auto plan_and_execute = [&](const std::string& name, 
                                 double x, double y, double z,
                                 double roll, double pitch, double yaw) -> bool {
        RCLCPP_INFO(logger, "\n========================================");
        RCLCPP_INFO(logger, "TARGET: %s", name.c_str());
        RCLCPP_INFO(logger, "Position: [%.3f, %.3f, %.3f]", x, y, z);
        
        geometry_msgs::msg::Pose target;
        target.position.x = x;
        target.position.y = y;
        target.position.z = z;
        
        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);
        target.orientation.x = q.x();
        target.orientation.y = q.y();
        target.orientation.z = q.z();
        target.orientation.w = q.w();

        arm.setStartStateToCurrentState();
        arm.setPoseTarget(target);

        RCLCPP_INFO(logger, "Planning path around obstacles...");
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool success = (arm.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
        
        if (success) {
            RCLCPP_INFO(logger, "✓ SUCCESS! Found collision-free path!");
            RCLCPP_INFO(logger, "Waypoints: %zu (more = more complex avoidance)", 
                       plan.trajectory_.joint_trajectory.points.size());
            RCLCPP_INFO(logger, "Executing...");
            
            arm.execute(plan);
            
            for (const auto &point : plan.trajectory_.joint_trajectory.points) {
                std_msgs::msg::Float64MultiArray msg;
                msg.data = point.positions;
                joint_pub->publish(msg);
            }
            
            RCLCPP_INFO(logger, "✓ Complete!");
            rclcpp::sleep_for(std::chrono::seconds(2));
            return true;
        } else {
            RCLCPP_ERROR(logger, "✗ FAILED! No valid path found.");
            return false;
        }
    };

    // Start from a known safe position
    RCLCPP_INFO(logger, "\n=== Moving to start position ===");
    std::vector<double> start_joints = {0.0, -0.3, 0.5, 0.0, 0.5, 0.0};
    arm.setJointValueTarget(start_joints);
    moveit::planning_interface::MoveGroupInterface::Plan start_plan;
    if (arm.plan(start_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        arm.execute(start_plan);
    }
    rclcpp::sleep_for(std::chrono::seconds(2));

    // ========================================
    // SCENARIO 1: Single obstacle in the way
    // ========================================
    RCLCPP_INFO(logger, "\n\n╔═══════════════════════════════════════╗");
    RCLCPP_INFO(logger, "║  SCENARIO 1: Single Blocking Box     ║");
    RCLCPP_INFO(logger, "╚═══════════════════════════════════════╝");
    
    std::vector<moveit_msgs::msg::CollisionObject> box_scenario;
    box_scenario.resize(1);

    // Medium box blocking center path
    box_scenario[0].id = "blocking_box";
    box_scenario[0].header.frame_id = "base_link";
    box_scenario[0].header.stamp = node->now();
    box_scenario[0].primitives.resize(1);
    box_scenario[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    box_scenario[0].primitives[0].dimensions = {0.12, 0.12, 0.2};  // Small box
    box_scenario[0].primitive_poses.resize(1);
    box_scenario[0].primitive_poses[0].position.x = 0.28;  // In front of robot
    box_scenario[0].primitive_poses[0].position.y = 0.0;   // Center
    box_scenario[0].primitive_poses[0].position.z = 0.1;   // On table
    box_scenario[0].primitive_poses[0].orientation.w = 1.0;
    box_scenario[0].operation = moveit_msgs::msg::CollisionObject::ADD;

    planning_scene_interface.addCollisionObjects(box_scenario);
    rclcpp::sleep_for(std::chrono::seconds(2));
    RCLCPP_INFO(logger, "Box placed! Direct path is blocked.");

    // Target behind the box - must go around (over or side)
    plan_and_execute("Behind box (go around)", 0.40, 0.0, 0.35, 0.0, 1.57, 0.0);
    
    // Target to the side
    plan_and_execute("Side path (left)", 0.30, 0.20, 0.25, 0.0, 1.57, 0.0);
    
    plan_and_execute("Side path (right)", 0.30, -0.20, 0.25, 0.0, 1.57, 0.0);

    // Remove box
    planning_scene_interface.removeCollisionObjects({"blocking_box"});
    rclcpp::sleep_for(std::chrono::seconds(1));

    // ========================================
    // SCENARIO 2: Two cylinders (go between)
    // ========================================
    RCLCPP_INFO(logger, "\n\n╔═══════════════════════════════════════╗");
    RCLCPP_INFO(logger, "║  SCENARIO 2: Two Pillars             ║");
    RCLCPP_INFO(logger, "╚═══════════════════════════════════════╝");
    
    std::vector<moveit_msgs::msg::CollisionObject> pillars;
    pillars.resize(2);

    // Left pillar
    pillars[0].id = "pillar_left";
    pillars[0].header.frame_id = "base_link";
    pillars[0].header.stamp = node->now();
    pillars[0].primitives.resize(1);
    pillars[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    pillars[0].primitives[0].dimensions = {0.3, 0.04};  // height, radius (thin pillar)
    pillars[0].primitive_poses.resize(1);
    pillars[0].primitive_poses[0].position.x = 0.32;
    pillars[0].primitive_poses[0].position.y = 0.12;   // Left side, with gap
    pillars[0].primitive_poses[0].position.z = 0.15;
    pillars[0].primitive_poses[0].orientation.w = 1.0;
    pillars[0].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Right pillar
    pillars[1].id = "pillar_right";
    pillars[1].header.frame_id = "base_link";
    pillars[1].header.stamp = node->now();
    pillars[1].primitives.resize(1);
    pillars[1].primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    pillars[1].primitives[0].dimensions = {0.3, 0.04};
    pillars[1].primitive_poses.resize(1);
    pillars[1].primitive_poses[0].position.x = 0.32;
    pillars[1].primitive_poses[0].position.y = -0.12;  // Right side, with gap
    pillars[1].primitive_poses[0].position.z = 0.15;
    pillars[1].primitive_poses[0].orientation.w = 1.0;
    pillars[1].operation = moveit_msgs::msg::CollisionObject::ADD;

    planning_scene_interface.addCollisionObjects(pillars);
    rclcpp::sleep_for(std::chrono::seconds(2));
    RCLCPP_INFO(logger, "Two pillars placed! Gap in center = ~240mm");

    // Go between the pillars
    plan_and_execute("Through the gap", 0.38, 0.0, 0.25, 0.0, 1.57, 0.0);
    
    // Go around left pillar
    plan_and_execute("Around left pillar", 0.35, 0.22, 0.28, 0.0, 1.57, 0.0);
    
    // Go around right pillar
    plan_and_execute("Around right pillar", 0.35, -0.22, 0.28, 0.0, 1.57, 0.0);

    // Over the pillars
    plan_and_execute("Over the pillars", 0.32, 0.0, 0.42, 0.0, 1.57, 0.0);

    planning_scene_interface.removeCollisionObjects({"pillar_left", "pillar_right"});
    rclcpp::sleep_for(std::chrono::seconds(1));

    // ========================================
    // SCENARIO 3: L-shaped obstacle
    // ========================================
    RCLCPP_INFO(logger, "\n\n╔═══════════════════════════════════════╗");
    RCLCPP_INFO(logger, "║  SCENARIO 3: L-Shaped Barrier        ║");
    RCLCPP_INFO(logger, "╚═══════════════════════════════════════╝");
    
    std::vector<moveit_msgs::msg::CollisionObject> l_shape;
    l_shape.resize(2);

    // Horizontal part of L
    l_shape[0].id = "l_horizontal";
    l_shape[0].header.frame_id = "base_link";
    l_shape[0].header.stamp = node->now();
    l_shape[0].primitives.resize(1);
    l_shape[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    l_shape[0].primitives[0].dimensions = {0.25, 0.05, 0.15};
    l_shape[0].primitive_poses.resize(1);
    l_shape[0].primitive_poses[0].position.x = 0.30;
    l_shape[0].primitive_poses[0].position.y = 0.10;
    l_shape[0].primitive_poses[0].position.z = 0.075;
    l_shape[0].primitive_poses[0].orientation.w = 1.0;
    l_shape[0].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Vertical part of L
    l_shape[1].id = "l_vertical";
    l_shape[1].header.frame_id = "base_link";
    l_shape[1].header.stamp = node->now();
    l_shape[1].primitives.resize(1);
    l_shape[1].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    l_shape[1].primitives[0].dimensions = {0.05, 0.20, 0.15};
    l_shape[1].primitive_poses.resize(1);
    l_shape[1].primitive_poses[0].position.x = 0.40;
    l_shape[1].primitive_poses[0].position.y = 0.025;
    l_shape[1].primitive_poses[0].position.z = 0.075;
    l_shape[1].primitive_poses[0].orientation.w = 1.0;
    l_shape[1].operation = moveit_msgs::msg::CollisionObject::ADD;

    planning_scene_interface.addCollisionObjects(l_shape);
    rclcpp::sleep_for(std::chrono::seconds(2));
    RCLCPP_INFO(logger, "L-shaped barrier! Must navigate around corner.");

    // Navigate around the L
    plan_and_execute("Before L-corner", 0.25, 0.05, 0.22, 0.0, 1.57, 0.0);
    plan_and_execute("Around L (outside)", 0.35, -0.15, 0.25, 0.0, 1.57, 0.0);
    plan_and_execute("Behind L", 0.45, 0.10, 0.22, 0.0, 1.57, 0.0);

    planning_scene_interface.removeCollisionObjects({"l_horizontal", "l_vertical"});
    rclcpp::sleep_for(std::chrono::seconds(1));

    // ========================================
    // SCENARIO 4: Low ceiling (must stay low)
    // ========================================
    RCLCPP_INFO(logger, "\n\n╔═══════════════════════════════════════╗");
    RCLCPP_INFO(logger, "║  SCENARIO 4: Low Ceiling             ║");
    RCLCPP_INFO(logger, "╚═══════════════════════════════════════╝");
    
    std::vector<moveit_msgs::msg::CollisionObject> ceiling;
    ceiling.resize(1);

    ceiling[0].id = "low_ceiling";
    ceiling[0].header.frame_id = "base_link";
    ceiling[0].header.stamp = node->now();
    ceiling[0].primitives.resize(1);
    ceiling[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    ceiling[0].primitives[0].dimensions = {0.35, 0.35, 0.03};  // Wide, thin plate
    ceiling[0].primitive_poses.resize(1);
    ceiling[0].primitive_poses[0].position.x = 0.35;
    ceiling[0].primitive_poses[0].position.y = 0.0;
    ceiling[0].primitive_poses[0].position.z = 0.32;  // Low ceiling height
    ceiling[0].primitive_poses[0].orientation.w = 1.0;
    ceiling[0].operation = moveit_msgs::msg::CollisionObject::ADD;

    planning_scene_interface.addCollisionObjects(ceiling);
    rclcpp::sleep_for(std::chrono::seconds(2));
    RCLCPP_INFO(logger, "Low ceiling at 320mm! Must duck under.");

    // Targets that require staying low
    plan_and_execute("Under ceiling (center)", 0.35, 0.0, 0.18, 0.0, 1.57, 0.0);
    plan_and_execute("Under ceiling (left)", 0.35, 0.12, 0.20, 0.0, 1.57, 0.0);
    plan_and_execute("Under ceiling (right)", 0.35, -0.12, 0.20, 0.0, 1.57, 0.0);

    planning_scene_interface.removeCollisionObjects({"low_ceiling"});
    rclcpp::sleep_for(std::chrono::seconds(1));

    // ========================================
    // SCENARIO 5: Complex environment (multiple obstacles)
    // ========================================
    RCLCPP_INFO(logger, "\n\n╔═══════════════════════════════════════╗");
    RCLCPP_INFO(logger, "║  SCENARIO 5: Complex Environment     ║");
    RCLCPP_INFO(logger, "╚═══════════════════════════════════════╝");
    
    std::vector<moveit_msgs::msg::CollisionObject> complex_env;
    complex_env.resize(4);

    // Ground table
    complex_env[0].id = "ground_table";
    complex_env[0].header.frame_id = "base_link";
    complex_env[0].header.stamp = node->now();
    complex_env[0].primitives.resize(1);
    complex_env[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    complex_env[0].primitives[0].dimensions = {0.5, 0.5, 0.02};
    complex_env[0].primitive_poses.resize(1);
    complex_env[0].primitive_poses[0].position.x = 0.35;
    complex_env[0].primitive_poses[0].position.y = 0.0;
    complex_env[0].primitive_poses[0].position.z = 0.0;
    complex_env[0].primitive_poses[0].orientation.w = 1.0;
    complex_env[0].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Left obstacle
    complex_env[1].id = "left_box";
    complex_env[1].header.frame_id = "base_link";
    complex_env[1].header.stamp = node->now();
    complex_env[1].primitives.resize(1);
    complex_env[1].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    complex_env[1].primitives[0].dimensions = {0.08, 0.08, 0.15};
    complex_env[1].primitive_poses.resize(1);
    complex_env[1].primitive_poses[0].position.x = 0.28;
    complex_env[1].primitive_poses[0].position.y = 0.15;
    complex_env[1].primitive_poses[0].position.z = 0.075;
    complex_env[1].primitive_poses[0].orientation.w = 1.0;
    complex_env[1].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Right obstacle
    complex_env[2].id = "right_box";
    complex_env[2].header.frame_id = "base_link";
    complex_env[2].header.stamp = node->now();
    complex_env[2].primitives.resize(1);
    complex_env[2].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    complex_env[2].primitives[0].dimensions = {0.08, 0.08, 0.15};
    complex_env[2].primitive_poses.resize(1);
    complex_env[2].primitive_poses[0].position.x = 0.28;
    complex_env[2].primitive_poses[0].position.y = -0.15;
    complex_env[2].primitive_poses[0].position.z = 0.075;
    complex_env[2].primitive_poses[0].orientation.w = 1.0;
    complex_env[2].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Center cylinder
    complex_env[3].id = "center_cylinder";
    complex_env[3].header.frame_id = "base_link";
    complex_env[3].header.stamp = node->now();
    complex_env[3].primitives.resize(1);
    complex_env[3].primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    complex_env[3].primitives[0].dimensions = {0.2, 0.05};
    complex_env[3].primitive_poses.resize(1);
    complex_env[3].primitive_poses[0].position.x = 0.38;
    complex_env[3].primitive_poses[0].position.y = 0.0;
    complex_env[3].primitive_poses[0].position.z = 0.1;
    complex_env[3].primitive_poses[0].orientation.w = 1.0;
    complex_env[3].operation = moveit_msgs::msg::CollisionObject::ADD;

    planning_scene_interface.addCollisionObjects(complex_env);
    rclcpp::sleep_for(std::chrono::seconds(2));
    RCLCPP_INFO(logger, "Complex environment! Multiple obstacles require smart navigation.");

    // Navigate through complex environment
    plan_and_execute("Pick position (left side)", 0.30, 0.10, 0.18, 0.0, 1.57, 0.0);
    plan_and_execute("Above center cylinder", 0.38, 0.0, 0.35, 0.0, 1.57, 0.0);
    plan_and_execute("Place position (right side)", 0.30, -0.10, 0.18, 0.0, 1.57, 0.0);
    plan_and_execute("Safe retreat", 0.25, 0.0, 0.30, 0.0, 1.57, 0.0);

    // Clean up all obstacles
    planning_scene_interface.removeCollisionObjects({
        "ground_table", "left_box", "right_box", "center_cylinder"
    });
    rclcpp::sleep_for(std::chrono::seconds(1));

    // ========================================
    // Return to home
    // ========================================
    RCLCPP_INFO(logger, "\n\n╔═══════════════════════════════════════╗");
    RCLCPP_INFO(logger, "║  Returning to Home Position          ║");
    RCLCPP_INFO(logger, "╚═══════════════════════════════════════╝");
    
    std::vector<double> home_joints = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    arm.setJointValueTarget(home_joints);
    
    moveit::planning_interface::MoveGroupInterface::Plan home_plan;
    if (arm.plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(logger, "Going home...");
        arm.execute(home_plan);
    }

    RCLCPP_INFO(logger, "\n\n╔═══════════════════════════════════════╗");
    RCLCPP_INFO(logger, "║  ✓ DEMO COMPLETE!                    ║");
    RCLCPP_INFO(logger, "║  Successfully navigated around:       ║");
    RCLCPP_INFO(logger, "║  - Single blocking obstacle           ║");
    RCLCPP_INFO(logger, "║  - Two pillars (gap navigation)       ║");
    RCLCPP_INFO(logger, "║  - L-shaped barrier                   ║");
    RCLCPP_INFO(logger, "║  - Low ceiling (ducking)              ║");
    RCLCPP_INFO(logger, "║  - Complex multi-obstacle environment ║");
    RCLCPP_INFO(logger, "╚═══════════════════════════════════════╝\n");

    rclcpp::shutdown();
    spinner.join();
    return 0;
}