#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2/LinearMath/Quaternion.h>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("test_moveit");
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    auto spinner = std::thread([&executor]() { executor.spin(); });

    auto arm = moveit::planning_interface::MoveGroupInterface(node, "arm");
    arm.setMaxVelocityScalingFactor(1.0);
    arm.setMaxAccelerationScalingFactor(1.0);

    // Publisher to send joint angles (for real robot control via Phidget later)
    auto joint_pub = node->create_publisher<std_msgs::msg::Float64MultiArray>("/joint_angles", 10);

    // Pose Goal
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, 0.0);
    q = q.normalize();

    geometry_msgs::msg::PoseStamped target_pose;
    target_pose.header.frame_id = "base_link";
    target_pose.pose.position.x = 0.5;
    target_pose.pose.position.y = 0.0;
    target_pose.pose.position.z = 0.4;
    target_pose.pose.orientation.x = q.getX();
    target_pose.pose.orientation.y = q.getY();
    target_pose.pose.orientation.z = q.getZ();
    target_pose.pose.orientation.w = q.getW();

    arm.setStartStateToCurrentState();
    arm.setPoseTarget(target_pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan1;
    bool success1 = (arm.plan(plan1) == moveit::core::MoveItErrorCode::SUCCESS);

    if (success1) {
        arm.execute(plan1);
    }

    // Cartesian Path
    std::vector<geometry_msgs::msg::Pose> waypoints;
    geometry_msgs::msg::Pose pose1 = arm.getCurrentPose().pose;
    pose1.position.z += -0.2;
    waypoints.push_back(pose1);
    geometry_msgs::msg::Pose pose2 = pose1;
    pose2.position.y += 0.2;
    waypoints.push_back(pose2); 
    geometry_msgs::msg::Pose pose3 = pose2;
    pose3.position.y += -0.2;
    pose3.position.z += 0.2;
    waypoints.push_back(pose3);

    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = arm.computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

    if (fraction == 1.0) {
        arm.execute(trajectory);

        // Real-time joint angle publishing
        for (const auto &point : trajectory.joint_trajectory.points)
        {
            std_msgs::msg::Float64MultiArray msg;
            msg.data = point.positions;  // Joint angles in radians

            joint_pub->publish(msg);

            // Log to terminal as well
            RCLCPP_INFO(node->get_logger(), "Joint Angles:");
            for (size_t i = 0; i < msg.data.size(); ++i)
                RCLCPP_INFO(node->get_logger(), "  Joint %zu: %.4f rad", i + 1, msg.data[i]);

            // Mimic real-time rate (optional)
            rclcpp::sleep_for(std::chrono::milliseconds(100));
        }
    }

    rclcpp::shutdown();
    spinner.join();
    return 0;
}
