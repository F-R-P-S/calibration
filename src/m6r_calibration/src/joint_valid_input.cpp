#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("test_moveit_from_file");

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    auto spinner = std::thread([&executor]() { executor.spin(); });

    auto logger = node->get_logger();

    // Initialize MoveGroupInterface
    auto arm = moveit::planning_interface::MoveGroupInterface(node, "arm");
    arm.setMaxVelocityScalingFactor(0.2);      // scale down for smooth motion
    arm.setMaxAccelerationScalingFactor(0.2);

    // Open the joint command file
    std::ifstream infile("joint_velocities_log.txt");
    if (!infile.is_open()) {
        RCLCPP_ERROR(logger, "Failed to open joint_commands.txt");
        rclcpp::shutdown();
        spinner.join();
        return 1;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::vector<double> joint_positions;
        std::string value;
        int term_count = 0; // Initialize counter for terms
        while (std::getline(ss, value, ' ')) {
            // ignore first two values as timestamps
           
            term_count++;         
            if (term_count <= 2) {
                continue;
            }     
            joint_positions.push_back(std::stod(value));
        }

        if (joint_positions.size() != 6) {
            RCLCPP_WARN(logger, "Skipping invalid line (expected 6 values): %s", line.c_str());
            continue;
        }

        arm.setStartStateToCurrentState();
        arm.setJointValueTarget(joint_positions);
        moveit::planning_interface::MoveGroupInterface::Plan plan;

        bool success = (arm.plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
        if (success) {
            arm.execute(plan);
        } else {
            RCLCPP_WARN(logger, "Failed to plan for joint positions");
        }

        // Optional: sleep to control execution rate
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 20 Hz
    }

    infile.close();

    rclcpp::shutdown();
    spinner.join();
    return 0;
}
