#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

int main(int argc, char **argv)
{

    // initialize ROS2
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("test_moveit");
    
    rclcpp::executors::SingleThreadedExecutor executor; // create executor
    executor.add_node(node); // add node to executor
    auto spinner = std::thread([&executor]() { executor.spin(); });// create thread and make it spin

    // Initialize MoveGroupInterface for the "arm" planning group
    auto arm = moveit::planning_interface::MoveGroupInterface(node, "arm");
    arm.setMaxVelocityScalingFactor(1.0);
    arm.setMaxAccelerationScalingFactor(1.0);

    // Named Goal

    //set the start state to the current state
    arm.setStartStateToCurrentState();
    arm.setNamedTarget("pose_1");
    moveit::planning_interface::MoveGroupInterface::Plan plan1;
    // Plan to the named target
    bool success1 = (arm.plan(plan1) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    //execute the plan if successful
    if (success1){
        arm.execute(plan1);
    }

    arm.setStartStateToCurrentState();
    arm.setNamedTarget("home");
    moveit::planning_interface::MoveGroupInterface::Plan plan2;
    // Plan to the named target
    bool success2 = (arm.plan(plan2) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    //execute the plan if successful
    if (success2){
        arm.execute(plan2);
    }

    arm.setStartStateToCurrentState();
    arm.setNamedTarget("pose_1");
    success1 = (arm.plan(plan1) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    //execute the plan if successful
    if (success1){
        arm.execute(plan1);
    }

    //Joint Goal
    


    rclcpp::shutdown();
    spinner.join();
    return 0;
}