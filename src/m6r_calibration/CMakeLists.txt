cmake_minimum_required(VERSION 3.8)
project(m6r_calibration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
# find_package(example_interfaces REQUIRED)
# add custom package
# find_package(m6r_interfaces REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)



# custome executables

add_executable(test_moveit src/test_moveit.cpp)
ament_target_dependencies(test_moveit rclcpp moveit_ros_planning_interface)

# position control with custom obstacles
# -------------------------------------------
add_executable(pose_obstacle src/position_control/pose_obstacle.cpp)
ament_target_dependencies(pose_obstacle rclcpp moveit_ros_planning_interface)

add_executable(pose_control src/position_control/pose_control.cpp)
ament_target_dependencies(pose_control rclcpp moveit_ros_planning_interface)

add_executable(collision_free_demo src/position_control/collision_free_demo.cpp)
ament_target_dependencies(collision_free_demo rclcpp moveit_ros_planning_interface)

add_executable(collision_load src/position_control/collision_load.cpp)
ament_target_dependencies(collision_load rclcpp yaml-cpp moveit_ros_planning_interface)

# CRITICAL: Link yaml-cpp library
target_link_libraries(collision_load
  ${YAML_CPP_LIBRARIES}  # Link yaml-cpp
)

# add_executable(generate_obstacle src/position_control/generate_obstacle.cpp)
# ament_target_dependencies(generate_obstacle rclcpp moveit_ros_planning_interface)


# ==========================================
add_executable(joint_control src/joint_control.cpp)
ament_target_dependencies(joint_control rclcpp moveit_ros_planning_interface)

add_executable(valid_joint src/joint_valid_input.cpp)
ament_target_dependencies(valid_joint rclcpp moveit_ros_planning_interface)

# add_executable(commander src/commander.cpp)
# ament_target_dependencies(commander rclcpp example_interfaces moveit_ros_planning_interface m6r_interfaces)


# ------------------------------------------
add_executable(findWordFrame baseInWorld/findWordFrame.cpp)
ament_target_dependencies(findWordFrame rclcpp moveit_ros_planning_interface)



install(TARGETS 
  test_moveit
  joint_control
  valid_joint
  # position control with custom obstacles
  pose_control
  pose_obstacle
  collision_free_demo
  collision_load

  # world Calibration
  findWordFrame
  
  # commander
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
