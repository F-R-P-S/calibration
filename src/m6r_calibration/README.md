# M6R Commander C++ Package

## ⚙️ Setup and Usage

To create and run a new C++ file in this package:

1. Place your new `.cpp` file inside the `src/` directory.
2. Open `CMakeLists.txt` and add the following lines (replace `your_node` with your file name):

```cmake
add_executable(your_node src/your_node.cpp)
ament_target_dependencies(your_node rclcpp)
install(TARGETS your_node DESTINATION lib/${PROJECT_NAME})
```

<!-- Each time you modify or add a C++ file, rebuild the package -->

Each time you modify or add a C++ file, rebuild the package:

```bash
colcon build --packages-select m6r_commander_cpp
source install/setup.bash
```

To run the C++ program:

```bash
ros2 run m6r_commander_cpp <node_name>
```

---

## Commander.cpp

The OOP class combines the above functions and creates a topic for different commands.

To run it, first start the bringup simulation, and in a second terminal start the commander:

```bash
ros2 run m6r_commander_cpp commander
```

In a third terminal:

1. Check if the newly added topic in the class is shown (here using `/joint_cmd` as an example):

```bash
ros2 topic list
```

2. Publish data into the topic and check Rviz2 for updates:

```bash
ros2 topic pub -1 /joint_cmd example_interfaces/msg/Float64MultiArray "data: [-0.5, 0.5, 0.0, 0.0, 0.0, 0.4]"
```


## Position Control with custom environment
Under src/position_control, have several packages for generating obstacles and test movement planner.


```bash
ros2 run m6r_commander_cpp collision_load   --ros-args   -p target_x:=0.6   -p target_y:=0.0   -p target_z:=0.25
```




