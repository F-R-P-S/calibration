import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # =============================
    # Package & File Paths
    # =============================
    description_pkg = "m6r_description"
    pkg_share_path = get_package_share_directory(description_pkg)
    
    # Path to your XACRO file
    urdf_path = os.path.join(pkg_share_path, "urdf", "m6r.urdf.xacro")
    rviz_config_path = os.path.join(pkg_share_path, "rviz", "gazebo_bringup.rviz")

    # =============================
    # Environment Variables
    # =============================
    # Gazebo Classic needs to know where the meshes are
    # This points to the 'share' folder so 'package://m6r_description' resolves
    set_model_path = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=os.path.dirname(pkg_share_path)
    )

    # =============================
    # 1. Robot State Publisher
    # =============================
    # Converts Xacro to URDF and publishes to /robot_description
    robot_description_content = Command(["xacro ", urdf_path])
    
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description_content,
            "use_sim_time": True
        }]
    )

    # =============================
    # 2. Gazebo Classic Server/Client
    # =============================
    gazebo = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare("gazebo_ros"),
            "launch",
            "gazebo.launch.py"
        ]),
        launch_arguments={'pause': 'false'}.items()
    )

    # =============================
    # 3. Spawn Robot Entity
    # =============================
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "m6r_robot",
            "-z", "0.1" # Spawn slightly above ground to avoid collision physics crash
        ],
        output="screen"
    )

    # =============================
    # 4. ROS 2 Controllers
    # =============================
    # Controller 1: Broadcaster
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen"
    )

    # Controller 2: Arm Position/Trajectory Controller
    arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        output="screen"
    )

    # =============================
    # 5. Event Handlers (The "Glue")
    # =============================
    
    # DON'T start broadcasters until the robot is spawned in Gazebo
    delay_jsb_after_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster],
        )
    )

    # DON'T start arm_controller until joint_state_broadcaster is active
    delay_arm_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[arm_controller],
        )
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
        parameters=[{'use_sim_time': True}] # CRITICAL: Syncs RViz with Gazebo
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # x y z yaw pitch roll parent_frame child_frame
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
    )

    return LaunchDescription([
        set_model_path,
        gazebo,
        robot_state_publisher,
        static_tf,
        spawn_robot,
        delay_jsb_after_spawn,
        delay_arm_after_jsb,
        # delay_moveit,        
        rviz_node
    ])