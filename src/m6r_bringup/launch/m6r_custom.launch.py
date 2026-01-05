import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import yaml

def load_yaml(package_name, file_path):
    """Load a yaml file from package"""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_launch_description():

    declare_x = DeclareLaunchArgument("base_x", default_value="0.0")
    declare_y = DeclareLaunchArgument("base_y", default_value="0.0")
    declare_z = DeclareLaunchArgument("base_z", default_value="0.0")
    declare_yaw = DeclareLaunchArgument("base_yaw", default_value="0.0")

    x = LaunchConfiguration("base_x")
    y = LaunchConfiguration("base_y")
    z = LaunchConfiguration("base_z")
    yaw = LaunchConfiguration("base_yaw")
    
    # Package paths
    m6r_description_share = FindPackageShare('m6r_description')
    m6r_bringup_share = FindPackageShare('m6r_bringup')
    m6r_moveit_share = FindPackageShare('m6r_moveit_config')
    
    # Paths
    urdf_path = PathJoinSubstitution([m6r_description_share, 'urdf', 'm6r.urdf.xacro'])
    rviz_config_path = PathJoinSubstitution([m6r_description_share, 'rviz', 'calib_bringup.rviz'])
    controllers_config = PathJoinSubstitution([m6r_bringup_share, 'config', 'ros2_controllers.yaml'])
    
    # Robot description
    robot_description = Command(['xacro ', urdf_path])
    
    # Static transform from world to robot base (calibrated position)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_base_link',
        arguments=[
            '--x', x,
            '--y', y,
            '--z', z,
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', yaw,
            '--frame-id', 'world',
            '--child-frame-id', 'base_link'
        ],
        output='screen'
    )
    
    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'frame_prefix': ''  # No prefix, base_link is the root
        }],
        output='screen'
    )
    
    # Controller manager
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_description},
            controllers_config
        ],
        output='screen'
    )
    
    # Spawn arm controller
    spawn_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller'],
        output='screen'
    )
    
    # Spawn joint state broadcaster
    spawn_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )
    
    # MoveIt
    moveit_launch = IncludeLaunchDescription(
        PathJoinSubstitution([m6r_moveit_share, 'launch', 'move_group.launch.py'])
    )
    
    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    
    return LaunchDescription([
        declare_x,
        declare_y,
        declare_z,
        declare_yaw,
        static_tf_node,  # Publish world -> base_link transform first
        robot_state_publisher_node,
        controller_manager_node,
        spawn_arm_controller,
        spawn_joint_state_broadcaster,
        moveit_launch,
        rviz_node
    ])