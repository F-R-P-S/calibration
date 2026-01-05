from launch import LaunchDescription
from launch.actions import TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get package paths
    m6r_description_share = FindPackageShare('m6r_description')
    m6r_bringup_share = FindPackageShare('m6r_bringup')
    m6r_moveit_config_share = FindPackageShare('m6r_moveit_config')
    
    # Define paths
    urdf_path = PathJoinSubstitution([m6r_description_share, 'urdf', 'm6r.urdf.xacro'])
    rviz_config_path = PathJoinSubstitution([m6r_description_share, 'rviz', 'm6r_bringup.rviz'])
    ros2_controllers_path = PathJoinSubstitution([m6r_bringup_share, 'config', 'ros2_controllers.yaml'])
    
    # Robot State Publisher - publishes TF transforms
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_path])
        }],
        output='screen'
    )
    
    # ROS2 Control Node - controller manager with robot_description
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': Command(['xacro ', urdf_path])},
            ros2_controllers_path
        ],
        output='screen'
    )
    
    # Spawn joint_state_broadcaster with delay (5 seconds)
    # This ensures controller_manager is ready
    spawn_joint_state_broadcaster = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen'
            )
        ]
    )
    
    # Spawn arm_controller with slightly longer delay (6 seconds)
    spawn_arm_controller = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['arm_controller'],
                output='screen'
            )
        ]
    )
    
    # Include MoveIt2 move_group launch file with delay (7 seconds)
    # This ensures controllers are loaded before MoveIt2 starts
    move_group = TimerAction(
        period=7.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        m6r_moveit_config_share,
                        'launch',
                        'move_group.launch.py'
                    ])
                ])
            )
        ]
    )
    
    # RViz2
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    
    return LaunchDescription([
        robot_state_publisher,
        ros2_control_node,
        spawn_joint_state_broadcaster,
        spawn_arm_controller,
        move_group,
        rviz2
    ])