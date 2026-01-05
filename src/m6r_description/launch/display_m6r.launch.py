from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    declared_arguments = []
    
    # Declare arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="m6r_description",
            description="Description package with robot URDF/XACRO files. Usually the argument "
            "is not set, it enables use of a custom description.",
        )
    )
    
    description_package = LaunchConfiguration("description_package")
    
    # Get robot description - NOTE: Using URDF directly, not xacro
    # If your file is pure URDF (not xacro), use cat instead of xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="cat")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare(description_package), "urdf", "m6r.urdf"]
            ),
        ]
    )
    
    robot_description = {
        "robot_description": ParameterValue(value=robot_description_content, value_type=str)
    }
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[robot_description], 
    )
    
    # Joint State Publisher GUI
    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="log",
    )
    
    # RViz
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(description_package), "rviz", "viwe_m6r.rviz"]
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )
    
    return LaunchDescription(
        declared_arguments + [
            robot_state_publisher,
            joint_state_publisher_gui,
            rviz_node,
        ]
    )