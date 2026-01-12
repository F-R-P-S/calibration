import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import math
import time

# TF2 imports for robot pose capture
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R

class M6RCalibrationExecutor(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name=node_name)
        self.bridge = CvBridge()
        
        # 1. TF2 setup to capture base_link -> wriste_roll
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 2. Movement Action Client
        self.joint_trajectory_action_client_ = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        
        # 3. Camera Subscriber (captures from ROS 2 topic)
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.latest_frame = None

        # 4. ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.05  # 5cm marker
        
        # 5. Load Camera Calibration (K and D matrices)
        cv_file = cv2.FileStorage("camera_params.yaml", cv2.FILE_STORAGE_READ)
        self.mtx = cv_file.getNode('K').mat()
        self.dst = cv_file.getNode('D').mat()
        cv_file.release()

        self.pose_count = 0
        self.data_file = "hand_eye_data.yaml"

        while not self.joint_trajectory_action_client_.wait_for_server(1):
            self.get_logger().info("Waiting for action server...")

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def get_robot_pose(self):
        try:
            # Capture the transform at the current time
            t = self.tf_buffer.lookup_transform('base_link', 'wriste_roll', rclpy.time.Time())
            pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            rot_matrix = R.from_quat(quat).as_matrix()
            return rot_matrix.tolist(), pos
        except TransformException as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            return None, None

    def execute_and_detect(self, positions: list, sec_from_start: int = 4):
        # --- Movement Part ---
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        point = JointTrajectoryPoint(positions=positions)
        point.time_from_start.sec = sec_from_start
        goal.trajectory.points.append(point)

        self.get_logger().info(f"Moving to Pose {self.pose_count}...")
        future = self.joint_trajectory_action_client_.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        # Stability delay for Gazebo physics
        time.sleep(2.0)

        # --- Detection Part ---
        if self.latest_frame is not None:
            corners, ids, _ = cv2.aruco.detectMarkers(self.latest_frame, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.mtx, self.dst)
                cam_R, _ = cv2.Rodrigues(rvecs[0])
                cam_T = tvecs[0][0].tolist()
                
                robot_R, robot_T = self.get_robot_pose()
                
                if robot_R:
                    sample = {
                        f'sample_{self.pose_count}': {
                            'robot_effector': {'R': robot_R, 'T': robot_T},
                            'marker_camera': {'R': cam_R.tolist(), 'T': cam_T}
                        }
                    }
                    with open(self.data_file, 'a') as f:
                        yaml.dump(sample, f)
                    self.get_logger().info(f"Sample {self.pose_count} saved successfully.")
                    self.pose_count += 1
            else:
                self.get_logger().warn("Marker not found in this pose.")

def main(args=None):
    rclpy.init(args=args)
    node = M6RCalibrationExecutor("m6r_calibrator")

    poses_degrees = [
        [0.0, 20.0, 15.0, 0.0, 20.0, 0.0],
        [30.0, 20.0, 30.0, 10.0, 45.0, 0.0],
        [-30.0, 20.0, 15.0, -10.0, 90.0, 0.0],
        [0.0, 2.0, 34.0, 0.0, 80.0, 0.0]
    ]

    poses = [[math.radians(a) for a in p] for p in poses_degrees]

    try:
        for pose in poses:
            node.execute_and_detect(pose)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()