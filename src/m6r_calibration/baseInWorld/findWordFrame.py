import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformException


class AutoBaseCalibrator(Node):
    def __init__(self):
        super().__init__('auto_base_calibrator')
        self.bridge = CvBridge()

        # ================================================================
        # 1. CONFIGURATION
        # ================================================================

        # World -> Tag pose (known)
        self.T_world_tag = np.eye(4)
        self.T_world_tag[:3, 3] = [1.0, 0.0, 0.0]
        self.T_world_tag[:3, :3] = R.from_euler(
            'xyz', [0, 0, 0], degrees=True
        ).as_matrix()

        # ArUco details
        self.MARKER_ID = 0
        self.MARKER_SIZE = 0.20  # meters

        # Hand–eye calibration (EE → Camera)
        self.T_ee_cam = np.eye(4)
        self.T_ee_cam[:3, 3] = [-0.8, 0.0252, 0.1855]
        self.T_ee_cam[:3, :3] = R.from_euler(
            'xyz', [96, -56, 78], degrees=True
        ).as_matrix()

        # ================================================================
        # ROS setup
        # ================================================================

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10
        )

        self.mtx = None
        self.dist = None
        self.latest_frame = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.get_logger().info("AutoBaseCalibrator initialized")

    def info_callback(self, msg):
        if self.mtx is None:
            self.mtx = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d) if len(msg.d) else np.zeros(5)
            self.get_logger().info("Camera intrinsics loaded")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(str(e))

    def move_robot(self, joints_rad):
        if not self.arm_client.wait_for_server(timeout_sec=1.0): return False
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        point = JointTrajectoryPoint()
        point.positions = joints_rad
        point.time_from_start.sec = 2 
        goal.trajectory.points.append(point)

        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res and res.accepted:
            res_future = res.get_result_async()
            rclpy.spin_until_future_complete(self, res_future)
            time.sleep(1.0) 
            return True
        return False

    def get_tf_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'wriste_roll', rclpy.time.Time()
            )
            pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            return T
        except TransformException as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

    def calculate_base(self):
        if self.latest_frame is None:
            self.get_logger().error("No image received")
            return

        gray = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None:
            self.get_logger().error("No ArUco markers detected")
            return
        
        ids = ids.flatten()
        if self.MARKER_ID not in ids:
            self.get_logger().error("Target marker not found")
            return

        idx = np.where(ids == self.MARKER_ID)[0][0]

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[idx], self.MARKER_SIZE, self.mtx, self.dist
        )

        rvec = rvecs[0][0]
        tvec = tvecs[0][0]

        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3], _ = cv2.Rodrigues(rvec)
        T_cam_tag[:3, 3] = tvec

        # --- NEW: PRINT DETECTED TAG ROTATION ---
        tag_rpy = R.from_matrix(T_cam_tag[:3, :3]).as_euler('xyz', degrees=True)
        print(f"\n[VISION] Detected Tag Rotation (Relative to Camera):")
        print(f"  Roll:  {tag_rpy[0]:.1f}")
        print(f"  Pitch: {tag_rpy[1]:.1f}")
        print(f"  Yaw:   {tag_rpy[2]:.1f}")
        print(f"  Dist:  {np.linalg.norm(tvec):.3f} m")
        # ----------------------------------------

        T_base_ee = self.get_tf_pose()
        if T_base_ee is None: return

        T_world_base = (
            self.T_world_tag @
            np.linalg.inv(T_cam_tag) @
            np.linalg.inv(self.T_ee_cam) @
            np.linalg.inv(T_base_ee)
        )

        self.print_results(T_world_base)

    def print_results(self, T):
        xyz = T[:3, 3]
        rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)

        print("-" * 40)
        print("CALCULATED ROBOT BASE POSE")
        print(f"X: {xyz[0]:.4f} m | Y: {xyz[1]:.4f} m | Z: {xyz[2]:.4f} m")
        print(f"RPY: {np.round(rpy, 2)}")
        print("=" * 40 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = AutoBaseCalibrator()

    while node.mtx is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    
    poses_deg = [
        [0, 10, 85, 0, 70, 45],     # Center + Roll
        [-20, 15, 85, 30, 50, -40], # Left Tilt
    ]
    for pose in poses_deg:
        if node.move_robot([math.radians(d) for d in pose]):
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.1)
            node.calculate_base()        

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()