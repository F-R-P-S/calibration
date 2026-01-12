import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import math
import time
from pathlib import Path
from datetime import datetime

# TF2 imports
from tf2_ros import TransformException, Buffer, TransformListener
from scipy.spatial.transform import Rotation as R

class M6RLegacyCalibrator(Node):
    def __init__(self):
        super().__init__('m6r_legacy_calibrator')
        self.bridge = CvBridge()
        
        # 1. Output Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"calib_results_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        self.data_file = self.output_dir / "hand_eye_data.yaml"

        # 2. TF2 & Action Client
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

        # 3. Camera Subscriptions
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        
        self.latest_frame = None
        self.mtx = None
        self.dist = None
        
        # =============================================================
        # 4. LEGACY CHARUCO SETUP (OpenCV 4.5.x API)
        # =============================================================
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Legacy constructor: (squaresX, squaresY, squareLength, markerLength, dictionary)
        self.board = cv2.aruco.CharucoBoard_create(4, 4, 0.05, 0.03, self.aruco_dict)
        # =============================================================

        self.pose_count = 0
        self.get_logger().info("✓ Legacy Calibrator Initialized (OpenCV 4.5.4). Waiting for Camera...")

    def info_callback(self, msg):
        if self.mtx is None:
            self.mtx = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d)
            self.get_logger().info("✓ Camera Matrix Received.")

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def get_robot_pose(self):
        try:
            # Capture transform from base to optical frame
            t = self.tf_buffer.lookup_transform('base_link', 'camera_link_optical', rclpy.time.Time())
            pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            return R.from_quat(quat).as_matrix().tolist(), pos
        except TransformException as e:
            self.get_logger().error(f"TF Failed: {e}")
            return None, None

    def move_robot(self, joints_rad):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        point = JointTrajectoryPoint(positions=joints_rad)
        point.time_from_start.sec = 4
        goal.trajectory.points.append(point)
        
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        res_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        time.sleep(2.0) 

    def detect_charuco(self):
        if self.latest_frame is None or self.mtx is None:
            return None, None, False

        # Step A: Detect Markers
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            self.latest_frame, self.aruco_dict, parameters=self.aruco_params
        )
        
        debug_img = self.latest_frame.copy()
        
        if marker_ids is not None:
            # Step B: Interpolate Chessboard Corners
            charuco_ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, self.latest_frame, self.board
            )
            
            # Step C: Estimate Pose if enough corners found
            if charuco_ret > 4:
                cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
                
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board, self.mtx, self.dist, None, None
                )
                
                if valid:
                    cv2.drawFrameAxes(debug_img, self.mtx, self.dist, rvec, tvec, 0.1)
                    cv2.imwrite(str(self.output_dir / f"pose_{self.pose_count}.jpg"), debug_img)
                    rot_mat, _ = cv2.Rodrigues(rvec)
                    return rot_mat.tolist(), tvec.flatten().tolist(), True
        
        return None, None, False

    def active_search_loop(self, base_pose, attempt=0):
        if attempt > 3:
            self.get_logger().error(f"✗ Failed to find board at Pose {self.pose_count} after 4 tries.")
            self.pose_count += 1 # Increment even on failure to stay in sync
            return

        # Micro-search pattern (wrist movements)
        search_offset = [0.0, 0.0, 0.0, 0.0, math.radians(attempt * 4), math.radians(attempt * 8)]
        current_target = [a + b for a, b in zip(base_pose, search_offset)]
        
        self.move_robot(current_target)
        # Process pending image callbacks
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        rot, trans, found = self.detect_charuco()

        if found:
            robot_r, robot_t = self.get_robot_pose()
            if robot_r:
                self.save_data(robot_r, robot_t, rot, trans)
        else:
            self.get_logger().warn(f"Board not found. Retrying with offset (Attempt {attempt + 1})...")
            self.active_search_loop(base_pose, attempt + 1)

    def save_data(self, rob_r, rob_t, cam_r, cam_t):
        sample = {
            f'sample_{self.pose_count}': {
                'robot_effector': {'R': rob_r, 'T': rob_t},
                'marker_camera': {'R': cam_r, 'T': cam_t}
            }
        }
        with open(self.data_file, 'a') as f:
            yaml.dump(sample, f, default_flow_style=False)
        self.get_logger().info(f"✓✓ Sample {self.pose_count} Saved!")
        self.pose_count += 1

def main():
    rclpy.init()
    node = M6RLegacyCalibrator()

    # Dynamic Pose Grid
    pans = [-20, 0, 20]
    tilts = [45, 60]
    calibration_grid = []
    for p in pans:
        for t in tilts:
            calibration_grid.append([math.radians(p), math.radians(45), math.radians(50), 0.0, math.radians(t), 0.0])

    try:
        for target in calibration_grid:
            node.active_search_loop(target)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()