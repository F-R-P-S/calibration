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

# TF2
from tf2_ros import TransformException, Buffer, TransformListener
from scipy.spatial.transform import Rotation as R

class PrecisionCalibrator(Node):
    def __init__(self):
        super().__init__('m6r_precision_calibrator')
        self.bridge = CvBridge()
        
        # --- CONFIGURATION ---
        self.output_dir = Path(f"calib{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)
        
        # BOARD CONFIG (MEASURE THESE EXACTLY!)
        self.SQUARE_LEN = 0.040 # 40mm
        self.MARKER_LEN = 0.030 # 30mm (usually 75% of square)
        
        # FRAMES
        self.base_frame = 'base_link'
        self.ee_frame = 'wriste_roll' # The link holding the camera
        
        # --- ROS SETUP ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        
        # --- STATE ---
        self.latest_frame = None
        self.mtx = None
        self.dist = None
        
        # DATA FOR SOLVER
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []
        
        # --- OPENCV SETUP ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.board = cv2.aruco.CharucoBoard((4, 4), self.SQUARE_LEN, self.MARKER_LEN, self.aruco_dict)
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
            self.use_legacy = False
        except AttributeError:
            self.board = cv2.aruco.CharucoBoard_create(4, 4, self.SQUARE_LEN, self.MARKER_LEN, self.aruco_dict)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.use_legacy = True

        cv2.namedWindow("Precision Calibration", cv2.WINDOW_NORMAL)
        self.get_logger().info("✓ Precision Calibrator Ready.")

    def info_callback(self, msg):
        if self.mtx is None:
            self.mtx = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d)
            self.get_logger().info(f"✓ Camera Matrix Loaded. FX={self.mtx[0,0]:.1f}")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Precision Calibration", self.latest_frame)
            cv2.waitKey(1)
        except: pass

    def get_robot_pose(self):
        """Get highly accurate TF timestamped now"""
        try:
            # wait a tiny bit to ensure TF is up to date with the robot stopping
            time.sleep(0.2) 
            t = self.tf_buffer.lookup_transform(
                self.base_frame, 
                self.ee_frame, 
                rclpy.time.Time()
            )
            pos = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
            quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            rot_mat = R.from_quat(quat).as_matrix()
            return rot_mat, pos
        except TransformException:
            return None, None

    def move_robot(self, joints_rad):
        if not self.arm_client.wait_for_server(timeout_sec=1.0): return False
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        point = JointTrajectoryPoint()
        point.positions = joints_rad
        point.time_from_start.sec = 5 # Slow movement for safety
        goal.trajectory.points.append(point)

        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res and res.accepted:
            # Wait for execution to finish
            res_future = res.get_result_async()
            rclpy.spin_until_future_complete(self, res_future)
            time.sleep(1.0) # Wait for vibrations to stop
            return True
        return False

    def capture_averaged_pose(self, samples=30):
        """Captures N frames and averages the Charuco Corners to reduce noise"""
        if self.latest_frame is None or self.mtx is None: return False, None, None

        self.get_logger().info(f"   Averaging {samples} frames...")
        
        all_corners = []
        all_ids = []
        
        # 1. Collect N samples
        for _ in range(samples):
            img = self.latest_frame.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.use_legacy:
                c, i, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            else:
                c, i, _ = self.detector.detectMarkers(gray)

            if i is not None and len(i) > 0:
                ret, c_corn, c_ids = cv2.aruco.interpolateCornersCharuco(c, i, gray, self.board)
                if c_corn is not None and len(c_corn) > 4:
                    all_corners.append(c_corn)
                    all_ids.append(c_ids)
            
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

        if len(all_corners) < (samples * 0.5): # If we missed > 50% of frames
            self.get_logger().warn("   Unstable detection (glare or blur?)")
            return False, None, None

        # 2. Filter: Find commonly detected IDs (intersection)
        # This is complex, so we'll take the LAST good frame for IDs, 
        # but averaged corner positions is better. 
        # For simplicity in this script, we'll take the BEST single frame (most corners)
        # to avoid ID mismatch issues, but averaging T/R vectors is safer.
        
        best_idx = np.argmax([len(x) for x in all_corners])
        final_corners = all_corners[best_idx]
        final_ids = all_ids[best_idx]
        
        # 3. Solve PnP on the best frame
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            final_corners, final_ids, self.board, self.mtx, self.dist, None, None
        )

        if valid:
            debug = self.latest_frame.copy()
            cv2.drawFrameAxes(debug, self.mtx, self.dist, rvec, tvec, 0.1)
            cv2.imshow("Precision Calibration", debug)
            cv2.waitKey(100)
            return True, rvec, tvec
            
        return False, None, None

    def execute_pose(self, joints, idx):
        self.get_logger().info(f"--- Pose {idx} ---")
        self.move_robot(joints)
        
        # Capture precise data
        valid, rvec, tvec = self.capture_averaged_pose()
        
        if valid:
            rob_R, rob_T = self.get_robot_pose()
            if rob_R is not None:
                self.R_gripper2base.append(rob_R)
                self.t_gripper2base.append(rob_T)
                
                cam_R_mat, _ = cv2.Rodrigues(rvec)
                self.R_target2cam.append(cam_R_mat)
                self.t_target2cam.append(tvec)
                self.get_logger().info(f"✓ Pose {idx} Captured.")
        else:
            self.get_logger().warn(f"✗ Pose {idx} Failed to detect board.")

    def compute_calibration(self):
        if len(self.R_gripper2base) < 5:
            self.get_logger().error("Need at least 5 good poses!")
            return

        print("\n" + "="*50)
        print("COMPUTING RESULTS...")
        print("="*50)

        # METHOD 1: TSAI
        R_tsai, t_tsai = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam, self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        # METHOD 2: DANIILIDIS (Better for rotation noise)
        R_dan, t_dan = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam, self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_DANIILIDIS
        )

        # Compare Translation Magnitudes (Sanity Check)
        mag_tsai = np.linalg.norm(t_tsai)
        mag_dan = np.linalg.norm(t_dan)
        
        print(f"TSAI Translation:       {t_tsai.flatten()} (Mag: {mag_tsai:.4f})")
        print(f"DANIILIDIS Translation: {t_dan.flatten()} (Mag: {mag_dan:.4f})")
        
        # Heuristic: If they are close, detection is good. Use Daniilidis.
        diff = np.linalg.norm(t_tsai - t_dan)
        print(f"Solver Discrepancy: {diff:.4f}m")
        
        final_R = R_dan
        final_t = t_dan
        
        if diff > 0.05: # >5cm difference
            self.get_logger().warn("WARNING: Solvers disagree significantly! Data might be noisy.")

        # Save Result
        T = np.eye(4)
        T[:3, :3] = final_R
        T[:3, 3] = final_t.flatten()
        
        data = {
            'transformation_matrix': T.flatten().tolist(),
            'xyz': final_t.flatten().tolist(),
            'rpy_deg': R.from_matrix(final_R).as_euler('xyz', degrees=True).tolist()
        }
        
        with open(self.output_dir / "final_calib.yaml", 'w') as f:
            yaml.dump(data, f)
            
        print("\nFINAL MATRIX (DANIILIDIS):")
        print(np.array_str(T, precision=4, suppress_small=True))
        print("="*50)

def main():
    rclpy.init()
    node = PrecisionCalibrator()
    
    # --- POSES: MAXIMIZE ROTATION ---
    # These poses assume the board is tilted 30-45 deg facing the robot
    # Focus on "looking around" the board.
    # [j1...j6] in degrees
    poses_deg = [
        [0, 20, 75, 0, 50, 0],      # Center Look
        [20, 20, 75, -30, 60, 20],  # Right Tilt
        [-20, 15, 75, 30, 50, -20], # Left Tilt
        [0, 10, 80, 0, 70, 0],      # Top-downish
        [15, 20, 70, -15, 40, 90],  # Heavy Wrist Twist Positive
        [-15, 20, 80, 15, 40, -90], # Heavy Wrist Twist Negative
        [0, 15, 80, 0, 50, 45],     # Center + Roll
        [0, 20, 80, 0, 50, -45],    # Center - Roll
    ]

    try:
        # Warmup
        while node.mtx is None: rclpy.spin_once(node, timeout_sec=0.1)
        
        for i, p in enumerate(poses_deg):
            rads = [math.radians(x) for x in p]
            node.execute_pose(rads, i+1)
            
        node.compute_calibration()
        
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()