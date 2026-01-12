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
import os
from pathlib import Path
from datetime import datetime

# TF2 imports for robot pose capture
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R


class M6RCalibrationExecutor(Node):
    def __init__(self, node_name: str = "m6r_calibrator") -> None:
        super().__init__(node_name=node_name)
        self.bridge = CvBridge()
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"output_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        self.get_logger().info(f"✓ Created output directory: {self.output_dir}")
        
        # 1. TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 2. Movement Action Client
        self.joint_trajectory_action_client_ = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        
        # 3. Camera Subscriber
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera_info", self.camera_info_callback, 10
        )
        
        self.latest_frame = None
        self.camera_info_received = False
        self.mtx = None
        self.dist = None
        self.frame_count = 0

        # 4. ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            # More sensitive detection
            self.aruco_params.adaptiveThreshWinSizeMin = 3
            self.aruco_params.adaptiveThreshWinSizeMax = 23
            self.aruco_params.adaptiveThreshWinSizeStep = 10
            self.aruco_params.minMarkerPerimeterRate = 0.02  # Lower = detect smaller markers
            self.aruco_params.maxMarkerPerimeterRate = 4.0
            self.aruco_params.polygonalApproxAccuracyRate = 0.05
            self.aruco_params.minCornerDistanceRate = 0.05
            self.aruco_params.minDistanceToBorder = 1  # Allow markers near edge
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_params.adaptiveThreshWinSizeMin = 3
            self.aruco_params.adaptiveThreshWinSizeMax = 23
            self.aruco_params.adaptiveThreshWinSizeStep = 10
            self.aruco_params.minMarkerPerimeterRate = 0.02
            self.aruco_params.maxMarkerPerimeterRate = 4.0
            self.aruco_params.polygonalApproxAccuracyRate = 0.05
            self.aruco_params.minCornerDistanceRate = 0.05
            self.aruco_params.minDistanceToBorder = 1
        
        self.marker_size = 0.10  # 10cm marker - ADJUST to your actual size!
        
        # 5. Load camera calibration
        self.load_camera_calibration()

        self.pose_count = 0
        self.data_file = self.output_dir / "hand_eye_data.yaml"
        self.failed_poses = []

        # Wait for action server
        self.get_logger().info("Waiting for arm_controller action server...")
        if not self.joint_trajectory_action_client_.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available!")
            raise RuntimeError("Action server timeout")
        
        self.get_logger().info("✓ Action server connected!")
        
        # Wait for camera
        self.get_logger().info("Waiting for camera feed...")
        timeout = 10.0
        start_time = time.time()
        while self.latest_frame is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.latest_frame is None:
            self.get_logger().error("✗ No camera image received!")
            raise RuntimeError("Camera timeout")
        else:
            self.get_logger().info(f"✓ Camera feed received! Image size: {self.latest_frame.shape}")
            # Save first frame
            cv2.imwrite(str(self.output_dir / "initial_camera_view.jpg"), self.latest_frame)
            self.get_logger().info(f"  Saved initial view to: {self.output_dir / 'initial_camera_view.jpg'}")

    def load_camera_calibration(self):
        """Load camera calibration from file if available"""
        camera_params_file = "camera_params.yaml"
        if os.path.exists(camera_params_file):
            try:
                cv_file = cv2.FileStorage(camera_params_file, cv2.FILE_STORAGE_READ)
                self.mtx = cv_file.getNode('K').mat()
                self.dist = cv_file.getNode('D').mat()
                cv_file.release()
                self.camera_info_received = True
                self.get_logger().info(f"✓ Loaded camera calibration from {camera_params_file}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load camera params: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        """Get camera calibration from camera_info topic"""
        if not self.camera_info_received:
            self.mtx = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info("✓ Received camera calibration from camera_info")
            self.get_logger().info(f"  Focal length: fx={self.mtx[0,0]:.1f}, fy={self.mtx[1,1]:.1f}")
            self.get_logger().info(f"  Principal point: cx={self.mtx[0,2]:.1f}, cy={self.mtx[1,2]:.1f}")

    def image_callback(self, msg: Image):
        """Store latest camera frame"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def get_robot_pose(self, target_frame: str = 'camera_optical_frame'):
        """Get robot end-effector pose"""
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 
                target_frame, 
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            quat = [t.transform.rotation.x, t.transform.rotation.y, 
                   t.transform.rotation.z, t.transform.rotation.w]
            rot_matrix = R.from_quat(quat).as_matrix()
            
            self.get_logger().info(f"  Robot pose: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
            return rot_matrix.tolist(), pos
            
        except TransformException as e:
            self.get_logger().error(f"TF Lookup failed: {e}")
            if target_frame != 'wriste_roll':
                self.get_logger().warn("  Falling back to 'wriste_roll' frame")
                return self.get_robot_pose(target_frame='wriste_roll')
            return None, None

    def detect_aruco_marker(self, frame, pose_num):
        """Detect ArUco marker and estimate its pose"""
        if not self.camera_info_received:
            self.get_logger().error("✗ No camera calibration available!")
            return None, None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Detect markers
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        
        # Create debug visualization
        debug_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Add grid overlay to see field of view
        for i in range(0, w, 80):
            cv2.line(debug_frame, (i, 0), (i, h), (100, 100, 100), 1)
        for i in range(0, h, 60):
            cv2.line(debug_frame, (0, i), (w, i), (100, 100, 100), 1)
        
        # Add info
        cv2.putText(debug_frame, f"Pose {pose_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Size: {w}x{h}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)
            self.get_logger().info(f"✓ Detected {len(ids)} marker(s): ID={ids.flatten()}")
            
            # Estimate pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.mtx, self.dist
            )
            
            # Log and visualize each marker
            for i, (rvec, tvec, mid) in enumerate(zip(rvecs, tvecs, ids)):
                distance = np.linalg.norm(tvec[0])
                
                # Calculate marker size in pixels
                corner = corners[i][0]
                edge_lengths = []
                for j in range(4):
                    p1 = corner[j]
                    p2 = corner[(j+1)%4]
                    edge_length = np.linalg.norm(p2 - p1)
                    edge_lengths.append(edge_length)
                avg_pixel_size = np.mean(edge_lengths)
                
                self.get_logger().info(
                    f"  Marker {mid[0]}: Distance={distance:.3f}m, "
                    f"Size={avg_pixel_size:.1f}px, "
                    f"X={tvec[0][0]:.3f}, Y={tvec[0][1]:.3f}, Z={tvec[0][2]:.3f}"
                )
                
                # Draw axis
                try:
                    cv2.drawFrameAxes(debug_frame, self.mtx, self.dist, rvec, tvec, 0.05)
                except AttributeError:
                    pass
                
                # Add detailed text
                center = corner.mean(axis=0).astype(int)
                cv2.putText(debug_frame, f"ID:{mid[0]}", (center[0]-30, center[1]-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(debug_frame, f"D:{distance:.2f}m", (center[0]-30, center[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(debug_frame, f"{avg_pixel_size:.0f}px", (center[0]-30, center[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save images
            cv2.imwrite(str(self.output_dir / f"pose_{pose_num:02d}_DETECTED.jpg"), debug_frame)
            cv2.imwrite(str(self.output_dir / f"pose_{pose_num:02d}_gray.jpg"), gray)
            
            # Convert to matrix
            cam_R, _ = cv2.Rodrigues(rvecs[0])
            cam_T = tvecs[0][0].tolist()
            
            return cam_R.tolist(), cam_T
            
        else:
            self.get_logger().warn(f"✗ No markers detected! Rejected: {len(rejected)}")
            
            # Draw rejected candidates
            for rej_corners in rejected:
                cv2.polylines(debug_frame, [rej_corners.astype(int)], True, (0, 0, 255), 2)
            
            cv2.putText(debug_frame, "NO MARKERS", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(debug_frame, f"Rejected: {len(rejected)}", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save debug images
            cv2.imwrite(str(self.output_dir / f"pose_{pose_num:02d}_NO_MARKER.jpg"), debug_frame)
            cv2.imwrite(str(self.output_dir / f"pose_{pose_num:02d}_gray.jpg"), gray)
            
            self.get_logger().warn(f"  Check: {self.output_dir / f'pose_{pose_num:02d}_NO_MARKER.jpg'}")
            
            return None, None

    def execute_and_detect(self, positions: list, sec_from_start: int = 5):
        """Move robot and detect marker"""
        pose_num = self.pose_count + 1
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"Pose {pose_num}")
        self.get_logger().info(f"{'='*60}")
        
        # Movement
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = sec_from_start
        goal.trajectory.points.append(point)

        self.get_logger().info(f"Moving to position...")
        self.get_logger().info(f"  Joints (deg): {[round(math.degrees(p), 1) for p in positions]}")
        
        future = self.joint_trajectory_action_client_.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        if not future.result() or not future.result().accepted:
            self.get_logger().error("✗ Movement failed!")
            self.failed_poses.append(self.pose_count)
            self.pose_count += 1
            return
        
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        self.get_logger().info("✓ Movement completed")
        
        # Wait for stabilization
        self.get_logger().info("Stabilizing...")
        time.sleep(3.0)
        
        # Get fresh frames
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Detection
        if self.latest_frame is None:
            self.get_logger().error("✗ No camera frame!")
            self.failed_poses.append(self.pose_count)
            self.pose_count += 1
            return
        
        self.get_logger().info("Detecting ArUco marker...")
        cam_R, cam_T = self.detect_aruco_marker(self.latest_frame, pose_num)
        
        if cam_R is None:
            self.failed_poses.append(self.pose_count)
            self.pose_count += 1
            return
        
        # Get robot pose
        robot_R, robot_T = self.get_robot_pose(target_frame='camera_optical_frame')
        
        if robot_R is None:
            self.get_logger().error("✗ TF failed!")
            self.failed_poses.append(self.pose_count)
            self.pose_count += 1
            return
        
        # Save data
        sample = {
            f'sample_{self.pose_count}': {
                'robot_effector': {'R': robot_R, 'T': robot_T},
                'marker_camera': {'R': cam_R, 'T': cam_T}
            }
        }
        
        with open(self.data_file, 'a') as f:
            yaml.dump(sample, f, default_flow_style=False)
        
        self.get_logger().info(f"✓✓✓ Sample {self.pose_count} SAVED! ✓✓✓")
        self.pose_count += 1

    def print_summary(self):
        """Print calibration summary"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("CALIBRATION SUMMARY")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Total poses:       {self.pose_count}")
        successful = self.pose_count - len(self.failed_poses)
        self.get_logger().info(f"Successful:        {successful}")
        self.get_logger().info(f"Failed:            {len(self.failed_poses)}")
        if self.failed_poses:
            self.get_logger().info(f"Failed numbers:    {self.failed_poses}")
        self.get_logger().info(f"Output directory:  {self.output_dir}")
        self.get_logger().info(f"Data file:         {self.data_file}")
        self.get_logger().info("="*60)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = M6RCalibrationExecutor("m6r_calibrator")

        # Calibration poses - adjust to see marker from different angles
        poses_degrees = [
            [0.0, 30.0, 60.0, 0.0, 50.0, 0.0],
            [20.0, 35.0, 55.0, -50.0, 60.0, 0.0],
            [-20.0, 30.0, 50.0, 10.0, 60.0, -5.0],
            [0.0, 30.0, 55.0, 0.0, 70.0, 0.0],
            [15.0, 30.0, 75.0, -15.0, 45.0, 0.0],
        ]

        poses = [[math.radians(a) for a in p] for p in poses_degrees]

        node.get_logger().info(f"\nStarting calibration with {len(poses)} poses\n")
        
        for pose in poses:
            node.execute_and_detect(pose)
            
        node.print_summary()
        
    except KeyboardInterrupt:
        node.get_logger().info("\n✗ Interrupted")
    except Exception as e:
        node.get_logger().error(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()