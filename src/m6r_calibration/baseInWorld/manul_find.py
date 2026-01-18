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
import os
import tkinter as tk
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformException
from datetime import datetime

class RealTimeCharuco(Node):
    def __init__(self):
        super().__init__('realtime_base_charuco')
        self.bridge = CvBridge()
        
        # =========================================================================
        # 1. CONFIGURATION
        # =========================================================================
        self.output_dir = "charuco_realtime_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        # WORLD -> BOARD Pose
        # Where is the board in your Gazebo world?
        self.T_world_board = np.eye(4)
        self.T_world_board[:3, 3] = [1.0, 0.0, 0.0] # 1m forward
        self.T_world_board[:3, :3] = R.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()

        # BOARD SPECS
        self.SQUARES_X = 5
        self.SQUARES_Y = 7
        self.SQUARE_LEN = 0.04
        self.MARKER_LEN = 0.03
        self.DICT_TYPE = cv2.aruco.DICT_4X4_50

        # Hand-Eye (EE -> Camera)
        self.T_ee_cam = np.eye(4)
        self.T_ee_cam[:3, 3] = [-0.8, 0.0252, 0.1855]
        self.T_ee_cam[:3, :3] = R.from_euler('xyz', [96, -56, 78], degrees=True).as_matrix()
        # =========================================================================

        # ROS Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)

        self.mtx = None
        self.dist = None
        self.latest_frame = None
        
        # Real-time state
        self.current_rvec = None
        self.current_tvec = None
        self.is_tracking = False
        self.collected_transforms = []

        # Charuco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.DICT_TYPE)
        try:
            self.board = cv2.aruco.CharucoBoard((self.SQUARES_X, self.SQUARES_Y), 
                                                self.SQUARE_LEN, self.MARKER_LEN, self.aruco_dict)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.board = cv2.aruco.CharucoBoard_create(self.SQUARES_X, self.SQUARES_Y, 
                                                       self.SQUARE_LEN, self.MARKER_LEN, self.aruco_dict)
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        # GUI State
        self.joint_targets = [0.0] * 6

    def info_callback(self, msg):
        if self.mtx is None:
            self.mtx = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d) if len(msg.d) else np.zeros(5)

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def send_joint_command(self):
        if not self.arm_client.wait_for_server(timeout_sec=0.1): return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        point = JointTrajectoryPoint()
        point.positions = [math.radians(x) for x in self.joint_targets]
        point.time_from_start.sec = 1
        goal.trajectory.points.append(point)
        self.arm_client.send_goal_async(goal)

    def get_tf_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'wriste_roll', rclpy.time.Time())
            pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            return T
        except TransformException: return None

    def process_frame(self):
        """Runs continuous detection and returns an annotated image"""
        if self.latest_frame is None or self.mtx is None:
            return None

        # Reset state
        self.is_tracking = False
        self.current_rvec = None
        self.current_tvec = None

        vis_img = self.latest_frame.copy()
        gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect Markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
            
            # 2. Interpolate Charuco
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board)

            if charuco_corners is not None and len(charuco_corners) > 4:
                cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids, (255, 0, 0))
                
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board, self.mtx, self.dist, None, None)

                if valid:
                    # Update State
                    self.is_tracking = True
                    self.current_rvec = rvec
                    self.current_tvec = tvec
                    
                    # Draw Axes
                    cv2.drawFrameAxes(vis_img, self.mtx, self.dist, rvec, tvec, 0.1)
                    
                    # Draw Green Box
                    cv2.putText(vis_img, "TRACKING", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    dist = np.linalg.norm(tvec)
                    cv2.putText(vis_img, f"Dist: {dist:.2f}m", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if not self.is_tracking:
            cv2.putText(vis_img, "NO BOARD", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return vis_img

    def save_measurement(self):
        """Called when user presses 'q'"""
        if not self.is_tracking:
            print("Cannot capture: Board not detected!")
            return

        # 1. Get Robot Pose
        T_base_ee = self.get_tf_pose()
        if T_base_ee is None:
            print("TF Error: Could not get robot pose.")
            return

        # 2. Calculate Base Pose
        # T_cam_board
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3], _ = cv2.Rodrigues(self.current_rvec)
        T_cam_board[:3, 3] = self.current_tvec.flatten()

        # Chain Calculation
        T_world_base = (
            self.T_world_board @ 
            np.linalg.inv(T_cam_board) @ 
            np.linalg.inv(self.T_ee_cam) @ 
            np.linalg.inv(T_base_ee)
        )
        
        self.collected_transforms.append(T_world_base)
        
        # 3. Save Image for verification
        idx = len(self.collected_transforms)
        filename = f"{self.output_dir}/capture_{idx:02d}.jpg"
        # We need to re-draw the frame or grab current display? 
        # Ideally we save the one we just saw, but for simplicity, re-grab or pass logic.
        # Since it's realtime, we just grab the frame currently processed.
        
        xyz = T_world_base[:3, 3]
        print(f"[Capture {idx}] Base Estimate: X={xyz[0]:.3f}, Y={xyz[1]:.3f}, Z={xyz[2]:.3f}")
        return True

    def compute_final_average(self):
        if not self.collected_transforms: return
        
        avg_trans = np.mean([T[:3, 3] for T in self.collected_transforms], axis=0)
        rots = R.from_matrix([T[:3, :3] for T in self.collected_transforms])
        avg_rot = rots.mean().as_matrix()
        rpy = R.from_matrix(avg_rot).as_euler('xyz', degrees=True)
        
        print("\n" + "="*50)
        print("FINAL BASE POSE (CHARUCO)")
        print(f"X: {avg_trans[0]:.4f} m | Y: {avg_trans[1]:.4f} m | Z: {avg_trans[2]:.4f} m")
        print(f"RPY: {np.round(rpy, 2)}")
        print("="*50)

# GUI (Same as before)
class RobotControlGUI:
    def __init__(self, ros_node):
        self.node = ros_node
        self.root = tk.Tk()
        self.root.title("Joint Control")
        self.root.geometry("400x350")
        self.sliders = []
        limits = [(-180, 180), (-90, 90), (-150, 150), (-180, 180), (-100, 100), (-180, 180)]
        for i in range(6):
            frame = tk.Frame(self.root)
            frame.pack(fill='x', padx=10, pady=5)
            tk.Label(frame, text=f"Joint {i+1}").pack(side='left')
            s = tk.Scale(frame, from_=limits[i][0], to=limits[i][1], orient='horizontal', length=300)
            s.set(0)
            s.pack(side='right')
            s.bind("<ButtonRelease-1>", self.update_joints)
            self.sliders.append(s)

    def update_joints(self, event=None):
        vals = [s.get() for s in self.sliders]
        self.node.joint_targets = vals
        self.node.send_joint_command()

    def update(self):
        self.root.update_idletasks()
        self.root.update()

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeCharuco()
    gui = RobotControlGUI(node)

    print("\n--- REAL-TIME CALIBRATION ---")
    print("1. Use sliders to look at the Charuco board")
    print("2. Wait for green 'TRACKING' text")
    print("3. Press 'q' to Capture Pose")
    print("4. Press 'ESC' to Finish")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            gui.update()
            
            # Process and Display Vision
            vis_frame = node.process_frame()
            
            if vis_frame is not None:
                cv2.imshow("Real-Time Vision", vis_frame)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    if node.save_measurement():
                        # Flash "CAPTURED" on screen momentarily
                        cv2.putText(vis_frame, "CAPTURED!", (200, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                        cv2.imshow("Real-Time Vision", vis_frame)
                        cv2.waitKey(200) # Pause to show capture
                elif key == 27: # ESC
                    break
                    
    except KeyboardInterrupt: pass
    finally:
        node.compute_final_average()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()