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

class SmartBaseCalibrator(Node):
    def __init__(self):
        super().__init__('smart_base_calibrator')
        self.bridge = CvBridge()
        
        # =========================================================================
        # 1. CONFIGURATION
        # =========================================================================
        self.output_dir = "smart_calib_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        # WORLD -> TAG (Where is the tag in Gazebo?)
        self.T_world_tag = np.eye(4)
        self.T_world_tag[:3, 3] = [1.0, 0.0, 0.0] 
        self.T_world_tag[:3, :3] = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()

        # EXPECTED ID & SIZE
        self.MARKER_ID = 0
        self.MARKER_SIZE = 0.10

        # HAND-EYE (EE -> Camera)
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
        
        # State
        self.captured_transforms = []
        self.current_calc_base = None 
        self.invert_image = False # Toggle with 'i'
        
        # --- SMART DICTIONARY SETUP ---
        self.available_dicts = {
            "4x4_50": cv2.aruco.DICT_4X4_50,
            "4x4_100": cv2.aruco.DICT_4X4_100,
            "5x5_50": cv2.aruco.DICT_5X5_50,
            "6x6_50": cv2.aruco.DICT_6X6_50,
            "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        }
        self.current_dict_name = "4x4_50" # Start here
        self.active_dict = cv2.aruco.getPredefinedDictionary(self.available_dicts[self.current_dict_name])

        # Robust Parameters for Gazebo
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10

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

    def process_and_show(self):
        if self.latest_frame is None or self.mtx is None: return None

        self.current_calc_base = None 
        vis_img = self.latest_frame.copy()
        
        # Toggle Inversion
        if self.invert_image:
            vis_img = cv2.bitwise_not(vis_img)
            
        gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Try Current Dictionary
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.active_dict, parameters=self.aruco_params)

        # 2. If Failed, Auto-Cycle Dictionaries
        if ids is None or len(ids) == 0:
            for name, dict_id in self.available_dicts.items():
                temp_dict = cv2.aruco.getPredefinedDictionary(dict_id)
                c, i, _ = cv2.aruco.detectMarkers(gray, temp_dict, parameters=self.aruco_params)
                if i is not None and len(i) > 0:
                    # Found a better dictionary! Switch to it.
                    print(f"Auto-Switched Dictionary to: {name}")
                    self.current_dict_name = name
                    self.active_dict = temp_dict
                    corners, ids = c, i
                    break

        is_tracking = False

        if ids is not None and len(ids) > 0:
            ids = ids.flatten()
            if self.MARKER_ID in ids:
                idx = np.where(ids == self.MARKER_ID)[0][0]
                
                # Draw
                cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)

                # Solve PnP
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[idx], self.MARKER_SIZE, self.mtx, self.dist
                )
                rvec, tvec = rvecs[0][0], tvecs[0][0]
                cv2.drawFrameAxes(vis_img, self.mtx, self.dist, rvec, tvec, 0.1)
                is_tracking = True

                # Calculate Base
                T_base_ee = self.get_tf_pose()
                if T_base_ee is not None:
                    T_cam_tag = np.eye(4)
                    T_cam_tag[:3, :3], _ = cv2.Rodrigues(rvec)
                    T_cam_tag[:3, 3] = tvec

                    T_world_base = (
                        self.T_world_tag @ 
                        np.linalg.inv(T_cam_tag) @ 
                        np.linalg.inv(self.T_ee_cam) @ 
                        np.linalg.inv(T_base_ee)
                    )
                    self.current_calc_base = T_world_base
                    
                    # Display Info
                    xyz = T_world_base[:3, 3]
                    cv2.putText(vis_img, f"BASE X: {xyz[0]:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(vis_img, f"BASE Y: {xyz[1]:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(vis_img, f"BASE Z: {xyz[2]:.3f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(vis_img, f"Dict: {self.current_dict_name}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if not is_tracking:
            status = "SCANNING..."
            if self.invert_image: status += " (INVERTED)"
            cv2.putText(vis_img, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(vis_img, f"Trying: {self.current_dict_name}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        return vis_img

    def save_snapshot(self):
        if self.current_calc_base is not None:
            self.captured_transforms.append(self.current_calc_base)
            print(f"Captured Sample {len(self.captured_transforms)}")
            return True
        return False

    def compute_final_average(self):
        if not self.captured_transforms: return
        avg_trans = np.mean([T[:3, 3] for T in self.captured_transforms], axis=0)
        rots = R.from_matrix([T[:3, :3] for T in self.captured_transforms])
        avg_rot = rots.mean().as_matrix()
        rpy = R.from_matrix(avg_rot).as_euler('xyz', degrees=True)
        
        print("\n" + "="*50)
        print("FINAL AVERAGED BASE POSE")
        print(f"X: {avg_trans[0]:.4f} m | Y: {avg_trans[1]:.4f} m | Z: {avg_trans[2]:.4f} m")
        print(f"RPY: {np.round(rpy, 3)}")
        print("="*50)

# GUI CLASS
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
    node = SmartBaseCalibrator()
    gui = RobotControlGUI(node)

    print("\n--- SMART BASE CALIBRATION ---")
    print("1. Use sliders to move robot.")
    print("2. If no tag: Press 'i' to invert image.")
    print("3. Press 'q' to capture sample.")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            gui.update()
            
            vis_frame = node.process_and_show()
            if vis_frame is not None:
                cv2.imshow("Smart View", vis_frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    if node.save_snapshot():
                        cv2.putText(vis_frame, "SAVED!", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        cv2.imshow("Smart View", vis_frame)
                        cv2.waitKey(200)
                elif key == ord('i'):
                    node.invert_image = not node.invert_image
                    print(f"Invert Image: {node.invert_image}")
                elif key == 27: break
    except KeyboardInterrupt: pass
    finally:
        node.compute_final_average()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()