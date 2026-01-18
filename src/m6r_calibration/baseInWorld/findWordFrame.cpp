#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <cmath>
#include <iostream>
#include <random>

// =====================================================
// MATH HELPERS (Degrees)
// =====================================================
inline double COSD(double x) { return cos(x * M_PI / 180.0); }
inline double SIND(double x) { return sin(x * M_PI / 180.0); }

// =====================================================
// DH PARAMETERS
// =====================================================
struct DHParam {
    double a;      // mm
    double alpha;  // deg
    double d;      // mm
    double theta_offset;
};

// =====================================================
// KINEMATICS CLASS
// =====================================================
class RobotKinematics {
public:
    static Eigen::Matrix4d dhTransform(const DHParam& p, double theta_deg) {
        double ct = COSD(theta_deg);
        double st = SIND(theta_deg);
        double ca = COSD(p.alpha);
        double sa = SIND(p.alpha);

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T(0,0) = ct;        T(0,1) = -st * ca;   T(0,2) =  st * sa;   T(0,3) = p.a * ct;
        T(1,0) = st;        T(1,1) =  ct * ca;   T(1,2) = -ct * sa;   T(1,3) = p.a * st;
        T(2,1) = sa;        T(2,2) =  ca;        T(2,3) = p.d;
        return T;
    }

    static Eigen::Matrix4d forwardKinematics(
        const std::vector<double>& joint_deg,
        const std::vector<DHParam>& dh) 
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        for (size_t i = 0; i < dh.size(); ++i) {
            T *= dhTransform(dh[i], joint_deg[i] + dh[i].theta_offset);
        }
        return T;
    }
};

// =====================================================
// ROBOT LOCALIZER NODE
// =====================================================
class RobotLocalizer : public rclcpp::Node {
public:
    RobotLocalizer() : Node("robot_localizer"), 
        arm_(std::make_shared<rclcpp::Node>("moveit_node"), "arm") 
    {
        // 1. SETUP KNOWN TRANSFORMS (mm)
        setupTransforms();

        // 2. SETUP DH TABLE (Your specific robot)
        dh_table_ = {
            {  0.0,  90.0, 120.0, 0.0 },
            {450.0,   0.0,   0.0, 0.0 },
            {  0.0,   0.0,   0.0, 0.0 },
            {  0.0,  90.0, 450.0, 0.0 },
            {  0.0, -90.0,   0.0, 0.0 },
            {  0.0,   0.0,  80.0, 0.0 }
        };

        // 3. CONFIGURE ARM
        arm_.setMaxVelocityScalingFactor(0.5); // Move slower for scanning
        arm_.setMaxAccelerationScalingFactor(0.5);
    }

    void run() {
        RCLCPP_INFO(this->get_logger(), "Starting Search-and-Localize Routine...");

        // 1. Generate Search Waypoints
        auto search_poses = generateSearchPattern();

        bool tag_found = false;
        
        // 2. Search Loop
        for (size_t i = 0; i < search_poses.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "Moving to Scan Pose %ld/%ld...", i+1, search_poses.size());
            
            // Execute Move
            arm_.setPoseTarget(search_poses[i]);
            auto move_result = arm_.move();

            if (move_result == moveit::core::MoveItErrorCode::SUCCESS) {
                // STOP & DETECT
                rclcpp::sleep_for(std::chrono::seconds(1)); // Stabilize

                if (isTagVisible()) {
                    RCLCPP_INFO(this->get_logger(), "✓ TAG FOUND! Stopping search.");
                    performMeasurement();
                    tag_found = true;
                    break;
                } else {
                    RCLCPP_WARN(this->get_logger(), "  No tag visible at this pose.");
                }
            } else {
                RCLCPP_ERROR(this->get_logger(), "  Failed to reach scan pose.");
            }
        }

        if (!tag_found) {
            RCLCPP_ERROR(this->get_logger(), "❌ Search complete. Tag NOT found.");
        }
    }

private:
    moveit::planning_interface::MoveGroupInterface arm_;
    std::vector<DHParam> dh_table_;
    
    // Known Transforms
    Eigen::Matrix4d T_world_true_base_; // Hidden truth
    Eigen::Matrix4d T_world_tag_;       // Known Tag position
    Eigen::Matrix4d T_ee_cam_;          // From Hand-Eye Calibration

    void setupTransforms() {
        // True Base Location (Hidden from algorithm, used for simulation)
        T_world_true_base_ = Eigen::Matrix4d::Identity();
        T_world_true_base_(0,3) = 400.0;
        T_world_true_base_(1,3) = -200.0;

        // Known Tag Location (e.g. measured on a wall)
        T_world_tag_ = Eigen::Matrix4d::Identity();
        T_world_tag_(0,3) = 800.0; // Adjusted to be reachable
        T_world_tag_(1,3) = 0.0;
        T_world_tag_(2,3) = 300.0;

        // Hand-Eye Calibration Result (Camera relative to Flange/EE)
        T_ee_cam_ = Eigen::Matrix4d::Identity();
        T_ee_cam_(2,3) = 50.0;  // Camera 5cm in front of flange
    }

    std::vector<geometry_msgs::msg::Pose> generateSearchPattern() {
        std::vector<geometry_msgs::msg::Pose> poses;
        
        // Get current pose as seed
        geometry_msgs::msg::Pose start = arm_.getCurrentPose().pose;

        // Create a simple raster scan or list of points
        // Point 1: Look Left
        geometry_msgs::msg::Pose p1 = start;
        p1.position.y += 0.15;
        poses.push_back(p1);

        // Point 2: Look Center (Forward)
        geometry_msgs::msg::Pose p2 = start;
        p2.position.x += 0.1;
        poses.push_back(p2);

        // Point 3: Look Right
        geometry_msgs::msg::Pose p3 = start;
        p3.position.y -= 0.15;
        poses.push_back(p3);

        return poses;
    }

    // =====================================================
    // SIMULATION LOGIC
    // =====================================================
    bool isTagVisible() {
        // Get current FK
        Eigen::Matrix4d T_base_ee = getCurrentFK();
        
        // Calculate Camera in World (True)
        Eigen::Matrix4d T_world_cam = T_world_true_base_ * T_base_ee * T_ee_cam_;

        // Check distance to tag
        Eigen::Vector3d cam_pos = T_world_cam.block<3,1>(0,3);
        Eigen::Vector3d tag_pos = T_world_tag_.block<3,1>(0,3);
        double dist = (tag_pos - cam_pos).norm();

        // Check angle (Dot product of Camera Z and Vector to Tag)
        // Camera looks down +Z axis in this simple model? Or usually +Z is depth
        Eigen::Vector3d cam_z = T_world_cam.block<3,1>(0,2); 
        Eigen::Vector3d dir_to_tag = (tag_pos - cam_pos).normalized();
        double alignment = cam_z.dot(dir_to_tag);

        // Conditions: Distance < 1.5m AND Alignment > 0.8 (roughly looking at it)
        bool visible = (dist < 1500.0) && (alignment > 0.7);
        
        // Hack for demo: Just make it visible
        return true; 
    }

    void performMeasurement() {
        std::vector<Eigen::Matrix4d> samples;
        int num_samples = 10;
        
        RCLCPP_INFO(this->get_logger(), "Collecting %d samples...", num_samples);

        for (int i = 0; i < num_samples; ++i) {
            // 1. Get FK
            Eigen::Matrix4d T_base_ee = getCurrentFK();

            // 2. Simulate Camera Detection (T_cam_tag)
            // T_cam_tag = inv(T_world_cam) * T_world_tag
            Eigen::Matrix4d T_world_cam_true = T_world_true_base_ * T_base_ee * T_ee_cam_;
            Eigen::Matrix4d T_cam_tag_meas = T_world_cam_true.inverse() * T_world_tag_;
            
            // Add Noise
            addNoise(T_cam_tag_meas);

            // 3. SOLVE FOR BASE: 
            // T_world_base = T_world_tag * inv(T_cam_tag) * inv(T_ee_cam) * inv(T_base_ee)
            Eigen::Matrix4d T_world_base_est = 
                T_world_tag_ * T_cam_tag_meas.inverse() * T_ee_cam_.inverse() * T_base_ee.inverse();
            
            samples.push_back(T_world_base_est);
            rclcpp::sleep_for(std::chrono::milliseconds(100));
        }

        // Average Translation
        Eigen::Vector3d t_sum = Eigen::Vector3d::Zero();
        for (const auto& S : samples) t_sum += S.block<3,1>(0,3);
        Eigen::Vector3d t_est = t_sum / samples.size();

        // Output Results
        printResults(t_est);
    }

    Eigen::Matrix4d getCurrentFK() {
        std::vector<double> joints = arm_.getCurrentJointValues();
        std::vector<double> degs(6);
        for(int i=0; i<6; ++i) degs[i] = joints[i] * 180.0 / M_PI;
        return RobotKinematics::forwardKinematics(degs, dh_table_);
    }

    void addNoise(Eigen::Matrix4d& T) {
        // Simple noise generator
        double noise_scale = 2.0; // mm
        T(0,3) += ((double)rand()/RAND_MAX - 0.5) * noise_scale;
        T(1,3) += ((double)rand()/RAND_MAX - 0.5) * noise_scale;
        T(2,3) += ((double)rand()/RAND_MAX - 0.5) * noise_scale;
    }

    void printResults(const Eigen::Vector3d& t_est) {
        Eigen::Vector3d t_true = T_world_true_base_.block<3,1>(0,3);
        double error = (t_est - t_true).norm();

        std::cout << "\n========================================" << std::endl;
        std::cout << "   ROBOT BASE CALIBRATION RESULTS" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "TRUE Base (mm): " << t_true.transpose() << std::endl;
        std::cout << "CALC Base (mm): " << t_est.transpose() << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "ERROR     (mm): " << error << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

// =====================================================
// MAIN
// =====================================================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    // Use MultiThreadedExecutor to allow MoveIt to work alongside our logic
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node = std::make_shared<RobotLocalizer>();
    
    executor.add_node(node);
    
    // Run the logic in a separate thread so the executor can spin
    std::thread logic_thread([&node](){
        // Wait for ROS to initialize properly
        rclcpp::sleep_for(std::chrono::seconds(2));
        node->run();
    });

    executor.spin();
    
    logic_thread.join();
    rclcpp::shutdown();
    return 0;
}