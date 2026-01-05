# MoveIt2 Setup Instructions

## ⚙️ Generation

If you encounter a **MoveIt2 Setup Assistant crash issue**, try downloading a **lower version of RViz2**.  
For details, see the issue discussion in the `ur10e_description` repository.

To launch the Setup Assistant, run:
```bash
ros2 launch moveit_setup_assistant setup_assistant.launch.py
```

---

## 🛠️ Modification

After generating the configuration, make the following changes:

1. **Convert integers to floats** in `joint_limits.yaml` to avoid type errors.

2. In `moveit_controllers.yaml`, add the following lines **under your controller definition** to ensure MoveIt2 knows where to connect the controller:
   ```yaml
   action_ns: follow_joint_trajectory
   default: true
   ```
3. The **ompl_planning.yaml** is manually added after creating the config package to support Path Planning
