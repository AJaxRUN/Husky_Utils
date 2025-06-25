import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from urdfpy import URDF
import pyrender
import trimesh
import atexit
import numpy as np
import matplotlib.pyplot as plt
import transforms3d.euler
from tf.transformations import euler_from_quaternion
import threading
import sys
import select
import csv
import tty
import termios
import yaml
import os

URDF_PATH = "husky.urdf"
Z_OFFSET = 0.165
CALIBRATION_FILE = "husky_calibration.yaml"
SELECTED_SEQUENCE = "default"
MOTION_SEQUENCES = dict({ 
    "default" :[
        (0.5, 0.0, 2.0),  # forward 2s
        (0.0, 1.0, 1.0),  # turn left 1s
        (0.5, 0.0, 1.0),  # forward 1s
        (0.0, -1.0, 1.0), # turn right 1s
        (0.5, 0.0, 1.0),  # forward 1s
    ],
    "straight" :[
        (0.5, 0.0, 3.0),  # forward 3s
    ]
})


# Store terminal settings
old_settings = termios.tcgetattr(sys.stdin.fileno())
def restore_terminal():
    """Restore terminal settings."""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

atexit.register(restore_terminal)

class HuskySim:
    def __init__(self, urdf_path):

        # Data storage
        self.cmd_vel = (0.0, 0.0)  # (linear_x, angular_z)
        self.discrepancies = []  # Store (x_error, y_error, timestamp)
        self.collect_data = False  # Flag to control data collection
        self.total_predicted_distance = 0.0
        self.husky_positions = []
        self.husky_distance = 0.0

        # Initialize calibration attributes
        self.offsets = {'x_offset': 0.0, 'y_offset': 0.0, 'yaw_offset': 0.0}
        self.is_calibrated = False
        self.load_calibration()  # Load calibration offsets
        rospy.init_node('husky_sim', anonymous=True)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_callback, queue_size=10)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback, queue_size=10)

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.robot = URDF.load(urdf_path)
        self.scene = pyrender.Scene()
        self.link_nodes = {}

        # Sync predicted pose with real pose at start
        self.pred_pose = self.huksy_pose.copy()

        self.add_ground()
        self.load_robot_meshes()
        
        # Initialize real-time plotting with enhanced visualization
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.ax1.set_title("Husky vs Predicted Trajectory")
        self.ax1.set_xlabel("X (m)")
        self.ax1.set_ylabel("Y (m)")
        self.ax1.grid(True)
        # Adjusted limits for better visibility
        self.ax1.set_xlim(-2, 2)
        self.ax1.set_ylim(-2, 2)
        self.real_line, = self.ax1.plot([], [], 'ro-', label="Real")
        self.pred_line, = self.ax1.plot([], [], 'b--', label="Predicted", alpha=0.7)
        self.ax2.grid(True)
        self.ax1.legend()

        self.ax2.set_title("Euclidean Distance Error (Total Position Error)")
        self.ax2.set_xlabel("Number of Samples (N)")
        self.ax2.set_ylabel("Error")
        self.ax2.grid(True)
        self.dist_err_line, = self.ax2.plot([], [], 'r-', label="Euclidean Distance Error")
        self.ax2.legend()

        # Start pyrender viewer in a separate thread
        self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True)
        # Start keyboard listener in a separate thread
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    def load_calibration(self):
        """Load calibration offsets from YAML file."""
        if not os.path.exists(CALIBRATION_FILE):
            rospy.logwarn(f"Calibration file {CALIBRATION_FILE} not found. Using zero offsets.")
            self.is_calibrated = False
            return
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                loaded_offsets = yaml.safe_load(f)
            if loaded_offsets and all(key in loaded_offsets for key in ['x_offset', 'y_offset', 'yaw_offset']):
                self.offsets = loaded_offsets
                self.is_calibrated = True
                rospy.loginfo(f"Loaded calibration offsets: {self.offsets}")
            else:
                rospy.logwarn("Invalid calibration file format. Using zero offsets.")
                self.is_calibrated = False
        except Exception as e:
            rospy.logwarn(f"Failed to load calibration file: {e}. Using zero offsets.")
            self.is_calibrated = False

    def add_ground(self):
        ground = trimesh.creation.box(extents=[10.0, 10.0, 0.01])
        ground.visual.face_colors = [180, 180, 180, 255]
        ground_pose = np.eye(4)
        ground_pose[2, 3] = -0.005
        ground.visual.material = None
        ground_mesh = pyrender.Mesh.from_trimesh(ground, smooth=False)
        self.scene.add(ground_mesh, pose=ground_pose)

    def load_robot_meshes(self):
        for link in self.robot.links:
            if not link.visuals:
                continue
            visual = link.visuals[0]
            if not hasattr(visual.geometry, 'mesh'):
                continue
            try:
                mesh = trimesh.load(visual.geometry.mesh.filename, force='mesh')
            except Exception as e:
                print(f"Failed to load {visual.geometry.mesh.filename}: {e}")
                continue

            mesh.apply_transform(visual.origin)
            pm = pyrender.Mesh.from_trimesh(mesh)
            node = pyrender.Node(mesh=pm, matrix=np.eye(4), name=link.name)
            self.link_nodes[link.name] = node
            self.scene.add_node(node)

    def cmd_callback(self, msg):
        """Store latest /cmd_vel."""
        self.cmd_vel = (msg.linear.x, msg.angular.z)
        rospy.loginfo(f"Buffered /cmd_vel: v={msg.linear.x:.2f}, omega={msg.angular.z:.2f}")

    def odom_callback(self, msg):
        """Store real robot's pose with calibration offsets."""
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        if self.is_calibrated:
            self.huksy_pose = np.array([
                pos.x - self.offsets['x_offset'],
                pos.y - self.offsets['y_offset'],
                yaw - self.offsets['yaw_offset']
            ])
        else:
            self.huksy_pose = np.array([pos.x, pos.y, yaw])
            rospy.logwarn_once("No calibration applied: Using raw odometry data.")
        
        if self.collect_data:
            if len(self.husky_positions) > 0:
                prev_pose = self.husky_positions[-1]
                dx = self.huksy_pose[0] - prev_pose[0]
                dy = self.huksy_pose[1] - prev_pose[1]
                dist = np.sqrt(dx**2 + dy**2)
                self.husky_distance += dist

        self.husky_positions.append(self.huksy_pose)

    def update_pose(self, dt):
        """Update predicted pose using linear kinematic equations without feedback."""
        linear_x, angular_z = self.cmd_vel
        x, y, yaw = self.pred_pose
        yaw_new = yaw + angular_z * dt
        x_new = x + linear_x * np.cos(yaw) * dt
        y_new = y + linear_x * np.sin(yaw) * dt
        self.pred_pose = np.array([x_new, y_new, yaw_new])
        # Store distance traveled for each step
        if self.collect_data:
            dist = np.sqrt((x_new - x)**2 + (y_new - y)**2)
            self.total_predicted_distance += dist

    def update_scene(self):
        """Update robot visualization in pyrender using husky's pose."""
        transforms = self.robot.link_fk()
        base_tf = np.eye(4)
        base_tf[:3, :3] = transforms3d.euler.euler2mat(0, 0, self.huksy_pose[2])  # Use huksy_pose
        base_tf[:3, 3] = [self.huksy_pose[0], self.huksy_pose[1], Z_OFFSET]  # Use huksy_pose
        for link_name, node in self.link_nodes.items():
            link = self.robot.link_map[link_name]
            local_tf = transforms.get(link, np.eye(4))
            self.scene.set_pose(node, pose=base_tf @ local_tf)

    def update_plot(self):
        """Update real-time discrepancy plot with difference visualization."""
        if not self.collect_data:
            return
        x_err = self.huksy_pose[0] - self.pred_pose[0]
        y_err = self.huksy_pose[1] - self.pred_pose[1]
        timestamp = rospy.Time.now().to_sec()
        self.discrepancies.append((x_err, y_err, timestamp, self.huksy_pose[0], self.huksy_pose[1], self.pred_pose[0], self.pred_pose[1]))

        # Update trajectory plot
        self.real_line.set_data(np.append(self.real_line.get_xdata(), self.huksy_pose[0]),
                               np.append(self.real_line.get_ydata(), self.huksy_pose[1]))
        self.pred_line.set_data(np.append(self.pred_line.get_xdata(), self.pred_pose[0]),
                               np.append(self.pred_line.get_ydata(), self.pred_pose[1]))
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Update error plot
        data_np = np.array(self.discrepancies)
        errs = np.sqrt(data_np[:, 0]**2 + data_np[:, 1]**2)
        samples = np.arange(errs.shape[0])
        self.dist_err_line.set_data(samples, errs)
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_data(self):
        """Save collected data to a CSV file."""
        csv_path = "husky_data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'huksy_x', 'huksy_y', 'pred_x', 'pred_y', 'x_error', 'y_error'])
            for x_err, y_err, timestamp, huksy_x, huksy_y, pred_x, pred_y in self.discrepancies:
                writer.writerow([timestamp, huksy_x, huksy_y, pred_x, pred_y, x_err, y_err])
        rospy.loginfo(f"Data saved to {csv_path}")

    def send_cmd(self, linear_x, angular_z, duration):
        """Publish velocity command for specified duration."""
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        end_time = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(10)
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def execute_motion_sequence(self):
        """Execute predefined motion and calculate total distance."""
        rospy.loginfo("Starting motion")
        self.collect_data = True
        self.discrepancies = []  # Clear previous data
        self.real_line.set_data([], [])  # Clear plots
        self.pred_line.set_data([], [])
        self.dist_err_line.set_data([], [])
        self.total_predicted_distance = 0  # Clear distance data
        sequence = MOTION_SEQUENCES[SELECTED_SEQUENCE]
        
        for linear_x, angular_z, duration in sequence:
            if rospy.is_shutdown():
                break
            self.send_cmd(linear_x, angular_z, duration)
        self.collect_data = False
        self.save_data()

        rospy.loginfo(f"Estimated total distance traveled by predicted trajectory: {self.total_predicted_distance:.2f} m")
        rospy.loginfo(f"Actual total distance traveled by Husky (odometry): {self.husky_distance:.2f} m")

    def keyboard_listener(self):
        """Listen for 'c' key to trigger motion sequence."""
        print("Press 'c' to execute motion sequence, 'q' to quit...")
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 'c':
                        self.execute_motion_sequence()
                    elif key.lower() == 'q':
                        self.save_data()
                        rospy.signal_shutdown("User requested shutdown")
                        break
        except Exception as e:
            rospy.logerr(f"Keyboard listener error: {e}")
        finally:
            restore_terminal()

    def run(self):
        rate = rospy.Rate(60)
        last_time = rospy.Time.now()
        try:
            while not rospy.is_shutdown():
                current_time = rospy.Time.now()
                dt = (current_time - last_time).to_sec()
                last_time = current_time

                self.update_pose(dt)
                self.update_scene()
                self.update_plot()
                rate.sleep()
        except Exception as e:
            rospy.logerr(f"Exception in main loop: {e}")
        finally:
            self.save_data()
            restore_terminal()
            rospy.signal_shutdown("Simulation ended")

if __name__ == '__main__':
    sim = HuskySim(URDF_PATH)
    sim.run()