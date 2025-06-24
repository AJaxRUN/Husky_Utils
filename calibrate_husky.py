import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np
import yaml
import csv
import sys
import termios
import atexit

CALIBRATION_FILE = "husky_calibration.yaml"
CALIBRATION_CSV = "husky_calibration_data.csv"

# Store terminal settings
old_settings = termios.tcgetattr(sys.stdin.fileno())

def restore_terminal():
    """Restore terminal settings."""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

atexit.register(restore_terminal)

class HuskyCalibrator:
    def __init__(self):
        rospy.init_node('husky_calibrator', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback, queue_size=10)
        self.real_pose = np.array([0.0, 0.0, 0.0])  # (x, y, yaw)
        self.calibration_data = []
        self.offsets = {'x_offset': 0.0, 'y_offset': 0.0, 'yaw_offset': 0.0}

    def odom_callback(self, msg):
        """Store real robot's pose."""
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.real_pose = np.array([pos.x, pos.y, yaw])

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

    def perform_calibration(self):
        """Perform calibration of robot axes and sensors."""
        rospy.loginfo("Starting calibration sequence...")
        self.calibration_data = []
        initial_pose = self.real_pose.copy()

        calibration_sequence = [
            (0.2, 0.0, 1.0),  # Forward
            (-0.2, 0.0, 1.0), # Backward
            (0.0, 0.5, 1.0),  # Turn left
            (0.0, -0.5, 1.0), # Turn right
        ]

        for linear_x, angular_z, duration in calibration_sequence:
            start_pose = self.real_pose.copy()
            self.send_cmd(linear_x, angular_z, duration)
            end_pose = self.real_pose.copy()
            self.calibration_data.append({
                'cmd': (linear_x, angular_z),
                'start_pose': start_pose,
                'end_pose': end_pose,
                'timestamp': rospy.Time.now().to_sec()
            })
            rospy.sleep(0.5)

        # Calculate offsets based on drift
        final_pose = self.real_pose
        self.offsets['x_offset'] = float(final_pose[0] - initial_pose[0])  # Convert to Python float
        self.offsets['y_offset'] = float(final_pose[1] - initial_pose[1])  # Convert to Python float
        self.offsets['yaw_offset'] = float(final_pose[2] - initial_pose[2])  # Convert to Python float

        # Validate calibration
        pose_drift = np.linalg.norm([self.offsets['x_offset'], self.offsets['y_offset']])
        yaw_drift = abs(self.offsets['yaw_offset'])
        rospy.loginfo(f"Calibration results: Position drift: {pose_drift:.4f}m, Yaw drift: {yaw_drift:.4f}rad")

        if pose_drift < 0.1 and yaw_drift < 0.1:
            rospy.loginfo("Calibration successful.")
            self.save_calibration()
            self.save_calibration_data()
            return True
        else:
            rospy.logwarn("Calibration failed due to excessive drift.")
            return False

    def save_calibration(self):
        """Save calibration offsets to YAML file."""
        with open(CALIBRATION_FILE, 'w') as f:
            yaml.dump(self.offsets, f, default_flow_style=False)
        rospy.loginfo(f"Calibration offsets saved to {CALIBRATION_FILE}")

    def save_calibration_data(self):
        """Save calibration data to CSV file."""
        with open(CALIBRATION_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cmd_linear_x', 'cmd_angular_z', 
                            'start_x', 'start_y', 'start_yaw',
                            'end_x', 'end_y', 'end_yaw'])
            for data in self.calibration_data:
                writer.writerow([
                    data['timestamp'],
                    data['cmd'][0], data['cmd'][1],
                    float(data['start_pose'][0]), float(data['start_pose'][1]), float(data['start_pose'][2]),
                    float(data['end_pose'][0]), float(data['end_pose'][1]), float(data['end_pose'][2])
                ])
        rospy.loginfo(f"Calibration data saved to {CALIBRATION_CSV}")

    def run(self):
        """Run calibration process."""
        rospy.sleep(1.0)  # Wait for subscribers to connect
        success = self.perform_calibration()
        if not success:
            rospy.logwarn("Retrying calibration...")
            success = self.perform_calibration()
            if not success:
                rospy.logerr("Calibration failed after retry. Exiting.")
                sys.exit(1)
        rospy.signal_shutdown("Calibration complete")

if __name__ == '__main__':
    calibrator = HuskyCalibrator()
    calibrator.run()