#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState

def callback(msg):
    print(f"[{rospy.get_time():.2f}] Joints: {msg.name}")

if __name__ == '__main__':
    rospy.init_node('joint_debug_sub', anonymous=True)
    rospy.Subscriber("/joint_states", JointState, callback, queue_size=1, tcp_nodelay=True)
    rospy.loginfo("✅ Listening to /joint_states...")
    rospy.spin()
