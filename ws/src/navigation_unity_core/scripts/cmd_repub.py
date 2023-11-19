#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped

def callback(cmd_vel):
    pub.publish(cmd_vel)

if __name__ == '__main__':
    rospy.init_node('cmd_vel_to_twist_stamped_node')

    # Subscribe to the /bluetooth_teleop/cmd_vel topic with the Twist message
    rospy.Subscriber('/bluetooth_teleop/cmd_vel', Twist, callback)

    # Publish to the /robot1/velocity_reference topic with the TwistStamped message
    pub = rospy.Publisher('/robot1/velocity_reference', Twist, queue_size=10)

    rospy.spin()

