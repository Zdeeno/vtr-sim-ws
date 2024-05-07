#!/usr/bin/env python

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, Twist

class TFtoOdometryConverter:
    def __init__(self):
        rospy.init_node('tf_to_odometry_converter')

        # Initialize a TF listener
        self.tf_listener = tf.TransformListener()

        # Publisher for odometry messages
        self.odom_pub = rospy.Publisher('/robot1/odometry', Odometry, queue_size=10)

        # Run the conversion loop
        self.convert_tf_to_odometry()

    def convert_tf_to_odometry(self):
        rate = rospy.Rate(60)  # Hz
        while not rospy.is_shutdown():
            try:
                # Lookup the transform between "/world_origin" and "/robot1/chassis_link"
                (trans, rot) = self.tf_listener.lookupTransform('world_origin', 'robot1/chassis_link', rospy.Time(0))
                tf_timestamp = self.tf_listener.getLatestCommonTime('world_origin', 'robot1/chassis_link')
                # Create and populate the odometry message
                odom_msg = Odometry()
                odom_msg.header.stamp = tf_timestamp
                odom_msg.header.frame_id = "map"
                odom_msg.child_frame_id = "/robot1/chassis_link"
                odom_msg.pose.pose = Pose(Point(*trans), Quaternion(*rot))
                odom_msg.twist.twist = Twist()

                # Publish the odometry message
                self.odom_pub.publish(odom_msg)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("TF lookup failed. Retrying...")

            rate.sleep()

if __name__ == '__main__':
    try:
        TFtoOdometryConverter()
    except rospy.ROSInterruptException:
        pass

