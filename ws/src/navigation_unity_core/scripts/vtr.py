import rospy
from pfvtr.msg import DistancedTwist, MapRepeaterAction, MapRepeaterResult, MapRepeaterGoal, SensorsInput, SensorsOutput
from pfvtr.srv import StopRepeater
import actionlib
import numpy as np
from abc import ABC, abstractmethod
import rosbag
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Quaternion
import copy


class BaseVTR(ABC):
    """
    Simulator can use all VTR frameworks implementing this interface!
    """

    @abstractmethod
    def repeat_map(self, start=float, end=float, map_name=str):
        """
        Start repeating of map with map_name in folder ~/.ros/simulator_maps"
        """
        raise NotImplementedError

    @abstractmethod
    def is_finished(self) -> bool:
        """
        Check for simulator whether the traversal is finished
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Stop traversal and reset all states of navigation framework.
        It is used at the end of run or when failure is detected.
        """
        raise NotImplementedError


class PFVTR(BaseVTR):
    def __init__(self, image_pub):
        rospy.logwarn("Trying to instantiate PFCTR class")
        self.image_pub = image_pub
        
        self.map_hists = None  # histogram comparing live vs map images
        self.live_diff_hist = None  # curr live img vs prev live img 
        self.dists = None  # distances of map images 
        self.map_diff_hist = None  # consecutive map images comparison
        self.est_dist = 0.0
        self.finished = True
        self.target_dist = 9999999999
        self.map_start = False
        
        self.obs_sub = rospy.Subscriber("/pfvtr/matched_repr", SensorsInput, self._obs_callback, queue_size=1)
        self.distance_sub = rospy.Subscriber("/pfvtr/repeat/output_dist", SensorsOutput, self._dist_callback, queue_size=1)

        rospy.wait_for_service("pfvtr/stop_repeater")
        self.stop_pfvtr = rospy.ServiceProxy("pfvtr/stop_repeater", StopRepeater)
        rospy.logwarn("Service for stopping available")
        
        self.client = actionlib.SimpleActionClient("/pfvtr/repeater", MapRepeaterAction) # for VTR
        rospy.logwarn("PFVTR successfully connected!")
        
    def repeat_map(self, start=float, end=float, map_name=str):
        """
        Required method!
        """
        self.finished = False
        self.target_dist = end
        self.map_start = True
        curr_action = MapRepeaterGoal(startPos=start, endPos=end, traversals=0, nullCmd=True, imagePub=self.image_pub, useDist=True, mapName=map_name)
        self.client.send_goal(curr_action)
        return
    
    def is_finished(self):
        return self.finished

    def reset(self):
        self.stop_pfvtr(True)
        return
    
    def _obs_callback(self, msg):
        # fetch all possible observations
        self.map_hists = np.array(msg.map_histograms[0].values).reshape(msg.map_histograms[0].shape)
        self.live_diff_hist = np.array(msg.live_histograms[0].values).reshape(msg.live_histograms[0].shape)
        self.dists = np.array(msg.map_distances)
        self.map_diff_hist = np.array(msg.map_transitions[0].values).reshape(msg.map_transitions[0].shape)
        
    def _dist_callback(self, msg):
        # fetch estimated distance
        if self.map_start:
            rospy.sleep(3)
            self.map_start = False
            return
        self.est_dist = msg.output
        if self.est_dist >= self.target_dist - 0.5:
            self.finished = True


class InformedVTR(BaseVTR):
    def __init__(self):
        rospy.logwarn("Trying to instantiate PFCTR class")
        self.map_dir = "/home/zdeeno/.ros/simulator_maps/"
        self.finished = False

        self.repeating = False
        self.final_dist = None
        self.map_traj = None
        self.map_phis = None
        self.actions = None
        self.dists = None
        self.control_pub = rospy.Publisher("/bluetooth_teleop/cmd_vel", Twist, queue_size=5)
        self.pos_sub = rospy.Subscriber("/robot1/odometry", Odometry, self.odom_callback)

    def repeat_map(self, start=float, end=float, map_name=str):
        dists, odoms, actions = self.fetch_map(map_name)
        x = []
        y = []
        phis = []
        for i in range(len(odoms)):
            x.append(odoms[i].pose.pose.position.x)
            y.append(odoms[i].pose.pose.position.y)
            phis.append(euler_from_quaternion([odoms[i].pose.pose.orientation.x, odoms[i].pose.pose.orientation.y, odoms[i].pose.pose.orientation.z, odoms[i].pose.pose.orientation.w])[-1])
        self.map_phis = np.array(phis)
        self.map_traj = np.column_stack((x, y))
        self.dists = dists
        self.actions = actions
        self.final_dist = end
        self.finished = False
        self.target_dist = end
        self.repeating = True
        return

    def is_finished(self):
        return self.finished

    def reset(self):
        self.finished = True
        self.repeating = False
        self.control_pub.publish(Twist())
        return

    def fetch_map(self, name):
        bag = rosbag.Bag(self.map_dir + name + "/" + name + ".bag", "r")
        dists = []
        actions = []
        odoms = []
        last_odom = None
        for topic, msg, t in bag.read_messages(topics=["/recorded_actions", "/recorded_odometry"]):
            if topic == "/recorded_odometry":
                last_odom = msg
            if topic == "/recorded_actions" and last_odom is not None:
                dists.append(float(msg.distance))
                actions.append(msg.twist)
                odoms.append(last_odom)
        dists = np.array(dists)
        rospy.logwarn("Informed navigator fetched map: " + name)
        return dists, odoms, actions

    def smallest_angle_diff(self, angle1, angle2):
        diff = angle2 - angle1
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return np.where(diff < -np.pi, diff + 2 * np.pi, diff)

    def shortest_distance(self, x1, y1, phi, x2, y2):
        distance = (np.cos(phi) * (y1 - y2)) - (np.sin(phi) * (x1 - x2))
        return distance

    def odom_callback(self, msg):
        # get curr position
        if self.repeating:
            curr_x = msg.pose.pose.position.x
            curr_y = msg.pose.pose.position.y
            a, b, c = euler_from_quaternion(
                [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                 msg.pose.pose.orientation.w])
            curr_phi = c
            curr_pos = np.array([curr_x, curr_y])

            # find the nearest point in trajectory
            dists = np.sqrt(np.sum((curr_pos - self.map_traj) ** 2, axis=1))
            # dists = abs(self.shortest_distance(self.map_traj[:, 0], self.map_traj[:, 1], self.map_phis, curr_pos[0], curr_pos[1]))
            nearest_idx = np.argmin(dists)
            # rospy.logwarn("Nearest IDX: " + str(nearest_idx))

            if self.dists[nearest_idx] >= self.final_dist - 0.3:
                self.repeating = False
                self.finished = True

            # control policy
            target_pos = self.map_traj[nearest_idx]
            target_phi = self.map_phis[nearest_idx]

            displacement = self.shortest_distance(target_pos[0], target_pos[1], target_phi, curr_pos[0], curr_pos[1])
            phi_diff = self.smallest_angle_diff(target_phi, curr_phi)
            correction = -phi_diff + (0.1 * displacement)
            correction = np.sign(correction) * min(abs(correction), 0.1)

            control_command = copy.copy(self.actions[nearest_idx])
            control_command.linear.x = max(control_command.linear.x, 1.0)
            control_command.angular.z += correction

            # rospy.logwarn("Trying to control the robot with displacement " + str(displacement) + " and rotation diff " + str(phi_diff) + "\nCorrection is: " + str(correction))
            # rospy.logwarn(str(control_command))
            self.control_pub.publish(control_command)
        else:
            self.control_pub.publish(Twist())






