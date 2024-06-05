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
from sensor_input import Processing
from data_fetching import DataFetching
import torch as t
from nn_model import FeedForward2, TransformerModel, TransformerObserver, Actor, Critic, SimpleObserver
from vtr import BaseVTR
from torchrl.envs import EnvBase
from torchrl.data import UnboundedContinuousTensorSpec, BoundedTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec
from simulator import Simulator
import os
import time
from tensordict import TensorDict

HOME = os.path.expanduser('~')
MAP_DIR = HOME + "/.ros/"

class BaseInformed(BaseVTR):
    def __init__(self):
        rospy.logwarn("Trying to instantiate VTR class")
        self.map_dir = MAP_DIR
        self.finished = False

        self.displacement = None
        self.phi_diff = None
        self.curr_dist = None
        self.last_curr_dist = None
        self.repeating = False
        self.final_dist = None
        self.map_traj = None
        self.map_phis = None
        self.actions = None
        self.dists = None
        self.curr_x = None
        self.curr_y = None
        self.curr_phi = None
        self.map_start_dist = None
        self.control_pub = rospy.Publisher("/bluetooth_teleop/cmd_vel", Twist, queue_size=5)
        self.pos_sub = rospy.Subscriber("/robot1/odometry", Odometry, self.odom_callback, queue_size=1)

    def repeat_map(self, start=float, end=float, map_name=str):
        dists, odoms, actions = self.fetch_map(map_name)
        x = []
        y = []
        phis = []
        for i in range(len(odoms)):
            x.append(odoms[i].pose.pose.position.x)
            y.append(odoms[i].pose.pose.position.y)
            phis.append(euler_from_quaternion([odoms[i].pose.pose.orientation.x, odoms[i].pose.pose.orientation.y, odoms[i].pose.pose.orientation.z, odoms[i].pose.pose.orientation.w])[-1])
        self.map_start_dist = start
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
        self.displacement = None
        self.last_curr_dist = None
        self.phi_diff = None
        self.finished = True
        self.curr_dist = None
        self.repeating = False
        self.map_start_dist = None
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

    def get_control_command(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        a, b, c = euler_from_quaternion(
            [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w])
        self.curr_phi = c
        curr_pos = np.array([self.curr_x, self.curr_y])

        # rospy.logwarn("curr pos: " + str(curr_pos))

        # find the nearest point in trajectory
        dists = np.sqrt(np.sum((curr_pos - self.map_traj) ** 2, axis=1))
        # dists = abs(self.shortest_distance(self.map_traj[:, 0], self.map_traj[:, 1], self.map_phis, curr_pos[0], curr_pos[1]))
        self.nearest_idx = np.argmin(dists)
        self.curr_dist = self.dists[self.nearest_idx]
        if self.last_curr_dist is None:
            self.last_curr_dist = self.curr_dist
        # rospy.logwarn("Nearest IDX: " + str(nearest_idx))

        if self.dists[self.nearest_idx] >= self.final_dist - 0.3:
            self.repeating = False
            self.finished = True
            return None

        # control policy
        target_pos = self.map_traj[self.nearest_idx]
        target_phi = self.map_phis[self.nearest_idx]

        self.displacement = self.shortest_distance(target_pos[0], target_pos[1], target_phi, curr_pos[0], curr_pos[1])
        self.phi_diff = self.smallest_angle_diff(target_phi, self.curr_phi)
        correction = (-self.phi_diff + (0.1 * self.displacement)) * 0.5
        correction = np.sign(correction) * min(abs(correction), 0.2)

        control_command = Twist()
        control_command.linear.x = max(self.actions[self.nearest_idx].linear.x, 1.0)
        control_command.angular.z = self.actions[self.nearest_idx].angular.z + correction
        # rospy.logwarn("Original command: " + str(self.actions[self.nearest_idx].angular.z) + ", correction " + str(correction))

        # rospy.logwarn("Trying to control the robot with displacement " + str(displacement) + " and rotation diff " + str(phi_diff) + "\nCorrection is: " + str(correction))
        # rospy.logwarn(str(control_command))

        # rospy.logwarn("Fetching control command at distance: " + str(self.curr_dist) + " with idx " + str(nearest_idx))

        return control_command

    def odom_callback(self, msg):
        # get curr position
        if self.repeating:
            control_command = self.get_control_command(msg)
            if control_command is None:
                self.control_pub.publish(Twist())
            else:
                self.control_pub.publish(control_command)
        else:
            self.control_pub.publish(Twist())


class VTREnv(BaseInformed):

    def __init__(self, input_size=5, training=False):
        super().__init__()
        # consts
        self.use_history = True
        self.training = training
        self.input_size = input_size
        self.max_dist_err = 3.0
        self.max_history = 1
        self.dist_span = 8
        self.min_step_dist = 0.2
        self.last_pos_hist = None
        self.finished = False

        # Data fetching
        self.processing = Processing()
        self.observation_buffer = DataFetching(self.input_size)

        # NN training
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        # vars
        self.curr_reward = 0.0
        self.last_x = None
        self.last_y = None
        self.last_phi = None
        self.target_action = None
        self.est_dist = None
        self.last_obs = None
        self.last_dist_err = None
        self.last_lat_err = None
        self.dist_err = None

    def render(self, mode='human'):
        pass

    def step(self, action):
        # CONTROL THE ROBOT
        if self.repeating:
            self.control_robot(action)

        # OBTAIN OBSERVATION
        obs, failure = self.get_observation()

        reward = self.curr_reward
        self.last_lat_err = self.displacement
        self.last_dist_err = self.dist_err
        self.last_curr_dist = self.curr_dist

        # CHECK FINISH
        if self.est_dist + 0.3 > self.final_dist:
            rospy.logwarn("Robot reached the end!")
            self.finished = True
            self.repeating = False
            self.control_pub.publish(Twist())

        # CHECK FAILURE
        if self.est_dist is not None and self.curr_dist is not None:
            dist_err = self.est_dist - self.curr_dist
        else:
            dist_err = self.max_dist_err
        if abs(dist_err) >= self.max_dist_err or failure:
            rospy.logwarn("!!!TRAVERSAL FAILED - INVALID DISTANCE ESTIMATE!!!")
            self.finished = True
            self.repeating = False
            self.control_pub.publish(Twist())

        rospy.logwarn("REWARD: " + str(reward))
        return obs, reward, self.finished, False

    def control_robot(self, action):
        nearest_z = self.get_nearest_command(self.est_dist)
        rospy.logwarn("Controlling robot with action: " + str(nearest_z) + " and correction " + str(action))

        control_command = Twist()
        control_command.linear.x = 2.0
        control_command.angular.z = action[0].cpu() + nearest_z

        self.est_dist -= action[1].cpu().numpy()
        self.control_pub.publish(control_command)

    def get_observation(self):
        if self.est_dist is None:
            self.est_dist = self.map_start_dist
        data = None
        counter = 0
        while data is None:
            data = self.observation_buffer.get_live_data()
            if data is None:
                rospy.logwarn("WAITING FOR NEW DATA!")
                rospy.sleep(0.01)
            counter += 1
            if counter >= 100 and self.last_obs is not None:
                rospy.logwarn("UNABLE TO OBTAIN NEW DATA - FAILURE!")
                return self.last_obs, True
        img_data = self.parse_hists(data[1:])
        img_pos = self.process_distance(data[0])

        obs = t.cat([img_data, img_pos]).float()
        self.last_obs = obs
        return obs, False

    def reset(self):
        super().reset()

        self.last_x = None
        self.last_y = None
        self.last_phi = None
        self.target_action = None
        self.est_dist = None
        self.last_pos_hist = None
        self.curr_reward = 0.0
        self.finished = False
        self.last_obs = None
        self.last_dist_err = None
        self.last_lat_err = None
        self.dist_err = None

    def fetch_map(self, name):
        self.processing.load_map(name)
        return super().fetch_map(name)

    def resize_histogram(self, x):
        # RESIZE BY FACTOR of 8
        reshaped_x = x.view(1, -1, 8)
        # Sum along the second dimension to aggregate every 8 bins into one
        resized_x = reshaped_x.sum(dim=2)
        return resized_x

    def parse_hists(self, obs):
        cat_tesnors = []
        # max_map_img = t.argmax(t.max(t.tensor(obs[0]), dim=1)[0])
        # rospy.logwarn("MAX MAP: " + str(max_map_img))
        for n_obs in obs:
            flat_tensor = t.tensor(n_obs[:, 256:-256], device=self.device).flatten()
            flat_tensor = self.resize_histogram(flat_tensor)
            norm_flat_tesnsor = (flat_tensor - t.mean(flat_tensor)) / t.std(flat_tensor)
            cat_tesnors.append(norm_flat_tesnsor.squeeze(0))
        new_obs = t.cat(cat_tesnors, dim=0).float()
        return new_obs

    def get_nearest_command(self, dist):
        # find the nearest point in trajectory
        dists = np.abs(dist - self.dists)
        nearest_idx = np.argmin(dists)
        target_z = self.actions[nearest_idx].angular.z
        rospy.logwarn("Fetched command: " + str(dist) + ", " + str(target_z))
        return target_z

    def process_odom(self):
        if self.last_x is not None:
            traveled_dist = np.sqrt((self.curr_x - self.last_x) ** 2 + (self.curr_y - self.last_y) ** 2)
            self.est_dist += traveled_dist
        self.last_x = self.curr_x
        self.last_y = self.curr_y

    def process_distance(self, img_dists):
        center = int((self.dist_span * 10) / 2.0)

        # GET POS HIST
        if self.last_pos_hist is None or self.last_x is None:
            self.last_pos_hist = t.ones(self.dist_span * 10 + 1, device=self.device) / (self.dist_span * 10 + 1)

        # GET IMG POS
        imgs_pos = t.zeros(self.dist_span * 10 + 1, device=self.device)
        img_dist_idxs = np.round((self.est_dist - img_dists) * 10)
        for index_diff in img_dist_idxs[0]:
            if abs(index_diff) < center:
                imgs_pos[int(index_diff) + center] = 0.2

        return imgs_pos

    def odom_callback(self, msg):
        # get curr position
        if self.repeating:
            if self.est_dist is None:
                self.est_dist = self.map_start_dist
                self.processing.pubSensorsInput(self.est_dist)
                return
            _ = super().get_control_command(msg)
            self.process_odom()
            if self.last_curr_dist is None or self.curr_dist - self.last_curr_dist > self.min_step_dist:
                self.processing.pubSensorsInput(self.est_dist)
                self.dist_err = abs(self.curr_dist - self.est_dist)
                if self.last_lat_err is None:
                    self.last_lat_err = self.displacement
                if self.last_dist_err is None:
                    self.last_dist_err = self.dist_err
                covered_dist = self.curr_dist - self.last_curr_dist
                self.curr_reward = covered_dist \
                                   - (self.dist_err - self.last_dist_err) \
                                   - (self.displacement - self.last_lat_err)


class GymEnvironment(EnvBase):

    def __init__(self):
        super().__init__()

        # subscribe observations
        self.map_idx = None
        self.finished = None
        self.dist = None
        self.sim = None
        self.map_name = None
        self.traversal_idx = 0
        self.eval = False

        self.sim = Simulator(MAP_DIR, pose_err_weight=1.0, rot_err_weight=np.pi / 16.0,
                             dist_weight=0.5, headless=True)
        self.vtr = VTREnv(training=True)
        self.device = self.vtr.device

        # TORCHRL specs
        self.batch_size = t.Size([1])
        self.observation_spec = CompositeSpec({"observation": UnboundedContinuousTensorSpec(shape=t.Size([1, 657]),
                                                                                            device=self.device)},
                                              shape=t.Size([1]))
        self.action_spec = CompositeSpec({"action": BoundedTensorSpec([[-0.25, -0.5]], [[0.25, 0.5]], t.Size([1, 2]), self.device)},
                                         shape=t.Size([1]))
        self.reward_spec = BoundedTensorSpec(-7.0, 5.0, t.Size([1, 1]), self.device)
        self.done_spec = BinaryDiscreteTensorSpec(1, shape=t.Size([1, 1]), dtype=t.bool)

    def round_setup(self, day_time=None, scene=None, random_teleport=None):
        time.sleep(3)
        self.traversal_idx += 1
        rospy.logwarn("------------ Starting round " + str(self.traversal_idx) + "! --------------")

        self.failure = False
        self.map_idx, self.map_name, self.dist = self.sim.reset_sim(day_time, scene, random_teleport)
        time.sleep(3)

    def _step(self, action):
        # main simulation loop
        obs, reward, self.finished, _ = self.vtr.step(action["action"][0])
        self.sim.plt_robot()
        failure = self.sim.failure_check()
        if failure:
            rospy.logwarn("!!! UNSUCCESSFUL TRAVERSAL !!!")
            self.finished = True
            failure = True
        if self.finished:
            self.vtr.control_pub.publish(Twist())
            self.sim.plt_robot(save_fig=True, idx=self.traversal_idx, eval=self.eval)
            if not failure:
                self.sim.traversal_summary()
        reward = max(reward, -7.0)
        # rospy.logwarn("observation: " + str(obs.shape) + ", has inf: " + str(t.any(t.isinf(obs))))
        return TensorDict({"observation": obs.unsqueeze(0),
                           "reward": t.tensor([reward], device=self.device).unsqueeze(0).float(),
                           "done": t.tensor([self.finished], device=self.device).unsqueeze(0)},
                          batch_size=[1])

    def set_eval(self, value: bool):
        self.eval = value

    def _reset(self, tensordict=None):
        self.vtr.reset()
        self.round_setup(day_time=np.random.uniform(0.3, 0.7), scene=0, random_teleport=True)
        self.sim.vtr_traversal(self.vtr, self.map_idx, self.dist)
        self.vtr.processing.pubSensorsInput(self.dist)
        rospy.sleep(0.25)
        obs, failure = self.vtr.get_observation()
        self.sim.plt_robot()
        # rospy.logwarn("observation: " + str(obs.shape) + ", has inf: " + str(t.any(t.isinf(obs))))
        return TensorDict({"observation": obs.unsqueeze(0)}, batch_size=[1])

    def _set_seed(self, seed):
        return

