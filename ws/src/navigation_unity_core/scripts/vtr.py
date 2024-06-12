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
from nn_model import FeedForward2, TransformerModel, PPOActorSimple
import os
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from tensordict.nn import TensorDictModule
from torchrl.data import UnboundedContinuousTensorSpec, BoundedTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec
import tensordict
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type


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
        self.distance_sub = rospy.Subscriber("/pfvtr/repeat/output_dist", SensorsOutput, self._dist_callback,
                                             queue_size=1)

        rospy.wait_for_service("pfvtr/stop_repeater")
        self.stop_pfvtr = rospy.ServiceProxy("pfvtr/stop_repeater", StopRepeater)
        rospy.logwarn("Service for stopping available")

        self.client = actionlib.SimpleActionClient("/pfvtr/repeater", MapRepeaterAction)  # for VTR
        rospy.logwarn("PFVTR successfully connected!")

    def repeat_map(self, start=float, end=float, map_name=str):
        """
        Required method!
        """
        self.finished = False
        self.target_dist = end
        self.map_start = True
        curr_action = MapRepeaterGoal(startPos=start, endPos=end, traversals=0, nullCmd=True, imagePub=self.image_pub,
                                      useDist=True, mapName=map_name)
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
        rospy.logwarn("Trying to instantiate VTR class")
        self.map_dir = "/home/zdeeno/.ros/"
        self.finished = False

        self.displacement = None
        self.phi_diff = None
        self.curr_dist = None
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
            phis.append(euler_from_quaternion(
                [odoms[i].pose.pose.orientation.x, odoms[i].pose.pose.orientation.y, odoms[i].pose.pose.orientation.z,
                 odoms[i].pose.pose.orientation.w])[-1])
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


class NeuralNet(InformedVTR):

    def __init__(self, input_size=5, training=False):
        super().__init__()
        # consts
        self.use_history = True
        self.LR = 0.0001
        self.training = training
        self.input_size = input_size
        self.max_dist_err = 2
        self.max_history = 1

        # vars
        self.action_buffer = []
        self.target_buffer = []
        self.vis_obs_buffer = []
        self.odom_obs_buffer = []

        # Data fetching
        self.processing = Processing()
        self.observation_buffer = DataFetching(self.input_size)

        # NN training
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        # self.model = FeedForward(2).to(self.device)
        self.model = TransformerModel(2).to(self.device)

        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.LR)
        self.loss = t.nn.MSELoss()

        self.last_x = None
        self.last_y = None
        self.last_phi = None
        self.target_action = None
        self.est_dist = None

    def reset(self):
        super().reset()
        if self.training:
            self.train_model()

        self.last_x = None
        self.last_y = None
        self.last_phi = None
        self.target_action = None
        self.est_dist = None
        self.action_buffer = []
        self.target_buffer = []
        self.vis_obs_buffer = []
        self.odom_obs_buffer = []

    def fetch_map(self, name):
        self.processing.load_map(name)
        return super().fetch_map(name)

    def train_model(self):
        # TODO: I expect this to fail miserably - dimensions
        rospy.logwarn("Training the model using " + str(len(self.action_buffer)) + " observations.")

        if len(self.action_buffer) > 0:
            actions = t.stack(self.action_buffer)
            targets = t.stack(self.target_buffer[:actions.shape[0]])
            l = self.loss(actions, targets)
            l.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def parse_hists(self, obs):
        cat_tesnors = []
        for n_obs in obs:
            flat_tensor = t.tensor(n_obs[:, 256:-256], device=self.device).flatten()
            norm_flat_tesnsor = (flat_tensor - t.std(flat_tensor)) / t.var(flat_tensor)
            cat_tesnors.append(norm_flat_tesnsor)
        new_obs = t.cat(cat_tesnors, dim=0).float()
        return new_obs

    def get_odom_data(self):
        # last live odom diff
        if self.last_x is not None:
            dist = np.sqrt((self.curr_x - self.last_x) ** 2 + (self.curr_y - self.last_y) ** 2)
            phi_diff = self.smallest_angle_diff(self.curr_phi, self.last_phi)
            live_odom_diff = t.tensor([dist, phi_diff], device=self.device).float()
        else:
            live_odom_diff = t.tensor([0.0, 0.0], device=self.device).float()
        self.last_x = self.curr_x
        self.last_y = self.curr_y
        self.last_phi = self.curr_phi

        # last live action
        if len(self.action_buffer) > 0:
            last_live_action = self.action_buffer[-1][:2].detach()
        else:
            last_live_action = t.tensor([0.0, 0.0], device=self.device).float()

        # curr map action
        # TODO: this is not valid!!! must use est dist
        curr_map_action = self.target_action[:2]

        # curr map odom diff
        # TODO: not trivial - must be between images

        out = t.cat([live_odom_diff, last_live_action, curr_map_action])
        return out

    def fetch_prev_obs(self, img_data, odom_data):
        if len(self.vis_obs_buffer) == 0:
            img_data = img_data.unsqueeze(0)
            odom_data = odom_data.unsqueeze(0)
            return img_data, odom_data
        else:
            past_img_data = self.vis_obs_buffer[-self.max_history:]
            past_img_data = past_img_data[::-1]
            past_odom_data = self.odom_obs_buffer[-self.max_history:]
            past_odom_data = past_odom_data[::-1]
            img_out = t.stack([img_data, *past_img_data], dim=0).to(self.device)
            odom_out = t.stack([odom_data, *past_odom_data], dim=0).to(self.device)
            return img_out, odom_out

    def odom_callback(self, msg):
        # get curr position
        if self.repeating:
            if self.est_dist is None:
                self.est_dist = self.map_start_dist
            target_command = super().get_control_command(msg)
            if target_command is None:
                return
            dist_err = self.curr_dist - self.est_dist
            self.target_action = t.tensor([target_command.linear.x, target_command.angular.z, dist_err],
                                          device=self.device).float()

            # TODO: When this is called just before parsing the hists, then the data are probably outdated !!!

            data = None
            while data is None:
                # TODO: Wrong input at the end !!!
                self.processing.pubSensorsInput(self.est_dist)
                data = self.observation_buffer.get_live_data()
            img_data = self.parse_hists(data)
            odom_data = self.get_odom_data()

            if self.use_history:
                img_data, odom_data = self.fetch_prev_obs(img_data, odom_data)
            action = self.model.forward(img_data, odom_data)

            control_command = Twist()
            control_command.linear.x = action[0].cpu()
            control_command.angular.z = action[1].cpu()
            self.est_dist += action[2].cpu().detach().numpy()

            self.control_pub.publish(control_command)

            if abs(dist_err) < self.max_dist_err:
                if self.use_history:
                    self.vis_obs_buffer.append(img_data[0])
                    self.odom_obs_buffer.append(odom_data[0])
                self.action_buffer.append(action)
                self.target_buffer.append(self.target_action)
                rospy.logwarn("action: " + str(action.cpu().detach().numpy()) + ", target: " + str(
                    self.target_action.cpu().detach().numpy()))
        else:
            self.control_pub.publish(Twist())


class NeuralNet2(InformedVTR):

    def __init__(self, input_size=5, training=False):
        super().__init__()
        # consts
        self.use_history = True
        self.LR = 1e-5
        self.training = training
        self.input_size = input_size
        self.max_dist_err = 3.0
        self.max_history = 1
        self.dist_span = 8
        self.last_pos_hist = None

        # vars
        self.output_buffer = []
        self.target_buffer = []

        # Data fetching
        self.processing = Processing()
        self.observation_buffer = DataFetching(self.input_size)

        # NN training
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.model = FeedForward2(2, self.dist_span).to(self.device)

        self.model.load_state_dict(t.load("/home/zdeeno/.ros/models/nn.pt"))

        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.LR)
        self.turn_loss = t.nn.MSELoss()
        self.dist_loss = t.nn.BCELoss()

        self.last_x = None
        self.last_y = None
        self.last_phi = None
        self.target_action = None
        self.est_dist = None

    def reset(self):
        super().reset()
        if self.training:
            self.train_model()

        self.last_x = None
        self.last_y = None
        self.last_phi = None
        self.target_action = None
        self.est_dist = None
        self.output_buffer = []
        self.target_buffer = []
        self.last_pos_hist = None

    def fetch_map(self, name):
        self.processing.load_map(name)
        return super().fetch_map(name)

    def train_model(self):
        rospy.logwarn("Training the model using " + str(len(self.output_buffer)) + " observations.")

        if len(self.output_buffer) > 0:
            actions = t.stack(self.output_buffer)
            targets = t.stack(self.target_buffer[:actions.shape[0]])
            l_turn = self.turn_loss(actions[:, 0], targets[:, 0])
            l_dist = self.dist_loss(actions[:, 1:], targets[:, 1:]) * 10.0
            l = l_dist + l_turn
            l.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            rospy.logwarn("Turn loss: " + str(l_turn.cpu().detach()) + ", Dist loss: " + str(l_dist.cpu().detach()))

        t.save(self.model.state_dict(), "/home/zdeeno/.ros/models/nn.pt")

    def parse_hists(self, obs):
        cat_tesnors = []
        # max_map_img = t.argmax(t.max(t.tensor(obs[0]), dim=1)[0])
        # rospy.logwarn("MAX MAP: " + str(max_map_img))
        for n_obs in obs:
            flat_tensor = t.tensor(n_obs[:, 256:-256], device=self.device).flatten()
            norm_flat_tesnsor = (flat_tensor - t.std(flat_tensor)) / t.var(flat_tensor)
            cat_tesnors.append(norm_flat_tesnsor)
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
                imgs_pos[int(index_diff) + center] = 1

        # GET TARGET
        target = t.zeros(self.dist_span * 10 + 1, device=self.device).float()
        if self.curr_dist is None:
            return None, None
        dist_err = self.est_dist - self.curr_dist
        dist_displacement = int(round(dist_err * 10.0))
        if abs(dist_displacement) < center - 4:
            # triangle smoothing for target

            target[center + 2] = 0.33 / 2.0
            target[center + 1] = 0.66 / 2.0
            target[center] = 1.0 / 2.0
            target[center - 1] = 0.66 / 2.0
            target[center - 2] = 0.33 / 2.0

            target[dist_displacement + center + 4] = 0.2
            target[dist_displacement + center + 3] = 0.4
            target[dist_displacement + center + 2] = 0.6
            target[dist_displacement + center + 1] = 0.8
            target[dist_displacement + center] = 1.0
            target[dist_displacement + center - 1] = 0.8
            target[dist_displacement + center - 2] = 0.6
            target[dist_displacement + center - 3] = 0.4
            target[dist_displacement + center - 4] = 0.2

        return target, imgs_pos

    def odom_callback(self, msg):
        # get curr position
        if self.repeating:
            if self.est_dist is None:
                self.est_dist = self.map_start_dist
                self.processing.pubSensorsInput(self.est_dist)
                return
            target_command = super().get_control_command(msg)
            self.process_odom()
            self.processing.pubSensorsInput(self.est_dist)
            if target_command is None:
                return
            rospy.sleep(0.05)
            data = self.observation_buffer.get_live_data()
            if data is None:
                return
            img_data = self.parse_hists(data[1:])
            target_hist, img_pos = self.process_distance(data[0])
            if target_hist is None:
                return
            self.target_action = t.empty(target_hist.size(0) + 1, device=self.device, dtype=t.float)
            self.target_action[0] = target_command.angular.z

            obs = t.cat([img_data, img_pos, self.last_pos_hist])
            nearest_z = self.get_nearest_command(self.est_dist)
            output = self.model.forward(obs, nearest_z)

            control_command = Twist()
            control_command.linear.x = 2.0
            control_command.angular.z = output[0].cpu()
            center = int((self.dist_span * 10.0) / 2)
            est_diff = int((np.argmax(output[1:].cpu().detach().numpy()) - center))
            true_diff = (np.argmax(target_hist.cpu().detach().numpy()) - center)

            dist_err = self.est_dist - self.curr_dist
            self.est_dist -= (est_diff / 10.0)
            tmp_hist = output[1:].detach()
            pos_hist = t.zeros_like(tmp_hist)
            if est_diff < 0:
                pos_hist[-est_diff:] = tmp_hist[:est_diff]
            if est_diff > 0:
                pos_hist[:-est_diff] = tmp_hist[est_diff:]

            # rospy.logwarn("BEFORE: " + str(tmp_hist.cpu().numpy().tolist()))
            # rospy.logwarn("AFTER: " + str(pos_hist.cpu().numpy().tolist()))
            # rospy.logwarn("DIFF: " + str(est_diff))

            if (abs(est_diff) <= 2 and abs(true_diff) > 4) or abs(est_diff - true_diff) < 5:
                target_hist[center - 2:center + 3] = 0.0
            self.target_action[1:] = target_hist

            self.last_pos_hist = pos_hist
            self.control_pub.publish(control_command)
            # self.processing.pubSensorsInput(self.est_dist)
            rospy.logwarn("action: " + str((output[0].cpu().detach(), est_diff)) + ", target: " + str(
                (self.target_action[0].cpu().detach().numpy(), true_diff)) + ", dist err:" + str(dist_err))

            if self.est_dist + 0.3 > self.final_dist:
                self.finished = True
                self.repeating = False
                return

            if abs(dist_err) < self.max_dist_err:
                self.output_buffer.append(output)
                self.target_buffer.append(self.target_action)
            else:
                rospy.logwarn("!!!TRAVERSAL FAILED - INVALID DISTANCE ESTIMATE!!!")
                self.finished = True
                self.repeating = False
                # rospy.logwarn("!!!INVALID DISTANCE ESTIMATE - FIXING DISTANCE!!!")
                # self.est_dist -= true_diff/10.0
        else:
            self.control_pub.publish(Twist())


class RLAgent(InformedVTR):

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
        self.finished = True
        self.end_dist = None

        # Data fetching
        self.processing = Processing()
        self.observation_buffer = DataFetching(self.input_size)
        self.control_pub = rospy.Publisher("/bluetooth_teleop/cmd_vel", Twist, queue_size=5)

        # NN device
        HOME = os.path.expanduser('~')
        SAVE_DIR = HOME + "/.ros/models/"
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        actor_net = PPOActorSimple(2, hidden_size=1024).float().to(self.device)
        actor_net.load_state_dict(t.load(SAVE_DIR + "actor_net.pt"))
        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )

        action_spec = CompositeSpec(
            {"action": BoundedTensorSpec([[-0.25, -0.5]], [[0.25, 0.5]], t.Size([1, 2]), self.device)},
            shape=t.Size([1]))

        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": t.tensor([-0.25, -0.5]),
                "max": t.tensor([0.25, 0.5]),
                # "event_dims": 2,
                # "tanh_loc": True
            },
            return_log_prob=True,
            default_interaction_type=tensordict.nn.InteractionType.MEAN,
            # we'll need the log-prob for the numerator of the importance weights
        )

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
        self.last_dist = None

    def repeat_map(self, start=float, end=float, map_name=str):
        self.processing.load_map(map_name)
        self.est_dist = start
        super().repeat_map(start=start, end=end, map_name=map_name)
        return True

    def is_finished(self) -> bool:
        return self.finished

    def control_robot(self, action):
        print(action)
        self.est_dist -= action[1].cpu().detach().numpy()

        nearest_z, nearest_x = self.get_nearest_command(self.est_dist)
        rospy.logwarn("Controlling robot with action: " + str(nearest_z) + " and correction " + str(action))

        control_command = Twist()
        control_command.linear.x = max(1.0, nearest_x)
        control_command.angular.z = action[0].cpu().detach().numpy() + nearest_z

        self.control_pub.publish(control_command)

    def get_observation(self):
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
        self.end_dist = None
        self.finished = True
        self.last_dist = None


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
        target_x = self.actions[nearest_idx].linear.x

        rospy.logwarn("Fetched command: " + str(dist) + ", " + str(target_z))
        return target_z, target_x

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
        if not self.finished and self.repeating:
            _ = super().get_control_command(msg)
            self.process_odom()
            self.processing.pubSensorsInput(self.est_dist)

            # CHECK FINISH
            if self.est_dist + 0.3 > self.final_dist:
                rospy.logwarn("Robot reached the end!")
                self.finished = True
                self.repeating = False
                self.control_pub.publish(Twist())

            # GET CONTROL COMMAND
            obs, _ = self.get_observation()
            net_in = TensorDict({"observation": obs.unsqueeze(0),
                        "reward": t.tensor([0.0], device=self.device).unsqueeze(0).float(),
                        "done": t.tensor([self.finished], device=self.device).unsqueeze(0)},
                       batch_size=[1])
            with set_exploration_type(ExplorationType.MEAN), t.no_grad():
                action = self.policy_module.forward(net_in)

            if self.est_dist > self.last_dist + self.min_step_dist or self.last_dist is None:
                self.control_robot(action["action"][0])
                self.last_dist = self.est_dist
            else:
                rospy.logwarn("Waiting for action ...")