import torch as t
import math
import numpy as np
import rospy
from tensordict.nn.distributions import NormalParamExtractor

class FeedForward(t.nn.Module):

    def __init__(self, lookaround: int):
        super().__init__()
        self.lookaround = lookaround
        map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        hist_size = 512
        input_size = total_size * hist_size

        # histograms from visual data
        self.visual = t.nn.Sequential(t.nn.Linear(input_size, 512),
                                      t.nn.ReLU(),
                                      t.nn.Linear(512, 512),
                                      t.nn.ReLU(),
                                      t.nn.Linear(512, 64))

        # odometry and action commands: live_odom_diff + last_action + map_action
        # TODO: add map_odom_diff (all map images)
        self.odometric = t.nn.Sequential(t.nn.Linear(6, 256),
                                         t.nn.ReLU(),
                                         t.nn.Linear(256, 256),
                                         t.nn.ReLU(),
                                         t.nn.Linear(256, 64))

        # output size: (x, phi, dist_diff)
        self.out_head = t.nn.Sequential(t.nn.Linear(128, 512),
                                        t.nn.ReLU(),
                                        t.nn.Linear(512, 512),
                                        t.nn.ReLU(),
                                        t.nn.Linear(512, 3))

    def forward(self, visual, odometric):
        y1 = self.visual(visual)
        y2 = self.odometric(odometric)
        cat = t.cat([y1, y2], dim=-1)
        action = self.out_head(cat)
        # normalize outputs:
        action[0] = t.sigmoid(action[0]) + 1.0
        action[1] = t.tanh(action[1]) * 2.0
        action[2] = t.tanh(action[2]) * 5.0
        return action


class FeedForward2(t.nn.Module):

    def __init__(self, lookaround: int, dist_window=8):
        super().__init__()
        self.lookaround = lookaround
        self.map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = self.map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        self.hist_size = 512
        input_size = total_size * self.hist_size
        self.dist_hist_size = dist_window * 10 + 1
        input_size += self.dist_hist_size * 2

        # histograms from visual data (1, 2, 5)
        self.live_map_encoder = t.nn.Sequential(t.nn.Linear(self.hist_size, 128),
                                                t.nn.ReLU(),
                                                t.nn.Linear(128, 64))

        self.trans_encoder = t.nn.Sequential(t.nn.Linear(self.hist_size, 128),
                                             t.nn.ReLU(),
                                             t.nn.Linear(128, 64))

        self.curr_trans_encoder = t.nn.Sequential(t.nn.Linear(self.hist_size, 128),
                                                  t.nn.ReLU(),
                                                  t.nn.Linear(128, 64))

        self.dist_encoder = t.nn.Sequential(t.nn.Linear(self.dist_hist_size * 2, 256),
                                            t.nn.ReLU(),
                                            t.nn.Linear(256, 64))

        self.out_turn = t.nn.Sequential(t.nn.Linear(10 * 64 + 64, 256),
                                        t.nn.ReLU(),
                                        t.nn.Linear(256, 1))

        self.out_dist = t.nn.Sequential(t.nn.Linear(10 * 64 + 64, 256),
                                        t.nn.ReLU(),
                                        t.nn.Linear(256, self.dist_hist_size))

    def forward(self, x, curr_action):
        anchor1 = self.map_obs_size * self.hist_size
        anchor2 = (self.map_obs_size * self.hist_size)*2 - self.hist_size
        anchor3 = anchor2 + self.hist_size
        # rospy.logwarn(str(x.shape))
        # rospy.logwarn(str(anchor1) + "," + str(anchor2) + ", " + str(anchor3))
        live_vs_map = self.live_map_encoder(x[:anchor1].view((self.map_obs_size, self.hist_size))).flatten()
        trans = self.trans_encoder(x[anchor1:anchor2].view((self.map_obs_size - 1, self.hist_size))).flatten()
        curr_trans = self.curr_trans_encoder(x[anchor2:anchor3].view((1, self.hist_size))).flatten()
        dists = self.dist_encoder(x[anchor3:].view((1, self.dist_hist_size * 2))).flatten()
        bottleneck = t.cat([live_vs_map, trans, curr_trans, dists])
        out_turn = self.out_turn(bottleneck)
        out_dist = self.out_dist(bottleneck)
        # normalize outputs:
        out = t.cat([out_turn, out_dist])
        out[0] = t.tanh(out[0]) * 0.5 + curr_action
        out[1:] = t.sigmoid(out[1:])
        return out


class PositionalEncoding(t.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 16):
        super().__init__()
        self.dropout = t.nn.Dropout(p=dropout)

        position = t.arange(max_len).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = t.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = t.sin(position * div_term)
        pe[:, 0, 1::2] = t.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(t.nn.Module):

    def __init__(self, lookaround: int):
        super().__init__()
        self.lookaround = lookaround
        map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        hist_size = 512
        input_size = total_size * hist_size
        embedding_size = 64
        max_in_size = 16 + 1

        self.action_embedding = t.nn.Parameter(t.normal(0, 0.1, (1, embedding_size)))
        # histograms from visual data
        self.visual = t.nn.Sequential(t.nn.Linear(input_size, 512),
                                      t.nn.ReLU(),
                                      t.nn.Linear(512, 512),
                                      t.nn.ReLU(),
                                      t.nn.Linear(512, embedding_size//2))

        # odometry and action commands: live_odom_diff + last_action + map_action
        # TODO: add map_odom_diff (all map images)
        self.odometric = t.nn.Sequential(t.nn.Linear(6, 256),
                                         t.nn.ReLU(),
                                         t.nn.Linear(256, 256),
                                         t.nn.ReLU(),
                                         t.nn.Linear(256, embedding_size//2))

        self.pe = PositionalEncoding(embedding_size, 0.1, max_in_size)
        # output size: (x, phi, dist_diff)
        encoder_layer = t.nn.TransformerEncoderLayer(embedding_size, 4, 256, 0.1)
        self.trans = t.nn.TransformerEncoder(encoder_layer, 4)
        self.out_head = t.nn.Sequential(t.nn.Linear(64, 128),
                                        t.nn.ReLU(),
                                        t.nn.Linear(128, 128),
                                        t.nn.ReLU(),
                                        t.nn.Linear(128, 3))

    def forward(self, visual, odometric):
        y1 = self.visual(visual)
        y2 = self.odometric(odometric)
        obs = t.cat([y1, y2], dim=-1)
        action_obs = t.cat([self.action_embedding, obs], dim=0)
        pe_cat = self.pe(action_obs)
        trans_out = self.trans(pe_cat)
        action = self.out_head(trans_out[0, 0])
        # normalize outputs:
        action[0] = t.sigmoid(action[0]) + 1.0
        action[1] = t.tanh(action[1]) * 2.0
        action[2] = t.tanh(action[2]) * 5.0
        return action


class TransformerObserver(t.nn.Module):

    def __init__(self, lookaround: int):
        super().__init__()
        self.lookaround = lookaround
        map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        hist_size = 512
        input_size = total_size * hist_size
        embedding_size = 64
        max_in_size = 16 + 1

        self.obs_embedding = t.nn.Parameter(t.normal(0, 0.1, (1, embedding_size)))
        # histograms from visual data
        self.visual = t.nn.Sequential(t.nn.Linear(input_size, 512),
                                      t.nn.ReLU(),
                                      t.nn.Linear(512, 512),
                                      t.nn.ReLU(),
                                      t.nn.Linear(512, embedding_size//2))

        # odometry and action commands: live_odom_diff + last_action + map_action
        # TODO: add map_odom_diff (all map images)
        self.odometric = t.nn.Sequential(t.nn.Linear(6, 256),
                                         t.nn.ReLU(),
                                         t.nn.Linear(256, 256),
                                         t.nn.ReLU(),
                                         t.nn.Linear(256, embedding_size//2))

        self.pe = PositionalEncoding(embedding_size, 0.1, max_in_size)
        # output size: (x, phi, dist_diff)
        encoder_layer = t.nn.TransformerEncoderLayer(embedding_size, 4, 256, 0.1)
        self.trans = t.nn.TransformerEncoder(encoder_layer, 4)

    def forward(self, visual, odometric):
        y1 = self.visual(visual)
        y2 = self.odometric(odometric)
        obs = t.cat([y1, y2], dim=-1)
        action_obs = t.cat([self.obs_embedding, obs], dim=0)
        pe_cat = self.pe(action_obs)
        obs_emb = self.trans(pe_cat)[0, 0]
        return obs_emb


class SimpleObserver(t.nn.Module):

    def __init__(self, lookaround: int):
        super().__init__()
        self.lookaround = lookaround
        map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        hist_size = 512
        input_size = total_size * hist_size
        embedding_size = 64
        max_in_size = 16 + 1

        self.obs_embedding = t.nn.Parameter(t.normal(0, 0.1, (1, embedding_size)))
        # histograms from visual data
        self.visual = t.nn.Sequential(t.nn.Linear(input_size, 512),
                                      t.nn.ReLU(),
                                      t.nn.Linear(512, 512),
                                      t.nn.ReLU(),
                                      t.nn.Linear(512, embedding_size//2))

        # odometry and action commands: live_odom_diff + last_action + map_action
        # TODO: add map_odom_diff (all map images)
        self.odometric = t.nn.Sequential(t.nn.Linear(6, 256),
                                         t.nn.ReLU(),
                                         t.nn.Linear(256, 256),
                                         t.nn.ReLU(),
                                         t.nn.Linear(256, embedding_size//2))

    def forward(self, visual, odometric):
        y1 = self.visual(visual)
        y2 = self.odometric(odometric)
        obs = t.cat([y1, y2], dim=-1)
        return obs[0]


class Actor(t.nn.Module):

    def __init__(self, lookaround: int):
        super().__init__()
        self.lookaround = lookaround
        map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        hist_size = 512
        input_size = total_size * hist_size
        embedding_size = 64
        max_in_size = 16 + 1

        self.actor = t.nn.Sequential(t.nn.Linear(embedding_size, 128),
                                     t.nn.ReLU(),
                                     t.nn.Linear(128, 128),
                                     t.nn.ReLU(),
                                     t.nn.Linear(128, 3))

    def forward(self, obs_emb):
        action = self.actor(obs_emb)
        # normalize outputs:
        action[0] = t.sigmoid(action[0]) + 1.0
        action[1] = t.tanh(action[1]) * 2.0
        action[2] = t.tanh(action[2]) * 3.0

        return action


class Critic(t.nn.Module):

    def __init__(self, lookaround: int):
        super().__init__()
        self.lookaround = lookaround
        map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        hist_size = 512
        input_size = total_size * hist_size
        embedding_size = 64
        max_in_size = 16 + 1

        self.critic = t.nn.Sequential(t.nn.Linear(embedding_size + 3, 128),
                                      t.nn.ReLU(),
                                      t.nn.Linear(128, 128),
                                      t.nn.ReLU(),
                                      t.nn.Linear(128, 2))

    def forward(self, obs_emb, action):
        critic_in = t.cat([obs_emb, action], dim=-1)
        estimate = self.critic(critic_in)
        estimate[0] = t.tanh(estimate[0]) * 3.0
        estimate[1] = t.tanh(estimate[1]) * 5.0
        return estimate


class SinActivation(t.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return t.sin(input)


class PPOActor(t.nn.Module):

    def __init__(self, lookaround: int, dist_window=8):
        super().__init__()
        self.lookaround = lookaround
        self.map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = self.map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        self.hist_size = 64
        input_size = total_size * self.hist_size
        self.dist_hist_size = dist_window * 10 + 1
        input_size += self.dist_hist_size * 2

        # histograms from visual data (1, 2, 5)
        self.live_map_encoder = t.nn.Sequential(t.nn.Linear(self.hist_size, 128),
                                                t.nn.Tanh(),
                                                t.nn.Linear(128, 64))

        self.trans_encoder = t.nn.Sequential(t.nn.Linear(self.hist_size, 128),
                                             t.nn.Tanh(),
                                             t.nn.Linear(128, 64))

        self.curr_trans_encoder = t.nn.Sequential(t.nn.Linear(self.hist_size, 128),
                                                  t.nn.Tanh(),
                                                  t.nn.Linear(128, 64))

        self.dist_encoder = t.nn.Sequential(t.nn.Linear(self.dist_hist_size, 256),
                                            t.nn.Tanh(),
                                            t.nn.Linear(256, 64))

        self.out = t.nn.Sequential(t.nn.Linear(10 * 64 + 64, 256),
                                   t.nn.Tanh(),
                                   t.nn.Linear(256, 4))

        self.norm = NormalParamExtractor()

    def pass_network(self, x):
        anchor1 = self.map_obs_size * self.hist_size
        anchor2 = (self.map_obs_size * self.hist_size)*2 - self.hist_size
        anchor3 = anchor2 + self.hist_size
        # rospy.logwarn(str(x.shape))
        # rospy.logwarn(str(anchor1) + "," + str(anchor2) + ", " + str(anchor3))
        # print(" -------------- INPUT SHAPE: \n" + str(x.shape))
        live_vs_map = self.live_map_encoder(x[:anchor1].view((self.map_obs_size, self.hist_size))).flatten()
        trans = self.trans_encoder(x[anchor1:anchor2].view((self.map_obs_size - 1, self.hist_size))).flatten()
        curr_trans = self.curr_trans_encoder(x[anchor2:anchor3].view((1, self.hist_size))).flatten()
        dists = self.dist_encoder(x[anchor3:].view((1, self.dist_hist_size))).flatten()
        bottleneck = t.cat([live_vs_map, trans, curr_trans, dists])
        out = self.out(bottleneck).unsqueeze(0)
        return out

    def forward(self, x):
        # print("PASSING ACTOR: " + str(x.shape))
        if x.shape[0] > 1:
            batch_size = x.shape[0]
            out_list = []
            for i in range(batch_size):
                x_one = x[i]
                out = self.pass_network(x_one)
                out_list.append(out)
            out = t.cat(out_list, dim=0)
        else:
            x = x[0, :]
            out = self.pass_network(x)
        return self.norm(out)



class PPOValue(t.nn.Module):

    def __init__(self, lookaround: int, dist_window=8):
        super().__init__()
        self.lookaround = lookaround
        self.map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = self.map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        self.hist_size = 64
        input_size = total_size * self.hist_size
        self.dist_hist_size = dist_window * 10 + 1
        input_size += self.dist_hist_size * 2

        # histograms from visual data (1, 2, 5)
        self.live_map_encoder = t.nn.Sequential(t.nn.Linear(self.hist_size, 128),
                                                t.nn.Tanh(),
                                                t.nn.Linear(128, 64))

        self.trans_encoder = t.nn.Sequential(t.nn.Linear(self.hist_size, 128),
                                             t.nn.Tanh(),
                                             t.nn.Linear(128, 64))

        self.curr_trans_encoder = t.nn.Sequential(t.nn.Linear(self.hist_size, 128),
                                                  t.nn.Tanh(),
                                                  t.nn.Linear(128, 64))

        self.dist_encoder = t.nn.Sequential(t.nn.Linear(self.dist_hist_size, 256),
                                            t.nn.Tanh(),
                                            t.nn.Linear(256, 64))

        self.out = t.nn.Sequential(t.nn.Linear(10 * 64 + 64, 256),
                                   t.nn.Tanh(),
                                   t.nn.Linear(256, 1))

    def pass_network(self, x):
        anchor1 = self.map_obs_size * self.hist_size
        anchor2 = (self.map_obs_size * self.hist_size) * 2 - self.hist_size
        anchor3 = anchor2 + self.hist_size
        # rospy.logwarn(str(x.shape))
        # rospy.logwarn(str(anchor1) + "," + str(anchor2) + ", " + str(anchor3))
        live_vs_map = self.live_map_encoder(x[:anchor1].view((self.map_obs_size, self.hist_size))).flatten()
        trans = self.trans_encoder(x[anchor1:anchor2].view((self.map_obs_size - 1, self.hist_size))).flatten()
        curr_trans = self.curr_trans_encoder(x[anchor2:anchor3].view((1, self.hist_size))).flatten()
        dists = self.dist_encoder(x[anchor3:].view((1, self.dist_hist_size))).flatten()
        bottleneck = t.cat([live_vs_map, trans, curr_trans, dists])
        out = self.out(bottleneck).unsqueeze(0)
        return out

    def forward(self, x):
        if len(x.shape) > 2:
            batched_x = x[0]
            batch_size = batched_x.shape[0]
            out_list = []
            for i in range(batch_size):
                x = batched_x[i]
                out = self.pass_network(x)
                out_list.append(out)
            out = t.cat(out_list, dim=1)
            return out

        if x.shape[0] > 1:
            batched_x = x
            batch_size = batched_x.shape[0]
            out_list = []
            for i in range(batch_size):
                x = batched_x[i]
                out = self.pass_network(x)
                out_list.append(out)
            out = t.cat(out_list, dim=1)
            return out[0]

        x = x[0, :]
        return self.pass_network(x)
