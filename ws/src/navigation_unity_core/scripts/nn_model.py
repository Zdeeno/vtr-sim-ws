import torch as t


class FeedForward(t.nn.Module):
    def __init__(self, lookaround: int):
        super().__init__()
        self.lookaround = lookaround
        map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        hist_size = 1024
        input_size = total_size * hist_size

        # histograms from visual data
        self.visual = t.nn.Sequential(t.nn.Linear(input_size, 1024),
                                      t.nn.ReLU(),
                                      t.nn.Linear(1024, 1024),
                                      t.nn.ReLU(),
                                      t.nn.Linear(1024, 64))

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
        return action



class SinActivation(t.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return t.sin(input)