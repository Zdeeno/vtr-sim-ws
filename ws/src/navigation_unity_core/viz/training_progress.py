from matplotlib import pyplot as plt
import numpy as np
import os
import rosbag


DIR = os.path.expanduser('~') + "/.ros/trajectory_plots_eval"
#map_idx = 0
#DIR_BAG = os.path.expanduser('~') + "/.ros/sim7_vtr/sim7_vtr.bag"
map_idx = 1
DIR_BAG = os.path.expanduser('~') + "/.ros/sim6_vtr/sim6_vtr.bag"


all_trajs = []


def fetch_map():
    bag = rosbag.Bag(DIR_BAG, "r")
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

    x = []
    y = []

    for i in range(len(odoms)):
        x.append(odoms[i].pose.pose.position.x)
        y.append(odoms[i].pose.pose.position.y)

    return x, y


for file in sorted(os.listdir(DIR), key=lambda x: int(x.split(".")[0].split("_")[0])):
    if file.endswith(str(map_idx) + ".csv"):
        print(file)
        traj = np.genfromtxt(os.path.join(DIR, file), delimiter=",")
        all_trajs.append(np.genfromtxt(os.path.join(DIR, file), delimiter=","))

plt.figure()


cmap = plt.cm.get_cmap('coolwarm')
for i, traj in enumerate(all_trajs):
    clr = cmap(i / (len(all_trajs) - 1))
    plt.plot(traj[0], traj[1], color=clr, alpha=0.75, linewidth=2.0)

x, y = fetch_map()
plt.plot(x, y, color="k", linewidth=3)
limit = 5

plt.ylim([min(y) - limit, max(y) + limit])
plt.xlim([min(x) - limit, max(x) + limit])
plt.grid()
plt.title("Plot of " + str(len(all_trajs)) + " traversals")

plt.show()