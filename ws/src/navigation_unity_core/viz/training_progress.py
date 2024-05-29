from matplotlib import pyplot as plt
import numpy as np
import os
import rosbag

DIR = os.path.expanduser('~') + "/.ros/trajectory_plots"

def fetch_map(dir_bag):
    bag = rosbag.Bag(dir_bag, "r")
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


def individual_travs():

    for plot_idx in range(1, 10):

        map_idx = plot_idx
        dir_bag = os.path.expanduser('~') + "/.ros/sim" + str(map_idx) + "_vtr/sim" + str(map_idx) + "_vtr.bag"

        all_trajs = []

        for file in sorted(os.listdir(DIR), key=lambda x: int(x.split(".")[0].split("_")[0])):
            if file.endswith(str(map_idx) + ".csv"):
                print(file)
                traj = np.genfromtxt(os.path.join(DIR, file), delimiter=",")
                all_trajs.append(np.genfromtxt(os.path.join(DIR, file), delimiter=","))

        plt.figure()

        cmap = plt.cm.get_cmap('coolwarm')
        for i, traj in enumerate(all_trajs):
            clr = cmap(i / (len(all_trajs)))
            plt.scatter(traj[0][0], traj[1][0], color=clr, alpha=0.75, marker="x", s=50)
            plt.plot(traj[0], traj[1], color=clr, alpha=0.75, linewidth=2.0)

        x, y = fetch_map(dir_bag)
        plt.plot(x, y, color="k", linewidth=3)
        limit = 5

        plt.ylim([min(y) - limit, max(y) + limit])
        plt.xlim([min(x) - limit, max(x) + limit])
        plt.grid()
        plt.title("Plot of " + str(len(all_trajs)) + " traversals")

        plt.show()


def convergence():

    all_maps_dists = []

    for plot_idx in range(1, 10):

        map_idx = plot_idx
        dir_bag = os.path.expanduser('~') + "/.ros/sim" + str(map_idx) + "_vtr/sim" + str(map_idx) + "_vtr.bag"
        x, y = fetch_map(dir_bag)

        all_dists = []

        for file in sorted(os.listdir(DIR), key=lambda x: int(x.split(".")[0].split("_")[0])):
            if file.endswith(str(map_idx) + ".csv"):
                print(file)
                traj = np.genfromtxt(os.path.join(DIR, file), delimiter=",")
                dists = []
                for i in range(traj.shape[1]):
                    dist = np.sqrt((traj[0, i] - x)**2 + (traj[1, i] - y)**2)
                    dists.append(min(dist))
                if not np.any(np.array(dists) > 6.0):
                    all_dists.append(dists)

        all_maps_dists.append(all_dists)

    plt.figure()

    for j in range(9):
        curr_dists = all_maps_dists[j]
        cmap = plt.cm.get_cmap('coolwarm')
        for i, traj in enumerate(curr_dists):
            clr = cmap(i / (len(curr_dists) - 1))
            plt.plot(curr_dists[i], color=clr, alpha=0.75, linewidth=2.0)

    plt.show()


convergence()
individual_travs()