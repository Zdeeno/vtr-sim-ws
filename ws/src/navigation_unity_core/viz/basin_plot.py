import seaborn as sb
import os
import numpy as np
import matplotlib.pyplot as plt

DIR = os.path.expanduser('~') + "/.ros/trajectory_plots/pfvtr_basin"
GRID_DIM_SIZE = 5


def get_all_stats(folder_path):
    files = os.listdir(folder_path)
    csv_files = [f for f in files if f.split("_")[-1] == "stats.csv"]
    chamfers = np.zeros(GRID_DIM_SIZE**2)
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        file_idx = int(filename.split("_")[0]) - 1
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Adjust skiprows if your CSV has headers
        chamfers[file_idx] = data[-1]

    return chamfers


def plot_values(data):
    basin_idxs = []
    basin_grid = []
    arr = np.linspace(-2, 2, GRID_DIM_SIZE)
    for x_idx, x in enumerate(arr):
        for y_idx, y in enumerate(arr):
            basin_grid.append((x, y))
            basin_idxs.append((x_idx, y_idx))
    out_mat = np.zeros((GRID_DIM_SIZE, GRID_DIM_SIZE))
    for idx, coord in enumerate(basin_idxs):
        out_mat[coord[0], coord[1]] = data[idx]
    sb.heatmap(out_mat, annot=True, xticklabels=[-2, -1, 0, 1, 2], yticklabels=[-2, -1, 0, 1, 2])
    plt.show()


chamfers = get_all_stats(DIR)
plot_values(chamfers)