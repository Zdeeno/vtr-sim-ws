import seaborn as sb
import os
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


NAME = "bearnav_2"
DIRS = [os.path.expanduser('~') + "/.ros/trajectory_plots/history/eval1/eval_" + NAME,
        os.path.expanduser('~') + "/.ros/trajectory_plots/history/eval2/eval_" + NAME,
        os.path.expanduser('~') + "/.ros/trajectory_plots/history/eval3/eval_" + NAME]
GRID_DIM_SIZE = 10
DISPLACEMENT = 3
MAX_DIST = 5.0


def get_all_stats(folder_path):
    files = os.listdir(folder_path)
    csv_files = [f for f in files if f.split("_")[-1] == "stats.csv"]
    chamfers = np.ones(GRID_DIM_SIZE**2) * MAX_DIST
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        file_idx = int(filename.split("_")[0]) - 1
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Adjust skiprows if your CSV has headers
        chamfers[file_idx] = data[-1]

    return chamfers


def get_matrix(data):
    basin_idxs = []
    basin_grid = []
    arr = np.around(np.linspace(-DISPLACEMENT, DISPLACEMENT, GRID_DIM_SIZE), decimals=2)
    for x_idx, x in enumerate(arr):
        for y_idx, y in enumerate(arr):
            basin_grid.append((x, y))
            basin_idxs.append((x_idx, y_idx))
    out_mat = np.zeros((GRID_DIM_SIZE, GRID_DIM_SIZE))
    for idx, coord in enumerate(basin_idxs):
        out_mat[coord[0], coord[1]] = min(MAX_DIST, data[idx])
    return out_mat


def plot_heatmap(matrix, save=False):
    arr = np.around(np.linspace(-DISPLACEMENT, DISPLACEMENT, GRID_DIM_SIZE), decimals=2)
    sb.heatmap(matrix, annot=True, xticklabels=arr, yticklabels=arr, vmax=MAX_DIST, vmin=0.0)
    # plt.show()
    if save:
        matplotlib.pyplot.savefig("./bearnav/" + "basin_" + NAME + ".pdf")

mat = None
for curr_dir in DIRS:
    chamfers = get_all_stats(curr_dir)
    matrix = get_matrix(chamfers)
    # plot_heatmap(mat)
    if mat is not None:
        mat += matrix
    else:
        mat = matrix

mat /= float(len(DIRS))
plot_heatmap(mat, save=True)
print(np.mean(mat))
