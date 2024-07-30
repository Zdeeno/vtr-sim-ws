import os
import numpy as np
import re
import matplotlib.pyplot as plt

DIR = os.path.expanduser('~') + "/.ros/trajectory_plots/"


def extract_last_elements(folder_path):
    last_elements = []
    indices = []

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter files to include only CSV files that match the expected pattern
    csv_files = [f for f in files if f.split("_")[-1] == "stats.csv"]
    # print(csv_files)
    # Parse iteration index from file names and sort files based on this index
    csv_files_sorted = sorted(csv_files, key=lambda x: int(x.split('_')[0]))

    for csv_file in csv_files_sorted:
        # Construct the full path to the CSV file
        file_path = os.path.join(folder_path, csv_file)
        file_idx = int(csv_file.split("_")[0])

        # Load the CSV file into a numpy array
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Adjust skiprows if your CSV has headers

        # Get the last element of the array
        last_element = data[-1]

        # Append the last element to the list
        last_elements.append(last_element)
        indices.append(file_idx)

    return last_elements, indices

# Example usage:
# folder_path = 'path/to/your/folder'
# last_elements = extract_last_elements(folder_path)
# print(last_elements)

all_models = ["new_model", "old_model", "pfvtr"]
plt.figure()

for model in all_models:
    last_elements, indices = extract_last_elements(os.path.join(DIR, model))
    plt.scatter(indices, last_elements, marker="x", s=100.0)
    print(model, np.mean(last_elements))

plt.grid()
plt.legend(all_models)
plt.title("avg chamfer distance per traversal")
plt.xlabel("traversal idx")
plt.show()