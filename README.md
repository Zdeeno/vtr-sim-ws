# Visual Teach and Repeat Simulator Workspace
Workspace for unity based simulator of VTR. \
The simulator used for this project is from [this](https://github.com/MrTomzor/navigation_unity_testbed) repository. \
All other dependecies are added via submodules or available through apt (see src folder).

## Initialization
- Clone this repository and don't forget to fetch the submodules `git submodule init && git submodule update`.
- Download built binaries [here](https://github.com/MrTomzor/navigation_unity_testbed) (see readme) and put it into the `sim_build` folder.
- Build the workspace using `cd ws && catkin b`
- Use the script `sim_tmux.sh` to start the simulation. Optionally run the `support.launch` to resize camera, publish odometry and enable control (with attached ps4 controller).
