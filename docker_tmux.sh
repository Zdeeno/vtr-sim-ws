tmux kill-session -t simulator

tmux new-session -d -s "simulator" -n "simulator"

tmux new-window -d -n "ros_tcp"
tmux new-window -d -n "control"
tmux new-window -d -n "pfvtr"
tmux new-window -d -n "simulation"
tmux new-window -d -n "view"


tmux send-keys -t simulator:simulator "source /app/ws/devel/setup.bash" Enter
tmux send-keys -t simulator:simulator "Xvfb :1 &" Enter
tmux send-keys -t simulator:simulator "export DISPLAY=:1" Enter
tmux send-keys -t simulator:simulator "/app/sim_build/hardnav_release_05_ROS1.x86_64" Enter

tmux send-keys -t simulator:ros_tcp "source /app/ws/devel/setup.bash" Enter
tmux send-keys -t simulator:ros_tcp "roscore" Enter
tmux split-window -t simulator:ros_tcp
tmux send-keys -t simulator:ros_tcp "source /app/ws/devel/setup.bash" Enter
tmux send-keys -t simulator:ros_tcp "sleep 2" Enter
tmux send-keys -t simulator:ros_tcp "roslaunch navigation_unity_core core_nodes.launch --wait" Enter

tmux send-keys -t simulator:control "source /app/ws/devel/setup.bash" Enter
tmux send-keys -t simulator:control "sleep 4" Enter
tmux send-keys -t simulator:control "roslaunch navigation_unity_core support.launch --wait" Enter

tmux send-keys -t simulator:pfvtr "source /app/ws/devel/setup.bash" Enter
tmux send-keys -t simulator:pfvtr "sleep 6" Enter
tmux send-keys -t simulator:pfvtr "roslaunch pfvtr repr-sim.launch model_path:='/app/ws/src/pfvtr/src/sensors/backends/siamese/model_tiny.pt' --wait" Enter

tmux send-keys -t simulator:view "source /app/ws/devel/setup.bash" Enter
tmux send-keys -t simulator:view "sleep 6" Enter
tmux send-keys -t simulator:view "rviz -d sim_build/rviz.rviz"

tmux send-keys -t simulator:simulation "source /app/ws/devel/setup.bash" Enter
tmux send-keys -t simulator:simulation "sleep 10" Enter
tmux send-keys -t simulator:simulation "python /app/ws/src/navigation_unity_core/scripts/gym_train_td3.py"
