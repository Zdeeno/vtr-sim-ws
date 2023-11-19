tmux kill-session -t simulator

tmux new-session -d -s "simulator" -n "simulator"

tmux new-window -d -n "ros_tcp"
tmux new-window -d -n "control"
tmux new-window -d -n "world"
tmux new-window -d -n "view"


x=$(echo $SHELL | sed 's:.*/::')

tmux send-keys -t simulator:simulator "source ws/devel/setup.$x" Enter
tmux send-keys -t simulator:simulator "build/hardnav_release_04_ROS1.x86_64" Enter

tmux send-keys -t simulator:ros_tcp "source ws/devel/setup.$x" Enter
tmux send-keys -t simulator:ros_tcp "roslaunch navigation_unity_core core_nodes.launch" Enter

tmux send-keys -t simulator:control "source ws/devel/setup.$x" Enter
tmux send-keys -t simulator:control "roslaunch navigation_unity_core support.launch"

tmux send-keys -t simulator:world "source ws/devel/setup.$x" Enter
tmux send-keys -t simulator:world "cd ws/src/navigation_unity_core/scripts" Enter
tmux send-keys -t simulator:world "python default_world_loader.py"

tmux send-keys -t simulator:view "source ws/devel/setup.$x" Enter
tmux send-keys -t simulator:view "rqt_image_view"
