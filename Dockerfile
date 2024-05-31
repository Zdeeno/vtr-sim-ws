FROM rwthika/ros-torch:noetic-desktop-full-torch2.0.1-py

RUN apt-get update
RUN apt-get install -y ros-noetic-jackal-simulator
RUN apt-get install -y ros-noetic-jackal-desktop
RUN apt-get install -y ros-noetic-jackal-navigation
RUN apt-get install -y tmux
RUN apt-get install -y xvfb

RUN pip install numpy==1.22
RUN pip install rosnumpy
RUN pip install torchvision
RUN pip install torchrl
RUN pip install tensordict
RUN pip install scipy
RUN pip install tqdm
RUN pip install wandb

RUN mkdir /app
RUN mkdir /root/.ros
RUN mkdir /root/.ros/trajectory_plots
RUN mkdir /root/.ros/trajectory_plots_eval
RUN mkdir /root/.ros/models
COPY . /app

COPY ./maps/sim1_vtr /root/.ros/sim1_vtr
COPY ./maps/sim2_vtr /root/.ros/sim2_vtr
COPY ./maps/sim3_vtr /root/.ros/sim3_vtr
COPY ./maps/sim4_vtr /root/.ros/sim4_vtr
COPY ./maps/sim5_vtr /root/.ros/sim5_vtr
COPY ./maps/sim6_vtr /root/.ros/sim6_vtr
COPY ./maps/sim7_vtr /root/.ros/sim7_vtr
COPY ./maps/sim8_vtr /root/.ros/sim8_vtr
COPY ./maps/sim9_vtr /root/.ros/sim9_vtr
COPY ./maps/sim10_vtr /root/.ros/sim10_vtr

WORKDIR /app/ws
RUN catkin clean -y
RUN source /opt/ros/noetic/setup.bash && catkin b

WORKDIR /app
