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
RUN mkdir ~/.ros
RUN mkdir ~/.ros/trajectory_plots
RUN mkdir ~/.ros/trajectory_plots_eval
RUN mkdir ~/.ros/models
COPY . /app

COPY ./maps/sim1_vtr ~/.ros
COPY ./maps/sim2_vtr ~/.ros
COPY ./maps/sim3_vtr ~/.ros
COPY ./maps/sim4_vtr ~/.ros
COPY ./maps/sim5_vtr ~/.ros
COPY ./maps/sim6_vtr ~/.ros
COPY ./maps/sim7_vtr ~/.ros
COPY ./maps/sim8_vtr ~/.ros
COPY ./maps/sim9_vtr ~/.ros
COPY ./maps/sim10_vtr ~/.ros

WORKDIR /app/ws
RUN catkin clean -y
RUN source /opt/ros/noetic/setup.bash && catkin b

WORKDIR /app