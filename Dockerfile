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
