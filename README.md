# uav-2526

Requirements

- pip install ultralytics
- pip install opencv-python
- pip install depthai
- pip install rich

### Running the ArUco detection
1. Ensure git lfs is installed and initialized:

    https://git-lfs.com/
2. Clone the repository:

    `git clone https://github.com/Iracema008/auto-uav.git && cd uav-2526`
3. Change the working directory:

    `cd vision`
4. Run the command:

    `source helper.sh`
5. Run the main program:

    `run`


### Running the Simulator
#### Ensure that the following prerequisites are met:
1. Install ArduPilot:
    - [Set up your ArduPilot Build Environment](https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux)
      - **NOTE**: Select the correct autopilot board when building with waf. We are using **PIXHAWK1**
2. QGroundControl is already downloaded to the repository.
#### To run the simulator:
1. Clone the repository: 

    `git clone https://github.com/Iracema008/auto-uav.git`

2. In the root directory, run the command:

    `sim_vehicle.py -v ArduCopter --out="localhost:14550" --out="localhost:14551"`
3. Run QGroundControl:

    `./QGroundControl.AppImage`

### Running the Camera
1. 
    `cd depthai-python/ examples`
2. 
    `python3 ColorCamera/rgb_preview.py`


### Troubleshooting

Please ensure that you grant camera permissions to the IDE you are using if you are running this on a Mac.

### Luxonis Camera Troubleshooting

If you run into this error: 
 "[warning] Insufficient permissions to communicate with X_LINK_UNBOOTED device with name "3.1". Make sure udev rules are set...” 


1.
   `echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules`
3.
   `sudo udevadm control --reload-rules && sudo udevadm trigger`
