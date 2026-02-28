# uav-2526

This project runs on Python 3.10 or 3.11 with a Raspberry Pi 5 and a Pixhawk 6X. It is fully automated and does not require GPS. 

* Vision is handled by the Luxonis OAK-D SD2 v3 stereo camera, which provides stereo and RGB imaging and integrates seamlessly with the DepthAI SDK for Python.
* The system also uses an Akida BrainChip for low-power, event-based AI computation.


### Running the ArUco detection
1. Clone the repository:

    `git clone https://github.com/Iracema008/uav-2526.git`
2. Set up the environment:

    `source helper.sh`
   
    `init_venv`
4. Run the main program:

    `run`


### Running the Simulator

Before starting, make sure you have everything set up:  

### Prerequisites

1. **ArduPilot**  
   - Follow the [ArduPilot build setup guide](https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux) to install and configure ArduPilot.  
   - **Note:** When building with `waf`, select the correct autopilot board. We use **PIXHAWK1** for our simulations.

2. **QGroundControl**  
   - QGroundControl is included in the repository. On macOS, ensure the app has execution permissions:

     ```bash
     chmod +x QGroundControl.AppImage
     ```


### Luxonis Camera Troubleshooting
If you run into this error: 
 "[warning] Insufficient permissions to communicate with X_LINK_UNBOOTED device with name "3.1". Make sure udev rules are set...” 

1.
   `echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules`
3.
   `sudo udevadm control --reload-rules && sudo udevadm trigger`
