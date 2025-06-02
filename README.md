## Overview

This project is a physical solver for **GamePigeon Cup Pong**. The system captures the mirrored iOS screen on a computer, uses OpenCV to analyze the game state, and determines the optimal shot target. Based on a pre-calibrated mapping between screen coordinates and the XY plotter's workspace, the software calculates the precise physical movement required. These movement commands are then sent serially to an Arduino microcontroller, which controls a XY plotter to physically interact with the mirrored screen using a stylus.

## Features

* **Real-time Screen Analysis:** Processes live video feed of the game screen using OpenCV.
* **Dynamic Cup Detection:** Identifies cup positions and their state (e.g., remaining cups).
* **Coordinate Transformation:** Maps screen pixels to physical coordinates for the plotter.
* **Precise Movement Control:** Generates commands for accurate XY plotter movement.
* **Serial Communication:** Reliable data transfer between the computer and Arduino.
* **Calibration Process:** A necessary initial step to align screen coordinates with the plotter's physical workspace.

## Technologies Used

* **Python 3.x:** The primary programming language.
* **OpenCV (`opencv-python`):** For image processing, screen analysis, and cup detection.
* **PySerial (`pyserial`):** For serial communication with the Arduino.
* **PyYAML (`PyYAML`):** Used for storing configuration and calibration data.
* **Arduino:** Microcontroller for controlling the XY plotter.
* **XY Plotter:** To physically interact with the screen.
* **Screen Mirroring Software:** (e.g., QuickTime Player) Used to display the iOS screen on the computer.

## How It Works

1. **Screen Capture and Preprocessing:** The Python script continuously captures frames from the mirrored iOS screen. These frames are processed using OpenCV. Typical preprocessing steps might include converting the image to a different color space (e.g., HSV) to make color-based detection easier, and potentially applying filters to reduce noise.
2. **Cup Detection:** The script uses **template matching** to locate the cups. A predefined image of a cup (`cup.png`) is used as a template. The `cv2.matchTemplate()` function is applied across the captured screen frame to find areas that are similar to the template. To improve robustness, **multi-scale template matching** is used, searching for cups at slightly different sizes. **Non-maximum suppression** is then applied to the detection results to eliminate overlapping or redundant bounding boxes and identify the most likely locations of the cups. The script identifies the top-left corner coordinates and dimensions of the bounding box for each detected cup.
3. **Coordinate Mapping and Movement Lookup:** Instead of a dynamic coordinate transformation, the script uses a predefined set of **known cup screen positions** (`known_cups`) and a corresponding mapping of **movements** (`cup_moves`) for each position. When a cup is detected, the script finds the closest known cup position to the detected position using the `match_to_cup` function. The index of this closest known position is then used to look up the specific `(dx, dy)` movement required for the XY plotter from the `cup_moves` dictionary. This approach relies on prior calibration to define these known positions and their associated plotter movements. The output is the predefined `(dx, dy)` movement pair for the XY plotter.
4. **Movement Command Generation:** Based on the current known position of the plotter's stylus and the calculated target (X, Y) position in the plotter's coordinate system, the script determines the necessary movement. This involves calculating the distance and direction of travel for the X and Y axes.
5. **Serial Communication Protocol:** The calculated movement commands are formatted into a specific string or byte sequence that the Arduino firmware is designed to interpret. The `pyserial` library is used to send this formatted command over the serial port connected to the Arduino.
6. **Arduino Control:** The Arduino sketch (`xyPlotter.ino`) listens for commands on its serial port. Upon receiving a valid movement command, the Arduino parses it, calculates the required steps for its stepper motors, and controls the motor drivers to move the plotter arm to the target physical coordinates. The firmware handles acceleration, deceleration, and maintaining position.
