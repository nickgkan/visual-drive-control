# Visual Driving of a Robotic Arm

Tested with Python3.5

## Goal
Given a camera adapted on the robot's effector, track a moving object (rectangle) so as both relative angle and position remain constant.

## Solution
We design a control law that ensures minimum error between the measured and the desired angle and position.

## Results
<p align="center">
  <img src="https://github.com/nickgkan/visual-drive-control/blob/master/robot_visual.gif?raw=true"/>
</p>

## Next step?
The effector tracks the moving object without estimating its next position and angle, resulting to a constant error. This could be alleviated with the use of a PID controller or Kalman filter.
