"""Perform visual tracking of a robotic arm with a camera."""

from __future__ import division

from imageio import mimsave
from matplotlib import pyplot as plt
import numpy as np


class VisualTracker():
    """
    A class to visually track a robotic arm.

    Initialize an instance of this class (optionally) providing the
    following arguments:
        - focal: float, focal length of the camera in m
        - z_obj: float, z-axis motion depth (object's position)
    """

    def __init__(self, focal=0.03, z_obj=0.8):
        """Initialize a class instance."""
        self._fz_ratio = focal / z_obj
        self._jnt_lengths = [1, 1, 0.6]  # robot's joint lengths
        self._obj_dims = [0.05, 0.1]
        self._x_obj_0 = 1.5
        self._y_obj_0 = 1  # initial object position (global reference frame)
        self._dt = 0.001  # sampling period
        img_jacobian = np.array([  # Image Jacobian
            [self._fz_ratio, 0],
            [0, self._fz_ratio],
            [self._fz_ratio, 0],
            [0, self._fz_ratio],
            [self._fz_ratio, 0],
            [0, self._fz_ratio],
            [self._fz_ratio, 0],
            [0, self._fz_ratio],
            [self._fz_ratio, 0],
            [0, self._fz_ratio]
        ])
        self._img_jacobian_psinv = np.matmul(  # pseudo-inverse
            np.linalg.inv(np.matmul(img_jacobian.T, img_jacobian)),
            img_jacobian.T
        )

    def simulate(self, x_obj_f=1.5, y_obj_f=0,
                 theta_init=-np.pi / 4, theta_final=-3 * np.pi / 4,
                 t_sim=5, kappa_c=5, kappa_s=0.2,
                 gif_name=None, gif_fps=3, plot_fps=3):
        """
        Simulate the tracking of the object's angle and position.

        Inputs:
            - x_obj_f: float, object's final x coordinate
            - y_obj_f: float, object's final y coordinate
            - theta_init: float, object's length and x-axis init. angle
            - theta_final: float, object's length and x-axis final angle
            - t_sim: float (positive), total simulation duration in secs
            - kappa_c: float, control gain for angle tracking
            - kappa_s: float, control gain for position tracking
            - gif_name: str or None, the name of the .gif file to create
                showing the arm's movement. If None, no file is created.
            - gif_fps: int, frames per second on the saved .gif file,
                ignored if gif_fps is None.
            - plot_fps: int, frames per second on when plotting
        Returns:
            - self: VisualTracker object
        """
        # Initialize parameters
        obj_coords = self._sim_init(
            x_obj_f, y_obj_f, theta_init, theta_final, t_sim,
            kappa_c, kappa_s, gif_name, gif_fps, plot_fps
        )

        # Main simulation
        for t_step in range(int(self._t_sim / self._dt) + 1):

            # Object motion
            obj_coords = self._object_step(obj_coords)

            # Robot Kinematic Equation
            self._compute_jacobian()

            # Forward kinematics
            self._forward_kinematics_step(t_step)

            # Update q considering both control subtasks
            self._q += (
                self._visual_servo_control(t_step, obj_coords)
                + self._relative_angle_control(obj_coords)
            )

            # Save Plot Data
            self._obj_coords_plot[t_step] = obj_coords

        # Plot simulation
        config_plots = self._plot_arm()
        self._plot_camera_view()
        if self._gif_name is not None:
            mimsave(gif_name, config_plots, fps=self._gif_fps)
        return self

    def _sim_init(self, x_obj_f=1.5, y_obj_f=0,
                  theta_init=-np.pi / 4, theta_final=-3 * np.pi / 4,
                  t_sim=5, kappa_c=5, kappa_s=0.2,
                  gif_name=None, gif_fps=10, plot_fps=3):
        """Initialize simulation parameters and variables."""
        self._x_obj_f = x_obj_f
        self._y_obj_f = y_obj_f
        self._theta_init = theta_init
        self._theta_final = theta_final
        self._t_sim = t_sim
        self._kappa_c = kappa_c
        self._kappa_s = kappa_s
        self._gif_name = gif_name
        self._gif_fps = gif_fps
        self._plot_fps = plot_fps
        self._q = self._inverse_kinematics()
        obj_coords, self._speed, self._rot_speed, self._feat_vec_desired = \
            self._object_kinematics_params()
        self._theta_desired = self._theta_init

        # Main simulation variables
        size = int(self._t_sim / self._dt) + 1
        self._x_1 = np.zeros((size,))
        self._y_1 = np.zeros((size,))
        self._x_2 = np.zeros((size,))
        self._y_2 = np.zeros((size,))
        self._x_e = np.zeros((size,))
        self._y_e = np.zeros((size,))
        self._feat_vec = np.zeros((size, 2, 5))
        self._obj_coords_plot = np.zeros((size, 2, 5))
        return obj_coords

    def _inverse_kinematics(self):
        """Initialize robot configuration - Inverse kinematics."""
        q_2 = np.arccos(
            (
                (self._x_obj_0 - self._jnt_lengths[2]) ** 2
                + self._y_obj_0 ** 2
                - self._jnt_lengths[0] ** 2
                - self._jnt_lengths[1] ** 2
            )
            / (2 * self._jnt_lengths[0] * self._jnt_lengths[1])
        )
        psi = np.arcsin(
            self._jnt_lengths[1] * np.sin(q_2)
            / np.sqrt(
                (self._x_obj_0 - self._jnt_lengths[2]) ** 2
                + self._y_obj_0 ** 2
            )
        )
        q_1 = (np.arctan2(self._x_obj_0 - self._jnt_lengths[2], self._y_obj_0)
               - psi)
        q_3 = np.pi / 2 - q_1 - q_2
        return np.array([q_1, q_2, q_3])

    def _object_kinematics_params(self):
        """Compute the parameters of object's kinematics."""
        obj_length, obj_width = self._obj_dims
        # Initial object position w.r.t its center
        obj_coords = np.matmul(  # (2, 5) array of x-y coords of five points
            np.array([  # rotational matrix
                [np.cos(self._theta_init), np.sin(self._theta_init)],
                [-np.sin(self._theta_init), np.cos(self._theta_init)]
            ]),
            0.5 * np.array([  # relative postion matrix
                [0, obj_length, obj_length, -obj_length, -obj_length],
                [0, obj_width, -obj_width, -obj_width, obj_width]
            ])
        )
        feat_vec_desired = obj_coords * self._fz_ratio

        # Global initial object position
        obj_coords += np.array([[self._x_obj_0], [self._y_obj_0]])
        speed = np.array([
            [(self._x_obj_f - self._x_obj_0) / self._t_sim],
            [(self._y_obj_f - self._y_obj_0) / self._t_sim]
        ])
        rot_speed = (self._theta_final - self._theta_init) / self._t_sim
        return obj_coords, speed, rot_speed, feat_vec_desired

    def _object_step(self, obj_coords):
        """Move the object a step of time ahead."""
        return obj_coords + self._dt * (
            self._speed
            + np.vstack([
                [-self._rot_speed * (obj_coords[1, :] - obj_coords[1, 0])],
                [self._rot_speed * (obj_coords[0, :] - obj_coords[0, 0])]
            ])
        )

    def _compute_jacobian(self):
        """Compute sines-cosines and jacobian matrix."""
        q_sum = np.cumsum(self._q)
        self._sines = np.sin(q_sum)
        self._cosines = np.cos(q_sum)
        (s_1, s_12, s_123) = self._sines
        (c_1, c_12, c_123) = self._cosines
        self._jacobian = np.array([
            np.cumsum([
                self._jnt_lengths[2] * c_123,
                self._jnt_lengths[1] * c_12,
                self._jnt_lengths[0] * c_1
            ])[::-1],  # compute jacobian 1st row
            np.cumsum([
                -self._jnt_lengths[2] * s_123,
                -self._jnt_lengths[1] * s_12,
                -self._jnt_lengths[0] * s_1
            ])[::-1]  # jacobian 2nd row
        ])
        self._jacobian_psinv = np.matmul(
            self._jacobian.T,
            np.linalg.inv(np.matmul(self._jacobian, self._jacobian.T))
        )

    def _forward_kinematics_step(self, t_step):
        """Move the robot a step of time ahead."""
        (s_1, s_12, s_123) = self._sines
        (c_1, c_12, c_123) = self._cosines
        self._x_1[t_step] = self._jnt_lengths[0] * s_1
        self._y_1[t_step] = self._jnt_lengths[0] * c_1
        self._x_2[t_step] = self._x_1[t_step] + self._jnt_lengths[1] * s_12
        self._y_2[t_step] = self._y_1[t_step] + self._jnt_lengths[1] * c_12
        self._x_e[t_step] = self._x_2[t_step] + self._jnt_lengths[2] * s_123
        self._y_e[t_step] = self._y_2[t_step] + self._jnt_lengths[2] * c_123

    def _visual_servo_control(self, t_step, obj_coords):
        """Control the arm's motion to follow the object's motion."""
        # Global frame (0) w.r.t. camera (cam)
        # (Position of Camera == Position of End_Effector)
        (_, _, s_123) = self._sines
        (_, _, c_123) = self._cosines
        rot_cam_0 = np.array([[s_123, c_123], [-c_123, s_123]])

        # Object coordinates w.r.t. to camera reference frame
        obj_coords_cam = np.matmul(
            rot_cam_0,
            obj_coords - np.vstack((self._x_e[t_step], self._y_e[t_step]))
        )
        # What camera sees
        self._feat_vec[t_step] = obj_coords_cam * self._fz_ratio
        delta_f = (self._feat_vec_desired - self._feat_vec[t_step]).flatten()

        # Visual Servo Control (task 1)
        u_ctrl = -self._kappa_s * np.matmul(self._img_jacobian_psinv, delta_f)
        return np.matmul(self._jacobian_psinv, u_ctrl)

    def _relative_angle_control(self, obj_coords):
        """Control the arm's motion to preserve relative angle."""
        (_, _, s_123) = self._sines
        (_, _, c_123) = self._cosines
        norm_cos_theta = (
            (obj_coords[:, 1] - obj_coords[:, 4])
            / np.sqrt(
                (obj_coords[0, 1] - obj_coords[0, 4]) ** 2
                + (obj_coords[1, 1] - obj_coords[1, 4]) ** 2
            )
        )
        cos_theta = np.dot(np.array([s_123, c_123]), norm_cos_theta)
        dcos_theta = np.dot(np.array([c_123, -s_123]), norm_cos_theta)
        dqr = -dcos_theta * (cos_theta - np.cos(self._theta_desired))
        return (
            self._kappa_c
            * np.dot(
                (np.eye(3) - np.matmul(self._jacobian_psinv, self._jacobian)),
                dqr * np.ones((3,))
            )
        )

    def _plot_arm(self):
        """Plot successive robot configurations."""
        fig, axs = plt.subplots()
        fig.show()
        axs.cla()
        axs.axis([-1, 2.5, -1, 2.5])
        axs.plot([0], [0], 'o')
        config_plots = []
        for t_step in range(0, int(self._t_sim / self._dt) + 1, 1000):
            axs.plot([0, self._x_1[t_step]], [0, self._y_1[t_step]])
            axs.plot(self._x_1[t_step], self._y_1[t_step], 'o')
            axs.plot(
                [self._x_1[t_step], self._x_2[t_step]],
                [self._y_1[t_step], self._y_2[t_step]]
            )
            axs.plot(self._x_2[t_step], self._y_2[t_step], 'o')
            axs.plot(
                [self._x_2[t_step], self._x_e[t_step]],
                [self._y_2[t_step], self._y_e[t_step]]
            )
            axs.plot(self._x_e[t_step], self._y_e[t_step], 'ro')
            axs.plot(
                self._obj_coords_plot[t_step, 0, 0],
                self._obj_coords_plot[t_step, 1, 0], 'g+')
            axs.plot(
                self._obj_coords_plot[t_step, 0, 1],
                self._obj_coords_plot[t_step, 1, 1], 'g.')
            axs.plot(
                self._obj_coords_plot[t_step, 0, 2],
                self._obj_coords_plot[t_step, 1, 2], 'g.')
            axs.plot(
                self._obj_coords_plot[t_step, 0, 3],
                self._obj_coords_plot[t_step, 1, 3], 'g.')
            axs.plot(
                self._obj_coords_plot[t_step, 0, 4],
                self._obj_coords_plot[t_step, 1, 4], 'g.')
            plt.axis('off')
            plt.pause(1 / self._plot_fps)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            config_plots.append(image.reshape(
                fig.canvas.get_width_height()[::-1] + (3, )))

        # Draw and create image
        return config_plots

    def _plot_camera_view(self):
        """Plot the object points as seen by the camera."""
        fig, axs = plt.subplots()
        fig.show()
        axs.cla()
        axs.axis([-0.003, 0.003, -0.003, 0.003])
        axs.grid()
        axs.plot([0], [0], 'r+')
        for t_step in range(0, int(self._t_sim / self._dt) + 1, 250):
            axs.plot(
                self._feat_vec[t_step, 0, 0],
                self._feat_vec[t_step, 1, 0], 'ro')
            axs.plot(
                self._feat_vec[t_step, 0, 1],
                self._feat_vec[t_step, 1, 1], 'bo')
            axs.plot(
                self._feat_vec[t_step, 0, 2],
                self._feat_vec[t_step, 1, 2], 'yo')
            axs.plot(
                self._feat_vec[t_step, 0, 3],
                self._feat_vec[t_step, 1, 3], 'go')
            axs.plot(
                self._feat_vec[t_step, 0, 4],
                self._feat_vec[t_step, 1, 4], 'ro')
            plt.pause(1 / self._plot_fps)

if __name__ == "__main__":
    VisualTracker().simulate(gif_name='robot_visual.gif')
