import rospy
import roslaunch
import tf
from geometry_msgs.msg import TransformStamped, WrenchStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
import time
import math
import numpy as np
from core.robot import Robot
from typing import Dict
from liecasadi import SO3
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
from copy import deepcopy
import utils_muav
import os


class Visualize_robot:
    def __init__(self, robot: Robot, fps: int = 31, plot_control_input: bool = True, print_time: bool = True):
        self.robot_param = {}
        self.robot_param["name"] = robot.name
        self.robot_param["fullpath_model"] = robot.fullpath_model
        self.robot_param["joint_list"] = robot.joint_list
        self.robot_param["aero_frame_list"] = robot.aero_frame_list
        self.robot_param["propellers_frame_list"] = robot.propellers_frame_list
        self.robot_param["ndofs"] = robot.ndofs
        self.robot_param["naero"] = robot.naero
        self.robot_param["nprop"] = robot.nprop
        self.robot_param["limits"] = robot.limits
        self.robot_param["world"] = robot.world
        self.robot_param["root_link"] = robot.root_link
        self.robot_param["joint_pos"] = {"ub": robot.limits["joint_pos"]["ub"], "lb": robot.limits["joint_pos"]["lb"]}
        self.robot_param["prop_thrust"] = {
            "ub": robot.limits["prop_thrust"]["ub"],
            "lb": robot.limits["prop_thrust"]["lb"],
        }
        self.config = {}
        self.config["propeller"] = {}
        self.config["propeller"]["max_amplitude"] = 0.5

        self.time_start = 0
        self.mk_id = 0
        self.ros_stamp = 0
        self.fps = fps
        self.b = {}

        self.__plot_control_input = plot_control_input
        self.__print_time = print_time

        self.__run_ros_launch_file()
        self.__set_publishers()

    def __run_ros_launch_file(self, name_launch_file: str = "move_and_display.launch"):
        repo_tree = utils_muav.get_repository_tree()
        # rospy.init_node('en_Mapping', anonymous=True)
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        cli_args = [
            os.path.join(repo_tree["ros_launch"], name_launch_file),
            f"model:='{self.robot_param['fullpath_model']}.urdf'",
        ]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        self.__launch_rviz_obj = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
        self.__launch_rviz_obj.start()
        rospy.loginfo("started")

    def close_rviz(self):
        self.__launch_rviz_obj.shutdown()

    def __update_ros_timestamp(self):
        self.ros_stamp = rospy.Time.now()

    def __set_publishers(self):
        rospy.init_node("dummy_robot_ros_interface_py")
        self.__update_ros_timestamp()
        # robot pose
        self.robot_tf_broadcaster = tf.TransformBroadcaster(queue_size=10)
        self.pose_state_obj = TransformStamped()
        self.pose_state_obj.header.frame_id = self.robot_param["world"]
        self.pose_state_obj.child_frame_id = self.robot_param["root_link"]
        # robot joint
        self.joint_state_pub = rospy.Publisher("joint_states", JointState, queue_size=10)
        self.joint_state_obj = JointState()
        self.joint_state_obj.header = Header()
        self.joint_state_obj.name = self.robot_param["joint_list"]
        # propellers
        self.propeller_state_pub = rospy.Publisher("visualization_propellers", MarkerArray, queue_size=10)
        self.propeller_state_obj = MarkerArray()
        for prop_frame in self.robot_param["propellers_frame_list"]:
            self.propeller_state_obj.markers.append(self.__get_marker_prop(prop_frame))
        self.propeller_state_pub.publish(self.propeller_state_obj)
        # aerodynamic
        self.aerodynamics_state_pub = rospy.Publisher("visualization_aerodynamics", WrenchStamped, queue_size=10)
        self.aerodynamics_state_obj = self.__get_empty_aero_wrench()
        self.aerodynamics_state_pub.publish(self.aerodynamics_state_obj)
        # twist
        self.base_twist_state_pub = rospy.Publisher("visualization_twist", TwistStamped, queue_size=10)
        self.base_twist_state_obj = self.__get_empty_base_twist()
        self.base_twist_state_pub.publish(self.base_twist_state_obj)
        # goal
        self.goal_pub = rospy.Publisher("visualization_goal", MarkerArray, queue_size=10)
        self.goal_obj = MarkerArray()
        # obstacle
        self.obstacle_pub = rospy.Publisher("visualization_obstacle", MarkerArray, queue_size=10)
        self.obstacle_obj = MarkerArray()

    def __assign_id(self):
        self.mk_id = self.mk_id + 1
        return self.mk_id

    def __get_marker_prop(self, prop_frame):
        marker = Marker()
        marker.header.frame_id = prop_frame
        marker.header.stamp = self.ros_stamp
        marker.type = 0
        marker.id = self.__assign_id()
        # Set the scale of the marker
        marker.scale.x = 0.2
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        # Set the color
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 1.0
        # Set the pose of the marker
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = -0.7071068
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 0.7071068
        return marker

    def __get_empty_aero_wrench(self):
        wrench = WrenchStamped()
        wrench.header.stamp = self.ros_stamp
        wrench.header.frame_id = self.robot_param["root_link"]
        wrench.wrench.force.x = 0
        wrench.wrench.force.y = 0
        wrench.wrench.force.z = 0
        wrench.wrench.torque.x = 0
        wrench.wrench.torque.y = 0
        wrench.wrench.torque.z = 0
        return wrench

    def __get_empty_base_twist(self):
        twist = TwistStamped()
        twist.header.stamp = self.ros_stamp
        twist.header.frame_id = self.robot_param["root_link"]
        twist.twist.linear.x = 0
        twist.twist.linear.y = 0
        twist.twist.linear.z = 0
        twist.twist.angular.x = 0
        twist.twist.angular.y = 0
        twist.twist.angular.z = 0
        return twist

    def __publish_joint_state(self, joint_values):
        self.joint_state_obj.header.stamp = self.ros_stamp
        self.joint_state_obj.position = joint_values
        self.joint_state_obj.velocity = []
        self.joint_state_obj.effort = []
        self.joint_state_pub.publish(self.joint_state_obj)

    def __publish_base_pose(self, base_state):
        self.pose_state_obj.transform.translation.x = base_state[0]
        self.pose_state_obj.transform.translation.y = base_state[1]
        self.pose_state_obj.transform.translation.z = base_state[2]
        self.pose_state_obj.transform.rotation.x = base_state[3]
        self.pose_state_obj.transform.rotation.y = base_state[4]
        self.pose_state_obj.transform.rotation.z = base_state[5]
        self.pose_state_obj.transform.rotation.w = base_state[6]
        self.pose_state_obj.header.stamp = self.ros_stamp
        self.robot_tf_broadcaster.sendTransform(
            (
                self.pose_state_obj.transform.translation.x,
                self.pose_state_obj.transform.translation.y,
                self.pose_state_obj.transform.translation.z,
            ),
            (
                self.pose_state_obj.transform.rotation.x,
                self.pose_state_obj.transform.rotation.y,
                self.pose_state_obj.transform.rotation.z,
                self.pose_state_obj.transform.rotation.w,
            ),
            self.ros_stamp,
            self.pose_state_obj.child_frame_id,
            self.pose_state_obj.header.frame_id,
        )

    def __publish_propellers(self, prop_values):
        if self.robot_param["limits"]["prop_thrust"]["ub"] is None:
            self.robot_param["limits"]["prop_thrust"]["ub"] = np.ones((self.robot_param["nprop"], 1))
        for i, marker in enumerate(self.propeller_state_obj.markers):
            marker.header.stamp = self.ros_stamp
            marker.scale.x = (
                self.config["propeller"]["max_amplitude"]
                * prop_values[i]
                / self.robot_param["limits"]["prop_thrust"]["ub"][i]
            )
        self.propeller_state_pub.publish(self.propeller_state_obj)

    def __publish_aerodynamics(self, aero_values, base_state):
        sumAeroForce_w = aero_values.T.reshape((-1, 6)).sum(axis=0)[:3]
        quat_w_b = base_state[3:]
        rotm_b_w = SO3(quat_w_b).as_matrix().full().T
        sumAeroForce_b = rotm_b_w @ sumAeroForce_w
        self.aerodynamics_state_obj.wrench.force.x = sumAeroForce_b[0]
        self.aerodynamics_state_obj.wrench.force.y = sumAeroForce_b[1]
        self.aerodynamics_state_obj.wrench.force.z = sumAeroForce_b[2]
        self.aerodynamics_state_obj.header.stamp = self.ros_stamp
        self.aerodynamics_state_pub.publish(self.aerodynamics_state_obj)

    def __publish_twist(self, base_twist_values, base_state):
        linVel_w = base_twist_values[:3]
        quat_w_b = base_state[3:]
        rotm_b_w = SO3(quat_w_b).as_matrix().full().T
        linVel_b = rotm_b_w @ linVel_w
        self.base_twist_state_obj.twist.linear.x = linVel_b[0]
        self.base_twist_state_obj.twist.linear.y = linVel_b[1]
        self.base_twist_state_obj.twist.linear.z = linVel_b[2]
        self.base_twist_state_obj.header.stamp = self.ros_stamp
        self.base_twist_state_pub.publish(self.base_twist_state_obj)

    def __get_marker_goal(self, xyz: np.ndarray):
        marker = Marker()
        marker.header.frame_id = self.robot_param["world"]
        marker.header.stamp = self.ros_stamp
        marker.type = 2
        marker.id = self.__assign_id()
        # Set the scale of the marker
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        # Set the color
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.color.a = 0.2
        # Set the pose of the marker
        marker.pose.position.x = xyz[0]
        marker.pose.position.y = xyz[1]
        marker.pose.position.z = xyz[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        return marker

    def __get_marker_obstacle(self, obstacle: "muav.trajectory.Obstacle"):
        marker = Marker()
        marker.header.frame_id = self.robot_param["world"]
        marker.header.stamp = self.ros_stamp
        marker.type = obstacle.type_rviz
        marker.id = self.__assign_id()
        # Set the color
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 0.2
        # Set the pose of the marker
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        if marker.type == 2:  # sphere
            marker.scale.x = obstacle.r * 2
            marker.scale.y = obstacle.r * 2
            marker.scale.z = obstacle.r * 2
            marker.pose.position.x = obstacle.xyz[0]
            marker.pose.position.y = obstacle.xyz[1]
            marker.pose.position.z = obstacle.xyz[2]
        if marker.type == 3:  # cylinder
            marker.scale.x = obstacle.r * 2
            marker.scale.y = obstacle.r * 2
            marker.scale.z = 100
            marker.pose.position.x = obstacle.xy[0]
            marker.pose.position.y = obstacle.xy[1]
            marker.pose.position.z = 0
        return marker

    def set_goals(self, goals: list):
        for goal in goals:
            print(f"viz: goal {goal.describe()} added")
            self.goal_obj.markers.append(self.__get_marker_goal(goal.position["value"]))
        self.goal_pub.publish(self.goal_obj)

    def set_obstacles(self, obstacles: list):
        for obstacle in obstacles:
            print("viz: obstacle {} added".format(obstacle))
            if obstacle.type_rviz is not None:
                self.obstacle_obj.markers.append(self.__get_marker_obstacle(obstacle))
        self.obstacle_pub.publish(self.obstacle_obj)

    def __interp_all(self, t, base_state, base_twist_values, joint_values, prop_values, aero_values):
        t_int = np.arange(t[0], t[-1], 1 / self.fps)

        base_twist_values = utils_muav.interp_2d(t_int, t, base_twist_values)
        joint_values = utils_muav.interp_2d(t_int, t, joint_values)
        prop_values = utils_muav.interp_2d(t_int, t, prop_values)
        if aero_values is not None:
            aero_values = utils_muav.interp_2d(t_int, t, aero_values)

        pos_state = base_state[:3, :]
        quat_state = base_state[3:, :]
        pos_state = utils_muav.interp_2d(t_int, t, pos_state)
        slerp_obj = Slerp(t, Rotation.from_quat(quat_state.T))
        quat_state = slerp_obj(t_int).as_quat().T
        base_state = np.concatenate((pos_state, quat_state))

        return t_int, base_state, base_twist_values, joint_values, prop_values, aero_values

    def __get_joint_short_name(self, joint_name):
        joint_name = joint_name.split("_")[2:-1]
        name = joint_name[1][0]
        if joint_name[0] != "":
            name += "_" + joint_name[0][0]
        return name

    def __get_propeller_short_name(self, propeller_name):
        propeller_name = propeller_name.split("_")[2:]
        name = ""
        for pn in propeller_name:
            name += pn
        return name

    def __create_control_input_bar_plot(self):
        if self.__plot_control_input:
            matplotlib.use("TkAgg")
            fig, axs = plt.subplots(2, 1, figsize=(6, 4))
            fig.canvas.manager.window.wm_geometry("+%d+%d" % (100, 100))
            self.__control_input_fig = fig
            axs[0].bar(
                np.arange(self.robot_param["ndofs"]).astype(str),
                self.robot_param["joint_pos"]["ub"].reshape(-1) * 180 / math.pi,
                color="grey",
            )
            axs[0].bar(
                np.arange(self.robot_param["ndofs"]).astype(str),
                self.robot_param["joint_pos"]["lb"].reshape(-1) * 180 / math.pi,
                color="grey",
            )
            axs[0].set_ylabel("[deg]")
            axs[0].set_xticks(np.arange(self.robot_param["ndofs"]))
            axs[0].set_xticklabels(
                [self.__get_joint_short_name(name_joint) for name_joint in self.robot_param["joint_list"]]
            )
            axs[1].bar(
                np.arange(self.robot_param["nprop"]).astype(str),
                self.robot_param["prop_thrust"]["ub"].reshape(-1),
                color="grey",
            )
            axs[1].bar(
                np.arange(self.robot_param["nprop"]).astype(str),
                self.robot_param["prop_thrust"]["lb"].reshape(-1),
                color="grey",
            )
            axs[1].set_ylabel("[N]")
            axs[1].set_xticks(np.arange(self.robot_param["nprop"]))
            axs[1].set_xticklabels(
                [self.__get_propeller_short_name(name_prop) for name_prop in self.robot_param["propellers_frame_list"]]
            )
            self.b["joint_values"] = axs[0].bar(
                np.arange(self.robot_param["ndofs"]).astype(str),
                self.robot_param["joint_pos"]["ub"].reshape(-1) * 0,
                color="blue",
            )
            self.b["prop_values"] = axs[1].bar(
                np.arange(self.robot_param["nprop"]).astype(str),
                self.robot_param["prop_thrust"]["lb"].reshape(-1) * 0,
                color="blue",
            )
            plt.show(block=False)

    def __update_control_input_bar_plot(self, joint_values, prop_values):
        if self.__plot_control_input:
            for i, rect in enumerate(self.b["joint_values"]):
                rect.set_height(joint_values[i] * 180 / math.pi)
                rect.set_color("blue" if joint_values[i] > 0 else "red")

            for i, rect in enumerate(self.b["prop_values"]):
                rect.set_height(prop_values[i])
                rect.set_color("blue" if prop_values[i] > 0 else "red")

            self.__control_input_fig.canvas.draw()
            self.__control_input_fig.canvas.flush_events()

    def run(
        self,
        t: np.ndarray,
        base_state: np.matrix = None,
        joint_values: np.matrix = None,
        prop_values: np.matrix = None,
        aero_values: Dict = None,
        base_twist_values: np.matrix = None,
        t_end: int = None,
        t_start: int = None,
    ):
        nt = len(t)
        if base_state is None:
            base_state = np.tile(np.array([0, 0, 0, 0, 0, 0, 1]), (nt, 1)).T
        if base_twist_values is None:
            base_twist_values = np.zeros((6, nt))
        if joint_values is None:
            joint_values = np.zeros((self.robot_param["ndofs"], nt))
        if prop_values is None:
            prop_values = np.zeros((self.robot_param["nprop"], nt))
        if t_end is None:
            t_end = t[-1]
        if t_start is None:
            t_start = t[0]

        aero_values = deepcopy(aero_values)

        t, base_state, base_twist_values, joint_values, prop_values, aero_values = self.__interp_all(
            t, base_state, base_twist_values, joint_values, prop_values, aero_values
        )

        try:
            index_start = np.where(t >= t_start)[0][0]
        except:
            index_start = 0
        try:
            index_end = np.where(t >= t_end)[0][0]
        except:
            index_end = len(t)

        self.__create_control_input_bar_plot()
        time.sleep(2)

        while not rospy.is_shutdown():
            self.time_start = time.time()
            for k in range(index_start, index_end):
                t_sleep = self.time_start + t[k] - t[index_start] - time.time()
                if t_sleep > 0:
                    time.sleep(t_sleep)

                self.__update_ros_timestamp()
                if self.__print_time:
                    print(f"Plotting {t[k]:.3f}...")
                self.__publish_joint_state(joint_values[:, k])
                self.__publish_base_pose(base_state[:, k])
                self.__publish_propellers(prop_values[:, k])
                if aero_values is not None:
                    self.__publish_aerodynamics(aero_values[:, k], base_state[:, k])
                self.__publish_twist(base_twist_values[:, k], base_state[:, k])
                # goal and obstacle
                self.goal_pub.publish(self.goal_obj)
                self.obstacle_pub.publish(self.obstacle_obj)

                self.__update_control_input_bar_plot(joint_values[:, k], prop_values[:, k])

            return


if __name__ == "__main__":
    robot = Robot(f"{utils_muav.get_repository_tree(relative_path=True)['urdf']}/drone_dst_t1")
    viz = Visualize_robot(robot, plot_control_input=False)
    N = 100
    T = np.linspace(0, 10, N)
    base_state = np.zeros((7, N))
    base_state[3:, :] = SO3.from_euler(np.array([0, -90, 0]) * math.pi / 180).as_quat().coeffs().full()
    joint_values = np.zeros((robot.ndofs, N))
    prop_values = np.zeros((robot.nprop, N))
    for k, t in enumerate(T):
        base_state[0, k] = t / 10
        prop_values[:, k] = t / 10
        joint_values[:, k] = 0.5 * math.sin(t % (2 * math.pi))
    viz.run(t=T, base_state=base_state, prop_values=prop_values, joint_values=joint_values)
    viz.close_rviz()
