from core.robot import Robot
import casadi as cs
import numpy as np
from core.visualizer import Visualize_robot
from liecasadi import SE3, SO3, SO3Tangent
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from typing import Dict, List
import utils_muav
import os
from scipy import signal
import time
from dataclasses import dataclass


class Trajectory:
    def __init__(self, robot: Robot, knots: int, time_horizon: float = None) -> None:
        self.robot = robot
        self.N = knots
        self.time_horizon = time_horizon
        self.optivar = {}
        self.optipar = {}
        self.constant = {}
        self._set_constant_values()
        self.cost = 0
        self.ngoals = 0
        self.nobstacles = 0
        self.__list_indexes_goals = [0]
        self._project_status = utils_muav.Project_status()
        self.name_trajectory = ""
        self._regularize_control_input_variations = False

    def _add_cost(self, cost):
        if not hasattr(self, "cost"):
            self.cost = cost
        else:
            self.cost += cost

    def _bound(self, lb, x, ub):
        # self.opti.subject_to(self.opti.bounded(lb, x, ub))
        self.opti.subject_to(x >= lb)
        self.opti.subject_to(x <= ub)

    def _set_solver_stgs(self):
        optiSettings = {"expand": True, "print_time": 0}
        solvSettings = {
            "print_level": 0,
            "sb": "yes",
            "linear_solver": "mumps",
            "nlp_scaling_max_gradient": 100.0,
            "nlp_scaling_min_value": 1e-6,
            "tol": 1e-3,
            "dual_inf_tol": 1000.0,
            "compl_inf_tol": 1e-2,
            "constr_viol_tol": 1e-4,
            "acceptable_tol": 1e0,
            "acceptable_iter": 2,
            "acceptable_compl_inf_tol": 1,
            "alpha_for_y": "dual-and-full",
            "max_iter": 1000,
            "warm_start_bound_frac": 1e-2,
            "warm_start_bound_push": 1e-2,
            "warm_start_mult_bound_push": 1e-2,
            "warm_start_slack_bound_frac": 1e-2,
            "warm_start_slack_bound_push": 1e-2,
            "warm_start_init_point": "yes",
            "required_infeasibility_reduction": 0.8,
            "perturb_dec_fact": 0.1,
            "max_hessian_perturbation": 100.0,
            "fast_step_computation": "yes",
            "hessian_approximation": "limited-memory",
        }
        self.opti.solver("ipopt", optiSettings, solvSettings)

    def _get_optivar_time(self):
        self.optivar["dt"] = self.opti.variable(self.ngoals, 1)
        self.__list_indexes_goals
        self.vector_dt = []
        for i, r in enumerate(np.diff(self.__list_indexes_goals)):
            self.vector_dt = cs.vertcat(self.vector_dt, cs.repmat(self.optivar["dt"][i], r, 1))

    def set_wind_parameters(self, air_density: float, air_viscosity: float, vel_w_wind: np.ndarray) -> None:
        self.constant["air_density"] = air_density
        self.constant["air_viscosity"] = air_viscosity
        self.constant["vel_w_wind"] = vel_w_wind.ravel()

    def set_gains(self, gains: "Gains_Trajectory") -> None:
        self.constant["gains"] = gains

    def _set_constant_values(self) -> None:
        self.constant["dt"] = {}
        self.constant["dt"]["lb"] = 0.001
        self.constant["dt"]["ub"] = 0.1
        self.constant["robot_attitude"] = {}
        self.constant["robot_attitude"]["pitch"] = {}
        self.constant["robot_attitude"]["pitch"]["lb"] = -30 * math.pi / 180
        self.constant["robot_attitude"]["pitch"]["ub"] = +30 * math.pi / 180
        self.constant["robot_attitude"]["roll"] = {}
        self.constant["robot_attitude"]["roll"]["lb"] = -30 * math.pi / 180
        self.constant["robot_attitude"]["roll"]["ub"] = +30 * math.pi / 180
        self.constant["robot_speed"] = {}
        self.constant["robot_speed"]["lin"] = 25
        self.constant["robot_speed"]["ang"] = 1
        self.constant["robot_acc"] = {}
        self.constant["robot_acc"]["ang"] = 10
        self.constant["obstacles"] = []
        self.constant["goals"] = []
        self.constant["gains"] = Gains_Trajectory()

    def _set_constraint_time_increment(self):
        self._bound(self.constant["dt"]["lb"], self.optivar["dt"], self.constant["dt"]["ub"])
        if self.time_horizon is not None:
            total_time = self.optivar["dt"].T @ np.diff(self.__list_indexes_goals)
            self._bound(-1e-3, total_time - self.time_horizon, 1e-3)

    def _set_feasibility_constraint_quaternion(self, tol: float = 0, index_range: int = None):
        if index_range is None:
            index_range = self.N + 1
        for k in range(index_range):
            q = self.optivar["quat_w_b"][:, k]
            self._bound(-1, q, 1)
            self._bound(-tol, q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2 - 1, tol)

    def _set_box_constraint_aerodynamic_angles(self):
        for i, aero_frame in enumerate(self.robot.aero_frame_list):
            for k in range(self.N + 1):
                alpha = self.optivar["vcat_alpha"][i : i + 1, k]
                beta = self.optivar["vcat_beta"][i : i + 1, k]
                self._bound(
                    self.robot.aero_alpha_limit_list[aero_frame]["lb"],
                    alpha,
                    self.robot.aero_alpha_limit_list[aero_frame]["ub"],
                )
                self._bound(
                    self.robot.aero_beta_limit_list[aero_frame]["lb"],
                    beta,
                    self.robot.aero_beta_limit_list[aero_frame]["ub"],
                )

    def _set_box_constraint_robot_base_rpy(self):
        for k in range(self.N + 1):
            quat = self.optivar["quat_w_b"][:, k]
            rpy = SO3(quat).as_euler()
            self._bound(
                self.constant["robot_attitude"]["pitch"]["lb"], rpy[1], self.constant["robot_attitude"]["pitch"]["ub"]
            )
            self._bound(
                self.constant["robot_attitude"]["roll"]["lb"], rpy[0], self.constant["robot_attitude"]["roll"]["ub"]
            )

    def _set_box_constraint_robot_base_vel(self):
        for k in range(self.N + 1):
            linVel_w_b = self.optivar["twist_w_b"][:3, k]
            angVel_w_b = self.optivar["twist_w_b"][3:, k]
            self.opti.subject_to(linVel_w_b.T @ linVel_w_b < self.constant["robot_speed"]["lin"] ** 2)
            self.opti.subject_to(angVel_w_b.T @ angVel_w_b < self.constant["robot_speed"]["ang"] ** 2)

    def _set_box_constraint_robot_base_acc(self):
        for k in range(self.N):
            angAcc_w_b = self.optivar["dot_twist_w_b"][3:, k]
            self.opti.subject_to(angAcc_w_b.T @ angAcc_w_b < self.constant["robot_acc"]["ang"] ** 2)

    def _set_box_constraint_joint_pos(self):
        if self.robot.ndofs > 0:
            for k in range(self.N + 1):
                self._bound(
                    self.robot.limits["joint_pos"]["lb"], self.optivar["s"][:, k], self.robot.limits["joint_pos"]["ub"]
                )

    def _set_box_constraint_joint_vel(self):
        if self.robot.ndofs > 0:
            for k in range(self.N + 1):
                self._bound(
                    self.robot.limits["joint_vel"]["lb"],
                    self.optivar["dot_s"][:, k],
                    self.robot.limits["joint_vel"]["ub"],
                )

    def _set_box_constraint_joint_acc(self):
        if self.robot.ndofs > 0:
            for k in range(self.N + 1):
                self._bound(
                    self.robot.limits["joint_acc"]["lb"],
                    self.optivar["ddot_s"][:, k],
                    self.robot.limits["joint_acc"]["ub"],
                )

    def _set_box_constraint_propeller_thrust(self):
        if self.robot.nprop > 0:
            for k in range(self.N + 1):
                self._bound(
                    self.robot.limits["prop_thrust"]["lb"],
                    self.optivar["thrust"][:, k],
                    self.robot.limits["prop_thrust"]["ub"],
                )

    def _set_box_constraint_propeller_dot_thrust(self):
        if self.robot.nprop > 0:
            for k in range(self.N + 1):
                self._bound(
                    self.robot.limits["prop_dot_thrust"]["lb"],
                    self.optivar["dot_thrust"][:, k],
                    self.robot.limits["prop_dot_thrust"]["ub"],
                )

    def _set_constraint_initial_condition(self, name_optivar: str, tol: float = 0):
        self._bound(-tol, self.optivar[name_optivar][:, 0] - self.initial_condition[name_optivar], tol)

    def _set_constraint_initial_condition_quat(self, name_optivar: str, tol: float = 0):
        quat_opti = SO3(self.optivar[name_optivar][:, 0])
        quat_init = SO3(self.initial_condition[name_optivar])
        self._bound(-tol, (quat_opti - quat_init).vec, tol)

    def _set_constraint_aerodynamic_forces(self):
        Vwind_w = self.constant["vel_w_wind"]
        air_density = self.constant["air_density"]
        air_viscosity = self.constant["air_viscosity"]
        for i, aero_frame in enumerate(self.robot.aero_frame_list):
            get_aerodynamic_wrench_fun = self.robot.get_aerodynamic_wrench_fun(aero_frame)
            get_aerodynamic_angles_and_speed_fun = self.robot.get_aerodynamic_angles_and_speed_fun(aero_frame)
            for k in range(self.N + 1):
                tform_w_b = self._get_tform_w_b_symbolic(k)
                twist_w_b = self.optivar["twist_w_b"][:, k]
                s = self.optivar["s"][:, k]
                ds = self.optivar["dot_s"][:, k]
                wrenchAero_w = get_aerodynamic_wrench_fun(
                    tform_w_b,
                    s,
                    air_density,
                    air_viscosity,
                    self.optivar["vcat_alpha"][i : i + 1, k],
                    self.optivar["vcat_beta"][i : i + 1, k],
                    self.optivar["vcat_vinf_norm"][i : i + 1, k],
                )
                alpha, beta, Vinf_norm = get_aerodynamic_angles_and_speed_fun(tform_w_b, twist_w_b, s, ds, Vwind_w)

                self.opti.subject_to(wrenchAero_w == self.optivar["vcat_wrenchAero_w"][6 * i : 6 + 6 * i, k])
                self.opti.subject_to(alpha == self.optivar["vcat_alpha"][i : i + 1, k])
                self.opti.subject_to(beta == self.optivar["vcat_beta"][i : i + 1, k])
                self.opti.subject_to(Vinf_norm == self.optivar["vcat_vinf_norm"][i : i + 1, k])

    def _minimize_time_horizon(self, gain: int = 1):
        self._add_cost(gain * cs.sum1(self.vector_dt))

    def _minimize_robot_thrust(self, gain: int = 1):
        for k in range(self.N + 1):
            T = self.optivar["thrust"][:, k]
            self._add_cost(gain * T.T @ T)

    def _minimize_robot_joint_vel(self, gain: int = 1):
        for k in range(self.N + 1):
            ds = self.optivar["dot_s"][:, k]
            self._add_cost(gain * ds.T @ ds)

    def _minimize_robot_joint_acc(self, gain: int = 1):
        for k in range(self.N + 1):
            dds = self.optivar["ddot_s"][:, k]
            self._add_cost(gain * dds.T @ dds)

    def _minimize_robot_joint_torque(self, gain: int = 1):
        for k in range(self.N + 1):
            tau = self.optivar["torque"][:, k]
            self._add_cost(gain * tau.T @ tau)

    def add_goal(self, goal: "Goal"):
        assert goal.k > 0
        assert goal.k <= self.N
        self.constant["goals"].append(goal)
        self.ngoals += 1
        self.__list_indexes_goals.append(goal.k)
        self.__list_indexes_goals.sort()

    def add_obstacle(self, obstacle: "Obstacle"):
        self.constant["obstacles"].append(obstacle)
        self.nobstacles += 1

    def _set_constraint_obstacles_avoidance(self, tol: float = 0):
        for collision_frame in self.robot.collisions_frame_list:
            forward_kinematics_fun = self.robot.kinDyn.forward_kinematics_fun(collision_frame)
            for obstacle in self.constant["obstacles"]:
                for k in range(self.N + 1):
                    tform_w_b = self._get_tform_w_b_symbolic(k)
                    s = self.optivar["s"][:, k]
                    pos_w_R = forward_kinematics_fun(tform_w_b, s)[:3, 3]
                    xR, yR, zR = pos_w_R[0], pos_w_R[1], pos_w_R[2]
                    if type(obstacle) == Obstacle_Sphere:
                        xO, yO, zO = obstacle.xyz[0], obstacle.xyz[1], obstacle.xyz[2]
                        rO = obstacle.r
                        # sphere equation: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2 where (x0,y0,z0) is the center of the sphere
                        self.opti.subject_to((xR - xO) ** 2 + (yR - yO) ** 2 + (zR - zO) ** 2 - rO**2 > tol**2)
                    elif type(obstacle) == Obstacle_InfiniteCylinder:
                        xO, yO = obstacle.xy[0], obstacle.xy[1]
                        rO = obstacle.r
                        # cylinder equation: (x-x0)^2 + (y-y0)^2 = r^2 where (x0,y0) is the center of the cylinder
                        self.opti.subject_to((xR - xO) ** 2 + (yR - yO) ** 2 - rO**2 > tol**2)
                    elif type(obstacle) == Obstacle_Plane:
                        xO, yO, zO = obstacle.xyz[0], obstacle.xyz[1], obstacle.xyz[2]
                        n = obstacle.normal
                        # plane equation: A(x-x0) + B(y-y0) + C(z-z0) = 0 where A,B,C are the normal components and (x0,y0,z0) is a point on the plane
                        self.opti.subject_to(n[0] * (xR - xO) + n[1] * (yR - yO) + n[2] * (zR - zO) > tol)
                    else:
                        Warning("Obstacle type not implemented")

    def _set_goals_condition(self):
        for goal in self.constant["goals"]:
            k = goal.k

            for q in [
                "position",
                "orientation",
                "linearVelocity",
                "linearVelocityBody",
                "normLinearVelocity",
                "angularVelocity",
            ]:
                g = getattr(goal, q)
                if bool(g):
                    if q == "position":
                        x = self.optivar["pos_w_b"][:, k] - g["value"]
                    elif q == "orientation":
                        quat_w_b = SO3(self.optivar["quat_w_b"][:, k])
                        quatDes_w_b = SO3(g["value"])
                        x = (quat_w_b - quatDes_w_b).vec  # rpy
                    elif q == "linearVelocity":
                        x = self.optivar["twist_w_b"][:3, k] - g["value"]
                    elif q == "linearVelocityBody":
                        v_w_b = self.optivar["twist_w_b"][:3, k]
                        rotm_b_w = SO3(self.optivar["quat_w_b"][:, k]).as_matrix().T
                        v_b_b = rotm_b_w @ v_w_b
                        x = v_b_b - g["value"]
                    elif q == "normLinearVelocity":
                        x = self.optivar["twist_w_b"][:3, k].T @ self.optivar["twist_w_b"][:3, k] - g["value"] ** 2
                    elif q == "angularVelocity":
                        x = self.optivar["twist_w_b"][3:, k] - g["value"]
                    if g["mandatory"]:
                        self._bound(-g["tol"], x, g["tol"])
                    else:
                        self._add_cost(g["gain"] * x.T @ x)

    def _get_empty_out(self):
        out = {}
        out["dt"] = None
        out["time"] = None
        out["computational_time"] = None
        # nlp_output
        out["nlp_output"] = {}
        out["nlp_output"]["cost_function"] = None
        out["nlp_output"]["iter_count"] = None
        out["nlp_output"]["success"] = False
        out["nlp_output"]["return_status"] = None
        # optivar
        out["optivar"] = {}
        # centroidal
        out["optivar"]["pos_w_com"] = np.zeros((3, self.N + 1))
        out["optivar"]["vel_w_com"] = np.zeros((3, self.N + 1))
        out["optivar"]["acc_w_com"] = np.zeros((3, self.N + 1))
        out["optivar"]["angMom_w"] = np.zeros((3, self.N + 1))
        out["optivar"]["dot_angMom_w"] = np.zeros((3, self.N + 1))
        # wind
        out["optivar"]["vcat_wrenchAero_w"] = np.zeros((self.robot.naero * 6, self.N + 1))
        out["optivar"]["vcat_alpha"] = np.zeros((self.robot.naero * 1, self.N + 1))
        out["optivar"]["vcat_beta"] = np.zeros((self.robot.naero * 1, self.N + 1))
        out["optivar"]["vcat_vinf_norm"] = np.zeros((self.robot.naero * 1, self.N + 1))
        # robot
        out["optivar"]["s"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["optivar"]["dot_s"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["optivar"]["ddot_s"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["optivar"]["static_torque"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["optivar"]["torque"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["optivar"]["dot_torque"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["optivar"]["thrust"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["optivar"]["dot_thrust"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["optivar"]["pos_w_b"] = np.zeros((3, self.N + 1))
        out["optivar"]["quat_w_b"] = np.zeros((4, self.N + 1))
        out["optivar"]["twist_w_b"] = np.zeros((6, self.N + 1))
        out["optivar"]["dot_twist_w_b"] = np.zeros((6, self.N + 1))
        return out

    def create(self):
        assert self.ngoals > 0
        assert self.__list_indexes_goals[-1] == self.N
        print("\nTrajectory")
        print(f"\trobot: \t{self.robot.fullpath_model}")
        print(f"\tgoals: \t{self.ngoals}")
        print(f"\tobsts: \t{self.nobstacles}\n")
        pass

    def debug(self):
        self.opti.debug.show_infeasibilities()

    def save(self, out: Dict = {}, folder_name: str = "result"):
        out["initiat_condition"] = self.initial_condition
        out["constant"] = self.constant
        out["robot"] = {}
        out["robot"]["limits"] = self.robot.limits
        out["robot"]["ndofs"] = self.robot.ndofs
        out["robot"]["nprop"] = self.robot.nprop
        out["robot"]["naero"] = self.robot.naero
        out["robot"]["joint_list"] = self.robot.joint_list
        out["robot"]["aero_frame_list"] = self.robot.aero_frame_list
        out["robot"]["propellers_frame_list"] = self.robot.propellers_frame_list
        out["robot"]["fullpath_model"] = self.robot.fullpath_model
        out["pp"] = {}
        # rpy
        out["pp"]["rpy_w_b"] = np.zeros((3, self.N + 1))
        for i, quat_w_b in enumerate(out["optivar"]["quat_w_b"].T):
            out["pp"]["rpy_w_b"][:, i] = SO3(quat_w_b).as_euler().full().T
        # aero_frame
        out["pp"]["pos_w_aero"] = np.zeros((3 * self.robot.naero, self.N + 1))
        for i, aero_frame in enumerate(self.robot.aero_frame_list):
            forward_kinematics_fun = self.robot.kinDyn.forward_kinematics_fun(aero_frame)
            for k in range(self.N + 1):
                s = out["optivar"]["s"][:, k]
                pos_w_b = out["optivar"]["pos_w_b"][:, k]
                quat_w_b = out["optivar"]["quat_w_b"][:, k]
                tform_w_b = SE3(pos=pos_w_b, xyzw=quat_w_b).as_matrix()
                out["pp"]["pos_w_aero"][3 * i : 3 * i + 3, k] = forward_kinematics_fun(tform_w_b, s).full()[:3, 3]
        # com
        out["pp"]["pos_w_com"] = np.zeros((3, self.N + 1))
        if out["optivar"]["pos_w_com"].sum() != 0:
            out["pp"]["pos_w_com"] = out["optivar"]["pos_w_com"]
        else:
            for k in range(self.N + 1):
                s = out["optivar"]["s"][:, k]
                pos_w_b = out["optivar"]["pos_w_b"][:, k]
                quat_w_b = out["optivar"]["quat_w_b"][:, k]
                tform_w_b = SE3(pos=pos_w_b, xyzw=quat_w_b).as_matrix()
                out["pp"]["pos_w_com"][:, k] = self.robot.kinDyn.CoM_position_fun()(tform_w_b, s).full().T
        # cop
        out["pp"]["pos_b_cop"] = np.zeros((3, self.N + 1))
        get_aerodynamic_cop_fun = self.robot.get_aerodynamic_cop_fun()
        for k in range(self.N + 1):
            s = out["optivar"]["s"][:, k]
            pos_w_b = out["optivar"]["pos_w_b"][:, k]
            quat_w_b = out["optivar"]["quat_w_b"][:, k]
            tform_w_b = SE3(pos=pos_w_b, xyzw=quat_w_b).as_matrix()
            vcat_wrenchAero_w = out["optivar"]["vcat_wrenchAero_w"][:, k]
            out["pp"]["pos_b_cop"][:, k] = get_aerodynamic_cop_fun(tform_w_b, s, vcat_wrenchAero_w).full().T
        # joint
        out["pp"]["ddot_s"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["pp"]["torque"] = np.zeros((self.robot.ndofs, self.N + 1))
        out["pp"]["joint_power"] = np.zeros((self.robot.ndofs, self.N + 1))
        if self.robot.ndofs > 0:
            out["pp"]["static_torque"] = np.zeros((self.robot.ndofs, self.N + 1))
            for k in range(self.N + 1):
                posquat_w_b = np.concatenate((out["optivar"]["pos_w_b"][:, k], out["optivar"]["quat_w_b"][:, k]))
                s = out["optivar"]["s"][:, k]
                prop_thrust = out["optivar"]["thrust"][:, k]
                vcat_wrenchAero_w = out["optivar"]["vcat_wrenchAero_w"][:, k]
                out["pp"]["static_torque"][:, k] = (
                    self.robot.get_joint_torque_static_fun()(posquat_w_b, s, prop_thrust, vcat_wrenchAero_w).full().T
                )
            if out["optivar"]["torque"].sum() != 0:
                out["pp"]["ddot_s"] = out["optivar"]["ddot_s"]
                out["pp"]["torque"] = out["optivar"]["torque"]
            else:
                t = out["time"]
                t_int = np.linspace(t[0], t[-1], self.N + 1)
                get_joint_torque_fun = self.robot.get_joint_torque_fun()
                for i, ds in enumerate(out["optivar"]["dot_s"]):
                    ds_int = np.interp(t_int, t, ds)
                    dds_int = signal.savgol_filter(
                        ds_int, window_length=7, polyorder=3, deriv=1, delta=t_int[1] - t_int[0]
                    )

                    out["pp"]["ddot_s"][i, :] = np.interp(t, t_int, dds_int)
                for k in range(self.N + 1):
                    posquat_w_b = np.concatenate((out["optivar"]["pos_w_b"][:, k], out["optivar"]["quat_w_b"][:, k]))
                    twist_w_b = out["optivar"]["twist_w_b"][:, k]
                    s = out["optivar"]["s"][:, k]
                    dot_s = out["optivar"]["dot_s"][:, k]
                    ddot_s = out["pp"]["ddot_s"][:, k]
                    thrust = out["optivar"]["thrust"][:, k]
                    vcat_wrenchAero_w = out["optivar"]["vcat_wrenchAero_w"][:, k]
                    out["pp"]["torque"][:, k] = (
                        get_joint_torque_fun(posquat_w_b, twist_w_b, s, dot_s, ddot_s, thrust, vcat_wrenchAero_w)
                        .full()
                        .T
                    )
            for k in range(self.N + 1):
                out["pp"]["joint_power"][:, k] = self.robot.get_joint_consumption(
                    out["pp"]["torque"][:, k], out["optivar"]["dot_s"][:, k]
                )
        # propeller
        out["pp"]["propeller_power"] = np.zeros((self.robot.nprop, self.N + 1))
        if self.robot.nprop > 0:
            for k in range(self.N + 1):
                out["pp"]["propeller_power"][:, k] = self.robot.get_propeller_consumption(
                    out["optivar"]["thrust"][:, k]
                )

        # save
        if os.path.exists(folder_name) is False:
            os.mkdir(folder_name)
        name = f"{folder_name}/{self.robot.name}-{utils_muav.get_date_str()}"
        pickle.dump({"out": out}, open(name + ".p", "wb"))
        self._project_status.create_report(name)
        self.name_trajectory = name
        return out

    def _get_tform_w_b_symbolic(self, k: int):
        tform_w_b = SE3(pos=self.optivar["pos_w_b"][:, k], xyzw=self.optivar["quat_w_b"][:, k]).as_matrix()
        return tform_w_b

    @staticmethod
    def rpy_from_quat(quat):
        [qx, qy, qz, qw] = [quat[0], quat[1], quat[2], quat[3]]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(2 * (qw * qy - qz * qx))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([roll, pitch, yaw])

    @staticmethod
    def pattern_cylinder_obstacles(
        N: int, x_lim: list, y_lim: list, r_lim: list, seed: int = None, savedir: str = None, plot: bool = False
    ):
        if seed is not None:
            np.random.seed(seed)
        x = np.random.uniform(x_lim[0], x_lim[1], N)
        y = np.random.uniform(y_lim[0], y_lim[1], N)
        r = np.random.uniform(r_lim[0], r_lim[1], N)
        list_obstacles = []
        for i in range(N):
            list_obstacles.append(Obstacle_InfiniteCylinder(xy=[x[i], y[i]], r=r[i]))
        if savedir is not None:
            pickle.dump(list_obstacles, open(f"{savedir}_{seed}.p", "wb"))
        if plot:
            plt.figure()
            for i in range(N):
                circle = plt.Circle((x[i], y[i]), r[i], color="r", fill=False)
                plt.gca().add_patch(circle)
            ax = plt.gca()
            ax.add_patch(plt.Rectangle((x_lim[0], y_lim[0]), x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], fill=False))
            plt.axis("equal")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.show()
        return list_obstacles


@dataclass
class Gains_Trajectory:
    cost_function_weight_energy: float = 1.0
    cost_function_weight_time: float = 1.0


@dataclass
class Obstacle:
    pass


@dataclass
class Obstacle_Sphere(Obstacle):
    xyz: list
    r: float

    def __post_init__(self):
        self.type_rviz = 2

    def plot_xy(self, color="r", ax=None):
        circle = plt.Circle((self.xyz[0], self.xyz[1]), self.r, color=color, fill=True, alpha=0.15)
        if ax is None:
            plt.gca().add_patch(circle)
        else:
            ax.add_patch(circle)


@dataclass
class Obstacle_InfiniteCylinder(Obstacle):
    xy: list
    r: float

    def __post_init__(self):
        self.type_rviz = 3

    def plot_xy(self, color="r", ax=None):
        circle = plt.Circle((self.xy[0], self.xy[1]), self.r, color=color, fill=True, alpha=0.15)
        if ax is None:
            plt.gca().add_patch(circle)
        else:
            ax.add_patch(circle)


@dataclass
class Obstacle_Plane(Obstacle):
    xyz: list
    normal: list

    def __post_init__(self):
        self.type_rviz = 4  # line strip

    def plot_xy(self, color="r"):
        # A*(x-x0)+B*(y-y0)=0
        # y = B/A*(x-x0)+y0 = m*x + q
        m = self.normal[1] / self.normal[0] if self.normal[0] != 0 else 0
        q1 = self.xyz[1]
        q2 = self.xyz[1] - 100 * self.normal[1]
        x = np.array([self.xyz[0] - 100, self.xyz[0] + 100])
        plt.gca()
        plt.fill_between(x, m * x + q1, m * x + q2, color=color, alpha=0.15)


class Goal:
    def __init__(self, index: int = 0) -> None:
        self.k = index
        self.position = {}
        self.orientation = {}
        self.linearVelocity = {}
        self.linearVelocityBody = {}
        self.normLinearVelocity = {}
        self.angularVelocity = {}

    def set_position(self, xyz: list, isStrict: bool = False, param: float = 0):
        self.position = self.__constructor(xyz, isStrict, param)
        return self

    def set_orientation(self, xyzw: list, isStrict: bool = False, param: float = 0):
        self.orientation = self.__constructor(xyzw, isStrict, param)
        return self

    def set_linearVelocity(self, xyz: list, isStrict: bool = False, param: float = 0):
        self.linearVelocity = self.__constructor(xyz, isStrict, param)
        return self

    def set_linearVelocityBody(self, xyz: list, isStrict: bool = False, param: float = 0):
        self.linearVelocityBody = self.__constructor(xyz, isStrict, param)
        return self

    def set_normLinearVelocity(self, v: float, isStrict: bool = False, param: float = 0):
        self.normLinearVelocity = self.__constructor(v, isStrict, param)
        return self

    def set_angularVelocity(self, xyz: list, isStrict: bool = False, param: float = 0):
        self.angularVelocity = self.__constructor(xyz, isStrict, param)
        return self

    def describe(self):
        str_goal = f' | pos: {self.position["value"]} |'
        if self.orientation != {}:
            rpy = SO3(self.orientation["value"]).as_euler().full().T
            str_goal += f" rpy: {rpy} |"
        if self.linearVelocity != {}:
            str_goal += f' linVel: {self.linearVelocity["value"]} |'
        if self.linearVelocityBody != {}:
            str_goal += f' linVelBody: {self.linearVelocityBody["value"]} |'
        if self.normLinearVelocity != {}:
            str_goal += f' normLinVel: {self.normLinearVelocity["value"]} |'
        if self.angularVelocity != {}:
            str_goal += f' angVel: {self.angularVelocity["value"]} |'
        return str_goal

    def plot_xy(self, color="g", ax=None):
        radius = self.position["tol"]
        circle = plt.Circle(
            (self.position["value"][0], self.position["value"][1]), radius, color=color, fill=True, alpha=0.15
        )
        if ax is None:
            plt.gca().add_patch(circle)
        else:
            ax.add_patch(circle)

    @staticmethod
    def __constructor(value: list, mandatory: bool, param: float) -> Dict:
        out = {}
        out["value"] = value
        out["mandatory"] = mandatory
        if mandatory:
            out["tol"] = param
        else:
            out["gain"] = param
        return out


@dataclass
class Task:
    name: str
    knots: int
    list_goals: List[Goal]
    list_obstacles: List[Obstacle]
    ic_twist_w_b: np.ndarray = np.array([10, 0, 0, 0, 0, 0])
    ic_quat_w_b: np.ndarray = np.array([0, 0, 0, 1])
    ic_position_w_b: np.ndarray = np.array([0, 0, 0])

    def __post_init__(self):
        list_obstacles = []
        for obstacle in self.list_obstacles:
            if isinstance(obstacle, list):
                list_obstacles += obstacle
            else:
                list_obstacles.append(obstacle)
        self.list_obstacles = list_obstacles
        self.out = {"knots": self.knots, "goals": self.list_goals, "obstacles": self.list_obstacles}

    def save(self) -> "Task":
        name = f"{utils_muav.get_repository_tree()['pickle_codesign_tasks']}/{self.name}.p"
        pickle.dump(self.out, open(name, "wb"))
        print(f"Task saved to {name}")
        return self

    def plot(self, xlim=None, ylim=None) -> "Task":
        plt.figure(figsize=(16, 9))
        Goal(index=0).set_position(xyz=[0, 0, 0], isStrict=True, param=0.1).plot_xy(color="b")
        for goal in self.list_goals:
            goal.plot_xy()
        for obstacle in self.list_obstacles:
            obstacle.plot_xy()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(xlim) if xlim else 0
        plt.ylim(ylim) if ylim else 0
        return self

    def plot_3d(self, xlim=None, ylim=None, zlim=None) -> "Task":
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection="3d")
        Goal(index=0).set_position(xyz=[0, 0, 0], isStrict=True, param=0.1).plot_xy(color="b")
        for goal in self.list_goals:
            goal.plot_xy()
        for obstacle in self.list_obstacles:
            obstacle.plot_xy()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(xlim) if xlim else 0
        plt.ylim(ylim) if ylim else 0
        return self

    @staticmethod
    def define_slalom_trajectory(
        distance_start_obstacle: float,
        obstacles_radius: float,
        final_velocity: List[float] = [10, 0, 0],
        initial_velocity: List[float] = [10, 0, 0, 0, 0, 0],
        initial_position: List[float] = [0, 0, 0],
        initial_orientation_rpy_deg: List[float] = [0, 0, 0],
    ) -> "Task":
        distance_obstacle_goal = distance_start_obstacle
        distance_obstacle_obstacle = distance_start_obstacle
        distance_start_goal = distance_start_obstacle + distance_obstacle_obstacle + distance_obstacle_goal
        radius_hidden_goals = distance_start_obstacle * 2 / 3
        knots = distance_start_goal + 30

        task = Task(
            name=f"slalom_{distance_start_obstacle}_{obstacles_radius}",
            knots=knots,
            list_goals=[
                Goal(index=knots)
                .set_position(xyz=[distance_start_goal, 0, 0], isStrict=True, param=0.1)
                .set_linearVelocity(xyz=final_velocity, isStrict=True, param=0.1),
                Goal(index=int(knots * 1 / 3)).set_position(
                    xyz=[distance_obstacle_goal, radius_hidden_goals, 0],
                    isStrict=True,
                    param=radius_hidden_goals,
                ),
                Goal(index=int(knots * 2 / 3)).set_position(
                    xyz=[
                        distance_obstacle_goal + distance_obstacle_obstacle,
                        -radius_hidden_goals,
                        0,
                    ],
                    isStrict=True,
                    param=radius_hidden_goals,
                ),
            ],
            list_obstacles=[
                Obstacle_InfiniteCylinder(xy=[distance_obstacle_goal, 0, 0], r=obstacles_radius),
                Obstacle_InfiniteCylinder(
                    xy=[distance_obstacle_goal + distance_obstacle_obstacle, 0, 0], r=obstacles_radius
                ),
                Obstacle_Plane(xyz=[0, 0, -2], normal=[0, 0, 1]),
            ],
            ic_twist_w_b=np.array(initial_velocity),
            ic_quat_w_b=np.squeeze(
                SO3.from_euler(np.array(initial_orientation_rpy_deg) * math.pi / 180).as_quat().coeffs().full()
            ),
            ic_position_w_b=np.array(initial_position),
        )

        return task


class Postprocess:
    def __init__(self, out: Dict = None) -> None:
        self.out = out
        self.__save_images = False
        self.__figsize = (16, 9)
        self.__init_stats()
        if self.out is not None:
            self.__compute_stats()

    def __init_stats(self):
        nan = float("NaN")
        self.stats = {
            "time": {"trajectory": nan, "computational": nan},
            "goals": {"individual": {}, "global": {}, "um": "m"},
            "goals_speed": {"individual": {}, "global": {}, "um": "m/s"},
            "obstacles": {"individual": {}, "global": {}, "um": "m"},
            "thrust": {"individual": {}, "global": {}, "um": "1"},
            "joint_pos": {"individual": {}, "global": {}, "um": "1"},
            "joint_vel": {"individual": {}, "global": {}, "um": "1/s"},
            "joint_tor": {"individual": {}, "global": {}, "um": "Nm"},
            "energy": {"individual": {}, "global": {}, "um": "J"},
        }
        self.stats["goals"]["individual"] = {"error": nan}
        self.stats["goals"]["global"] = {"average_error": nan, "total_error": nan}
        self.stats["goals_speed"]["individual"] = {"error": nan}
        self.stats["goals_speed"]["global"] = {"total_error": nan}
        self.stats["thrust"]["individual"] = {"max": nan, "mean": nan, "std": nan}
        self.stats["thrust"]["global"] = {"mean": nan, "std": nan}
        self.stats["joint_pos"]["individual"] = {"mean": nan, "std": nan, "Δ": nan}
        self.stats["joint_pos"]["global"] = {"mean": nan, "std": nan, "Δ": nan}
        self.stats["joint_vel"]["individual"] = {"max": nan, "mean": nan, "std": nan}
        self.stats["joint_vel"]["global"] = {"mean": nan, "std": nan}
        self.stats["joint_tor"]["individual"] = {"max": nan, "mean": nan, "std": nan}
        self.stats["joint_tor"]["global"] = {"mean": nan, "std": nan}
        self.stats["energy"]["individual"] = {"propeller": nan, "joint": nan}
        self.stats["energy"]["global"] = {"propeller": nan, "joint": nan}

    def __compute_stats(self):
        out = self.out
        self.stats["time"]["trajectory"] = out["time"][-1]
        self.stats["time"]["computational"] = out["computational_time"]
        # Goal
        list_error_goals = np.array([])
        list_error_goals_speed = np.array([])
        for goal in out["constant"]["goals"]:
            list_error_goals = np.append(
                list_error_goals,
                sum((out["optivar"]["pos_w_b"][:, goal.k] - np.array(goal.position["value"])) ** 2) ** 0.5,
            )
            if len(goal.linearVelocity) > 0:
                list_error_goals_speed = np.append(
                    list_error_goals_speed,
                    sum((out["optivar"]["twist_w_b"][:3, goal.k] - np.array(goal.linearVelocity["value"])) ** 2) ** 0.5,
                )
        self.stats["goals"]["individual"]["error"] = list_error_goals
        self.stats["goals"]["global"]["average_error"] = list_error_goals.mean()
        self.stats["goals"]["global"]["total_error"] = list_error_goals.sum()
        self.stats["goals_speed"]["individual"]["error"] = list_error_goals_speed
        self.stats["goals_speed"]["global"]["total_error"] = list_error_goals_speed.sum()

        # Energy
        self.stats["energy"]["individual"]["propeller"] = out["dt"].T @ (out["pp"]["propeller_power"][:, :-1].T)
        self.stats["energy"]["individual"]["joint"] = out["dt"].T @ (out["pp"]["joint_power"][:, :-1].T)
        self.stats["energy"]["global"]["propeller"] = self.stats["energy"]["individual"]["propeller"].sum()
        self.stats["energy"]["global"]["joint"] = self.stats["energy"]["individual"]["joint"].sum()
        # joint_vel
        t = np.arange(out["time"][0], out["time"][-1], 1 / 100)
        joint_vel = utils_muav.interp_2d(t, out["time"], out["optivar"]["dot_s"])
        joint_vel_norm = np.diag(1 / out["robot"]["limits"]["joint_vel"]["ub"]) @ joint_vel
        self.stats["joint_vel"]["individual"]["max"] = abs(joint_vel_norm).max(axis=1)
        self.stats["joint_vel"]["individual"]["mean"] = joint_vel_norm.mean(axis=1)
        self.stats["joint_vel"]["individual"]["std"] = joint_vel_norm.std(axis=1)
        self.stats["joint_vel"]["global"]["mean"] = np.mean(self.stats["joint_vel"]["individual"]["mean"])
        self.stats["joint_vel"]["global"]["std"] = np.mean(self.stats["joint_vel"]["individual"]["std"])
        # joint_tor
        joint_tor = utils_muav.interp_2d(t, out["time"], out["pp"]["torque"])
        self.stats["joint_tor"]["individual"]["max"] = abs(joint_tor).max(axis=1)
        self.stats["joint_tor"]["individual"]["mean"] = joint_tor.mean(axis=1)
        self.stats["joint_tor"]["individual"]["std"] = joint_tor.std(axis=1)
        self.stats["joint_tor"]["global"]["mean"] = np.mean(self.stats["joint_tor"]["individual"]["mean"])
        self.stats["joint_tor"]["global"]["std"] = np.mean(self.stats["joint_tor"]["individual"]["std"])
        # thrust
        thrust = utils_muav.interp_2d(t, out["time"], out["optivar"]["thrust"])
        thrust_norm = np.diag(1 / out["robot"]["limits"]["prop_thrust"]["ub"]) @ thrust
        self.stats["thrust"]["individual"]["max"] = abs(thrust_norm).max(axis=1)
        self.stats["thrust"]["individual"]["mean"] = thrust_norm.mean(axis=1)
        self.stats["thrust"]["individual"]["std"] = thrust_norm.std(axis=1)
        self.stats["thrust"]["global"]["mean"] = np.mean(self.stats["thrust"]["individual"]["mean"])
        self.stats["thrust"]["global"]["std"] = np.mean(self.stats["thrust"]["individual"]["std"])
        # joint_pos
        joint_pos = utils_muav.interp_2d(t, out["time"], out["optivar"]["s"])
        joint_pos_lb = utils_muav.get_2d_ndarray(out["robot"]["limits"]["joint_pos"]["lb"]).T
        scaling = np.diag(1 / (out["robot"]["limits"]["joint_pos"]["ub"] - out["robot"]["limits"]["joint_pos"]["lb"]))
        joint_pos_norm = scaling @ (joint_pos - joint_pos_lb)
        self.stats["joint_pos"]["individual"]["Δ"] = joint_pos_norm.max(axis=1) - joint_pos_norm.min(axis=1)
        self.stats["joint_pos"]["individual"]["mean"] = joint_pos_norm.mean(axis=1)
        self.stats["joint_pos"]["individual"]["std"] = joint_pos_norm.std(axis=1)
        self.stats["joint_pos"]["global"]["Δ"] = np.mean(self.stats["joint_pos"]["individual"]["Δ"])
        self.stats["joint_pos"]["global"]["mean"] = np.mean(self.stats["joint_pos"]["individual"]["mean"])
        self.stats["joint_pos"]["global"]["std"] = np.mean(self.stats["joint_pos"]["individual"]["std"])

    def save_pics(self, name_directory_images: str = None, figsize: tuple = (16, 9)):
        self.__save_images = True
        self.__figsize = figsize
        self.__name_directory_images = name_directory_images
        if os.path.isdir(self.__name_directory_images) == False:
            os.mkdir(self.__name_directory_images)

    def __save(self, name: str):
        if self.__save_images:
            plt.savefig("{}/{}".format(self.__name_directory_images, name))

    def thrust(self):
        out = self.out
        N = out["robot"]["nprop"]
        if N == 0:
            return
        fig, axs = plt.subplots(N, figsize=self.__figsize)
        if N == 1:
            i = 0
            axs.plot(out["time"], out["optivar"]["thrust"][i, :])
            axs.plot(out["time"], out["robot"]["limits"]["prop_thrust"]["ub"][i] * np.ones(out["time"].shape), "r--")
            axs.plot(out["time"], out["robot"]["limits"]["prop_thrust"]["lb"][i] * np.ones(out["time"].shape), "r--")
            axs.set_title(out["robot"]["propellers_frame_list"][i])
            axs.set_xlabel("time [s]")
            axs.set_ylabel("thrust [N]")
        else:
            for i in range(N):
                axs[i].plot(out["time"], out["optivar"]["thrust"][i, :])
                axs[i].plot(
                    out["time"], out["robot"]["limits"]["prop_thrust"]["ub"][i] * np.ones(out["time"].shape), "r--"
                )
                axs[i].plot(
                    out["time"], out["robot"]["limits"]["prop_thrust"]["lb"][i] * np.ones(out["time"].shape), "r--"
                )
                axs[i].set_title(out["robot"]["propellers_frame_list"][i])
                axs[i].set_xlabel("time [s]")
                axs[i].set_ylabel("thrust [N]")
        self.__save("thrust")

    def joint(self):
        out = self.out
        N = out["robot"]["ndofs"]
        if N == 0:
            return
        fig, axs = plt.subplots(N, 4, figsize=self.__figsize)
        fig.suptitle("Joint")
        for i in range(N):
            # position
            axs[i, 0].plot(out["time"], out["optivar"]["s"][i, :] * 180 / math.pi)
            axs[i, 0].plot(
                out["time"],
                out["robot"]["limits"]["joint_pos"]["ub"][i] * np.ones(out["time"].shape) * 180 / math.pi,
                "r--",
            )
            axs[i, 0].plot(
                out["time"],
                out["robot"]["limits"]["joint_pos"]["lb"][i] * np.ones(out["time"].shape) * 180 / math.pi,
                "r--",
            )
            axs[i, 0].set_title(out["robot"]["joint_list"][i])
            axs[i, 0].set_xlabel("time [s]")
            axs[i, 0].set_ylabel("position [deg]")
            # velocity
            axs[i, 1].plot(out["time"], out["optivar"]["dot_s"][i, :] * 180 / math.pi)
            axs[i, 1].plot(
                out["time"],
                out["robot"]["limits"]["joint_vel"]["ub"][i] * np.ones(out["time"].shape) * 180 / math.pi,
                "r--",
            )
            axs[i, 1].plot(
                out["time"],
                out["robot"]["limits"]["joint_vel"]["lb"][i] * np.ones(out["time"].shape) * 180 / math.pi,
                "r--",
            )
            axs[i, 1].set_title(out["robot"]["joint_list"][i])
            axs[i, 1].set_xlabel("time [s]")
            axs[i, 1].set_ylabel("velocity [deg/s]")
            # acceleration
            if out["optivar"]["ddot_s"].sum() == 0:
                axs[i, 2].plot(out["time"], out["pp"]["ddot_s"][i, :] * 180 / math.pi)
            else:
                axs[i, 2].plot(out["time"], out["optivar"]["ddot_s"][i, :] * 180 / math.pi)
            axs[i, 2].set_title(out["robot"]["joint_list"][i])
            axs[i, 2].set_xlabel("time [s]")
            axs[i, 2].set_ylabel("acceleration [deg/s^2]")
            # torque
            axs[i, 3].plot(out["time"], out["pp"]["torque"][i, :])
            axs[i, 3].plot(
                out["time"], out["robot"]["limits"]["joint_tor"]["ub"][i] * np.ones(out["time"].shape), "r--"
            )
            axs[i, 3].plot(
                out["time"], out["robot"]["limits"]["joint_tor"]["lb"][i] * np.ones(out["time"].shape), "r--"
            )

            axs[i, 3].plot(out["time"], out["pp"]["static_torque"][i, :])
            axs[i, 3].set_title(out["robot"]["joint_list"][i])
            axs[i, 3].set_xlabel("time [s]")
            axs[i, 3].set_ylabel("torque [Nm]")
        self.__save("joint")

    def plot_moving(self):
        fps = 60
        out = self.out
        figsize = (3, 3)
        background_color = [0.160, 0.160, 0.164]
        text_color = [1, 1, 1]
        # plt.style.use('dark_background')

        def update(frame):
            x = X[:frame]
            y = Y[:frame, :]
            for i, l in enumerate(line):
                l.set_xdata(x[:frame])
                l.set_ydata(y[:, i])
            return line

        # joint
        fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
        ax.set_facecolor(background_color)
        X = np.arange(out["time"][0], out["time"][-1], 1 / fps)
        Y = utils_muav.interp_2d(X, out["time"], out["optivar"]["s"]).T * 180 / math.pi
        line = plt.plot(X, Y)
        ax.set_xlabel("time [s]", color=text_color)
        ax.set_ylabel("position [deg]", color=text_color)
        ax.set(xlim=[X[0], X[-1]])
        plt.legend(["r.sweep", "r.incid.", "l.sweep", "l.incid."], loc="upper right")
        plt.tight_layout()
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
        ax.tick_params(axis="x", colors=text_color)
        ax.tick_params(axis="y", colors=text_color)
        ani = FuncAnimation(fig=fig, func=update, frames=len(X), blit=True, interval=1000 / fps)
        video_writer = ani.save("joint_pos.mp4", fps=fps / 2, extra_args=["-vcodec", "libx264"], dpi=400)

        # thrust
        fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
        ax.set_facecolor(background_color)
        X = np.arange(out["time"][0], out["time"][-1], 1 / fps)
        Y = utils_muav.interp_2d(X, out["time"], out["optivar"]["thrust"]).T
        line = plt.plot(X, Y)
        ax.set_xlabel("time [s]", color=text_color)
        ax.set_ylabel("force [N]", color=text_color)
        ax.set(xlim=[X[0], X[-1]])
        plt.tight_layout()
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
        ax.tick_params(axis="x", colors=text_color)
        ax.tick_params(axis="y", colors=text_color)
        ani = FuncAnimation(fig=fig, func=update, frames=len(X), blit=True, interval=1000 / fps)
        video_writer = ani.save("joint_thrust.mp4", fps=fps / 2, extra_args=["-vcodec", "libx264"], dpi=400)

    def base_3d(self):
        out = self.out
        fig = plt.figure(figsize=self.__figsize)
        ax = plt.axes(projection="3d")
        # 3D
        ax.plot3D(
            out["optivar"]["pos_w_b"][0, :],
            out["optivar"]["pos_w_b"][1, :],
            out["optivar"]["pos_w_b"][2, :],
            color="black",
        )
        ax.plot3D(0, 0, 0, marker="o", color="blue")
        for goal in out["constant"]["goals"]:
            if bool(goal.position):
                ax.plot3D(
                    goal.position["value"][0],
                    goal.position["value"][1],
                    goal.position["value"][2],
                    marker="o",
                    color="green",
                )
        for obstacle in out["constant"]["obstacles"]:
            if type(obstacle) == Obstacle_Sphere:
                u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
                x = obstacle.xyz[0] + np.cos(u) * np.sin(v) * obstacle.r
                y = obstacle.xyz[1] + np.sin(u) * np.sin(v) * obstacle.r
                z = obstacle.xyz[2] + np.cos(v) * obstacle.r
                ax.plot_surface(x, y, z, color="r", alpha=0.7, linewidth=0)
            elif type(obstacle) == Obstacle_InfiniteCylinder:
                z = np.linspace(out["optivar"]["pos_w_b"][2, :].min(), out["optivar"]["pos_w_b"][2, :].max(), 20)
                theta = np.linspace(0, 2 * np.pi, 20)
                theta_grid, z_grid = np.meshgrid(theta, z)
                x_grid = obstacle.r * np.cos(theta_grid) + obstacle.xy[0]
                y_grid = obstacle.r * np.sin(theta_grid) + obstacle.xy[1]
                ax.plot_surface(x_grid, y_grid, z_grid, color="r", alpha=0.7, linewidth=0)
        ax.set_title("position")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.axis("equal")
        ax.view_init(40, -140)
        self.__save("base_3d")

    def base(self):
        out = self.out
        fig, axs = plt.subplots(3, 2, figsize=self.__figsize)
        fig.suptitle("base")
        # pos
        axs[0, 0].plot(out["time"], out["optivar"]["pos_w_b"].T)
        axs[0, 0].set_title("position")
        axs[0, 0].set_xlabel("time [s]")
        axs[0, 0].set_ylabel("[m]")
        axs[0, 0].legend(["x", "y", "z"])
        # lin-vel
        axs[1, 0].plot(out["time"], out["optivar"]["twist_w_b"][:3, :].T)
        axs[1, 0].set_title("lin velocity")
        axs[1, 0].set_xlabel("time [s]")
        axs[1, 0].set_ylabel("[m/s]")
        axs[1, 0].legend(["x", "y", "z"])
        # lin-acc
        axs[2, 0].plot(out["time"], out["optivar"]["dot_twist_w_b"][:3, :].T)
        axs[2, 0].set_title("lin acceleration")
        axs[2, 0].set_xlabel("time [s]")
        axs[2, 0].set_ylabel("[m/s**2]")
        axs[2, 0].legend(["x", "y", "z"])
        # rpy
        axs[0, 1].plot(out["time"], out["pp"]["rpy_w_b"].T * 180 / math.pi)
        axs[0, 1].set_title("angle")
        axs[0, 1].set_xlabel("time [s]")
        axs[0, 1].set_ylabel("[deg]")
        axs[0, 1].legend(["r", "p", "y"])
        # ang-vel
        axs[1, 1].plot(out["time"], out["optivar"]["twist_w_b"][3:, :].T * 180 / math.pi)
        axs[1, 1].set_title("ang velocity")
        axs[1, 1].set_xlabel("time [s]")
        axs[1, 1].set_ylabel("[deg/s]")
        axs[1, 1].legend(["x", "y", "z"])
        # ang-acc
        axs[2, 1].plot(out["time"], out["optivar"]["dot_twist_w_b"][3:, :].T * 180 / math.pi)
        axs[2, 1].set_title("ang acceleration")
        axs[2, 1].set_xlabel("time [s]")
        axs[2, 1].set_ylabel("[deg/s**2]")
        axs[2, 1].legend(["x", "y", "z"])
        #
        self.__save("base")

    def centroidal(self):
        out = self.out
        fig = plt.figure(figsize=self.__figsize)
        # pos
        ax1 = plt.subplot(321)
        ax1.plot(out["time"], out["pp"]["pos_w_com"].T)
        ax1.set_title("position")
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("[m]")
        ax1.legend(["x", "y", "z"])
        # lin-vel
        ax2 = plt.subplot(323)
        ax2.plot(out["time"], out["optivar"]["vel_w_com"].T)
        ax2.set_title("lin velocity")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("[m/s]")
        ax2.legend(["x", "y", "z"])
        # lin-acc
        ax3 = plt.subplot(325)
        ax3.plot(out["time"], out["optivar"]["acc_w_com"].T)
        ax3.set_title("lin acceleration")
        ax3.set_xlabel("time [s]")
        ax3.set_ylabel("[m/s2]")
        ax3.legend(["x", "y", "z"])
        # angMom_w
        ax4 = plt.subplot(222)
        ax4.plot(out["time"], out["optivar"]["angMom_w"].T)
        ax4.set_title("angMom")
        ax4.set_xlabel("time [s]")
        ax4.set_ylabel("[Nms]")
        ax4.legend(["rx", "ry", "rz"])
        # ang-vel
        ax5 = plt.subplot(224)
        ax5.plot(out["time"], out["optivar"]["dot_angMom_w"].T)
        ax5.set_title("dot_angMom")
        ax5.set_xlabel("time [s]")
        ax5.set_ylabel("[Nm]")
        ax5.legend(["rx", "ry", "rz"])
        self.__save("centroidal")

    def com_bodies(self):
        out = self.out
        fig, axs = plt.subplots(3, 1, figsize=self.__figsize)
        # pos_b_com
        pos_b_com = np.zeros(out["optivar"]["pos_w_b"].shape)
        for k, p in enumerate(out["optivar"]["pos_w_b"].T):
            pos_w_b = out["optivar"]["pos_w_b"][:, k]
            pos_w_com = out["pp"]["pos_w_com"][:, k]
            quat_w_b = out["optivar"]["quat_w_b"][:, k]
            rotm_b_w = SO3(quat_w_b).as_matrix().full().T
            pos_b_com[:, k] = rotm_b_w @ (pos_w_com - pos_w_b)
        for j, title in enumerate(["pos_b_com_x", "pos_b_com_y", "pos_b_com_z"]):
            axs[j].plot(out["time"], pos_b_com[j, :].T)
            axs[j].plot(out["time"], out["pp"]["pos_b_cop"][j, :].T)
            axs[j].set_title(title)
            axs[j].set_xlabel("time [s]")
            axs[j].set_ylabel("[m]")
            axs[j].legend(["com", "cop"])
        self.__save("com_bodies")

    def frame_bodies(self):
        out = self.out
        naero = out["robot"]["naero"]
        fig, axs = plt.subplots(3, naero, figsize=self.__figsize)
        # pos_b_com
        pos_b_aero = np.zeros(out["pp"]["pos_w_aero"].shape)
        for k, p in enumerate(out["optivar"]["pos_w_b"].T):
            pos_w_b = out["optivar"]["pos_w_b"][:, k]
            pos_w_com = out["pp"]["pos_w_com"][:, k]
            quat_w_b = out["optivar"]["quat_w_b"][:, k]
            rotm_b_w = SO3(quat_w_b).as_matrix().full().T
            for i in range(naero):
                pos_w_aero = out["pp"]["pos_w_aero"][3 * i : 3 * i + 3, k]
                pos_b_aero[3 * i : 3 * i + 3, k] = rotm_b_w @ (pos_w_b - pos_w_aero)
        for j, xyz in enumerate(["pos_b_com_x [m]", "pos_b_com_y [m]", "pos_b_com_z [m]"]):
            for k, aero_frame in enumerate(out["robot"]["aero_frame_list"]):
                if naero > 1:
                    axs[j, k].plot(out["time"], pos_b_aero[3 * k + j : 3 * k + j + 1, :].T)
                    if j == 0:
                        axs[j, k].set_title(aero_frame)
                    if k == 0:
                        axs[j, k].set_xlabel("time [s]")
                        axs[j, k].set_ylabel(xyz)
                else:
                    axs[j].plot(out["time"], pos_b_aero[3 * k + j : 3 * k + j + 1, :].T)
                    if j == 0:
                        axs[j].set_title(aero_frame)
                    if k == 0:
                        axs[j].set_xlabel("time [s]")
                        axs[j].set_ylabel(xyz)
        self.__save("frame_bodies")
        fig = plt.figure(figsize=self.__figsize)
        ax = plt.axes(projection="3d")
        for k, aero_frame in enumerate(out["robot"]["aero_frame_list"]):
            ax.plot3D(
                pos_b_aero[3 * k : 3 * k + 1, :].T,
                pos_b_aero[3 * k + 1 : 3 * k + 2, :].T,
                pos_b_aero[3 * k + 2 : 3 * k + 3, :].T,
            )
        ax.set_title("position")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.axis("equal")
        ax.view_init(40, -140)
        self.__save("frame_bodies_3d")

    def wind(self):
        out = self.out
        naero = out["robot"]["naero"]
        fig, axs = plt.subplots(4, 2, figsize=self.__figsize)
        fig.suptitle("Aero Wrench (moments CoM)")
        T0 = ["x", "y", "z"]
        T1 = ["rx", "ry", "rz"]
        vcat_wrenchAero_w_com = np.zeros(out["optivar"]["vcat_wrenchAero_w"].shape)
        for k, p in enumerate(out["optivar"]["pos_w_b"].T):
            quat_w_b = out["optivar"]["quat_w_b"][:, k]
            rotm_b_w = SO3(quat_w_b).as_matrix().full().T
            pos_w_com = out["pp"]["pos_w_com"][:, k]
            for i in range(naero):
                pos_w_aero = out["pp"]["pos_w_aero"][3 * i : 3 * i + 3, k]
                force_w_aero = out["optivar"]["vcat_wrenchAero_w"][6 * i : 6 * i + 3, k]
                moment_w_aero = out["optivar"]["vcat_wrenchAero_w"][6 * i + 3 : 6 * i + 6, k]
                vcat_wrenchAero_w_com[6 * i : 6 * i + 3, k] = force_w_aero
                vcat_wrenchAero_w_com[6 * i + 3 : 6 * i + 6, k] = moment_w_aero + np.cross(
                    force_w_aero, pos_w_com - pos_w_aero
                )
        for i in range(3):
            sum_forces = 0
            sum_moments = 0
            for j, aero_frame in enumerate(out["robot"]["aero_frame_list"]):
                axs[i, 0].plot(out["time"], vcat_wrenchAero_w_com[i + 6 * j, :].T, label=aero_frame)
                sum_forces += vcat_wrenchAero_w_com[i + 6 * j, :].T
            axs[i, 0].plot(out["time"], sum_forces, label="total", linestyle="--")
            axs[i, 0].set_title(T0[i])
            axs[i, 0].set_xlabel("time [s]")
            axs[i, 0].set_ylabel("[N]")
            axs[i, 0].legend(loc="upper right")
            for j, aero_frame in enumerate(out["robot"]["aero_frame_list"]):
                axs[i, 1].plot(out["time"], vcat_wrenchAero_w_com[i + 3 + 6 * j, :].T, label=aero_frame)
                sum_moments += vcat_wrenchAero_w_com[i + 3 + 6 * j, :].T
            axs[i, 1].plot(out["time"], sum_moments, label="total", linestyle="--")
            axs[i, 1].set_title(T1[i])
            axs[i, 1].set_xlabel("time [s]")
            axs[i, 1].set_ylabel("[Nm]")
            axs[i, 1].legend(loc="upper right")

        for j, aero_frame in enumerate(out["robot"]["aero_frame_list"]):
            axs[3, 0].plot(out["time"], out["optivar"]["vcat_alpha"][j].T * 180 / math.pi, label=aero_frame)
            axs[3, 1].plot(out["time"], out["optivar"]["vcat_beta"][j].T * 180 / math.pi, label=aero_frame)
        axs[3, 0].set_title("alpha")
        axs[3, 0].set_xlabel("time [s]")
        axs[3, 0].set_ylabel("[deg]")
        axs[3, 0].legend(loc="upper right")
        axs[3, 1].set_title("beta")
        axs[3, 1].set_xlabel("time [s]")
        axs[3, 1].set_ylabel("[deg]")
        axs[3, 1].legend(loc="upper right")
        self.__save("wind")

    def wind_body(self):
        out = self.out
        naero = out["robot"]["naero"]
        vcat_wrenchAero_b_com = np.zeros(out["optivar"]["vcat_wrenchAero_w"].shape)
        for k, p in enumerate(out["optivar"]["pos_w_b"].T):
            quat_w_b = out["optivar"]["quat_w_b"][:, k]
            rotm_b_w = SO3(quat_w_b).as_matrix().full().T
            pos_w_com = out["pp"]["pos_w_com"][:, k]
            for i in range(naero):
                pos_w_aero = out["pp"]["pos_w_aero"][3 * i : 3 * i + 3, k]
                force_w_aero = out["optivar"]["vcat_wrenchAero_w"][6 * i : 6 * i + 3, k]
                moment_w_aero = out["optivar"]["vcat_wrenchAero_w"][6 * i + 3 : 6 * i + 6, k]
                vcat_wrenchAero_b_com[6 * i : 6 * i + 3, k] = rotm_b_w @ force_w_aero
                vcat_wrenchAero_b_com[6 * i + 3 : 6 * i + 6, k] = rotm_b_w @ (
                    moment_w_aero + np.cross(force_w_aero, pos_w_com - pos_w_aero)
                )
        fig, axs = plt.subplots(4, 2, figsize=self.__figsize)
        fig.suptitle("Aero Wrench Base Frame (Moments CoM)")
        T0 = ["x", "y", "z"]
        T1 = ["rx", "ry", "rz"]
        for i in range(3):
            sum_forces = 0
            sum_moments = 0
            for j, aero_frame in enumerate(out["robot"]["aero_frame_list"]):
                axs[i, 0].plot(out["time"], vcat_wrenchAero_b_com[i + 6 * j, :].T, label=aero_frame)
                sum_forces += vcat_wrenchAero_b_com[i + 6 * j, :].T
            axs[i, 0].plot(out["time"], sum_forces, label="total", linestyle="--")
            axs[i, 0].set_title(T0[i])
            axs[i, 0].set_xlabel("time [s]")
            axs[i, 0].set_ylabel("[N]")
            axs[i, 0].legend(loc="upper right")
            for j, aero_frame in enumerate(out["robot"]["aero_frame_list"]):
                axs[i, 1].plot(out["time"], vcat_wrenchAero_b_com[i + 3 + 6 * j, :].T, label=aero_frame)
                sum_moments += vcat_wrenchAero_b_com[i + 3 + 6 * j, :].T
            axs[i, 1].plot(out["time"], sum_moments, label="total", linestyle="--")
            axs[i, 1].set_title(T1[i])
            axs[i, 1].set_xlabel("time [s]")
            axs[i, 1].set_ylabel("[Nm]")
            axs[i, 1].legend(loc="upper right")

        for j, aero_frame in enumerate(out["robot"]["aero_frame_list"]):
            axs[3, 0].plot(out["time"], out["optivar"]["vcat_alpha"][j].T * 180 / math.pi, label=aero_frame)
            axs[3, 1].plot(out["time"], out["optivar"]["vcat_beta"][j].T * 180 / math.pi, label=aero_frame)
        axs[3, 0].set_title("alpha")
        axs[3, 0].set_xlabel("time [s]")
        axs[3, 0].set_ylabel("[deg]")
        axs[3, 0].legend(loc="upper right")
        axs[3, 1].set_title("beta")
        axs[3, 1].set_xlabel("time [s]")
        axs[3, 1].set_ylabel("[deg]")
        axs[3, 1].legend(loc="upper right")
        self.__save("wind")

    def print_stats(self) -> str:
        string = ""
        for key1 in ["goals", "goals_speed", "obstacles", "thrust", "joint_pos", "joint_vel", "joint_tor", "energy"]:
            string += f'\n[{key1}] -> [{self.stats[key1]["um"]}]'
            for key2 in ["individual", "global"]:
                string += f"\n\t [{key2}]"
                for key3 in self.stats[key1][key2].keys():
                    string += f"\n\t\t [{key3}]"
                    if type(self.stats[key1][key2][key3]) is np.ndarray:
                        string += f"\t {np.array2string(self.stats[key1][key2][key3], precision=2, floatmode='fixed')}"
                    else:
                        string += f"\t {self.stats[key1][key2][key3]:.2f}"
        print(string)
        return string

    @staticmethod
    def show(block: bool = True):
        plt.show(block=block)


if __name__ == "__main__":
    Trajectory.pattern_cylinder_obstacles(
        N=10, x_lim=[10, 50], y_lim=[-5, 5], r_lim=[0.5, 1], seed=10, plot=True, savedir="obstacle"
    )
