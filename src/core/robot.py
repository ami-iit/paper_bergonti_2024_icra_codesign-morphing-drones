import toml
import aero
from adam.casadi.computations import KinDynComputations
import casadi as cs
import numpy as np
import math
from liecasadi import SE3, SO3
import os
from typing import Dict
import utils_muav


class Robot:
    def __init__(self, fullpath_model: str):
        print("\nRobot")
        print(f"\tImporting robot...\n\t{fullpath_model}")

        self.name = os.path.basename(fullpath_model)
        self.fullpath_model = fullpath_model
        config = toml.load(fullpath_model + ".toml")
        self.world = config["world"]
        self.root_link = config["root_link"]
        self.joint_list = config["joints"]["list"]
        self.aero_frame_list = config["aerodynamics"]["list_frames"]
        self.aero_model_list = {}
        self.aero_cross_section_list = {}
        self.aero_chord_list = {}
        self.aero_alpha_limit_list = {}
        self.aero_beta_limit_list = {}
        for i, name_pickle in enumerate(config["aerodynamics"]["list_model"]):
            self.aero_model_list[self.aero_frame_list[i]] = aero.load_pickle_aerodynamic_model(name_pickle)
            self.aero_cross_section_list[self.aero_frame_list[i]] = config["aerodynamics"]["cross_section"][i]
            self.aero_chord_list[self.aero_frame_list[i]] = config["aerodynamics"]["chord"][i]
            self.aero_alpha_limit_list[self.aero_frame_list[i]] = {
                "lb": np.deg2rad(config["aerodynamics"]["alpha_limits_lb"][i]),
                "ub": np.deg2rad(config["aerodynamics"]["alpha_limits_ub"][i]),
            }
            self.aero_beta_limit_list[self.aero_frame_list[i]] = {
                "lb": np.deg2rad(config["aerodynamics"]["beta_limits_lb"][i]),
                "ub": np.deg2rad(config["aerodynamics"]["beta_limits_ub"][i]),
            }

        self.propellers_frame_list = config["propellers"]["list_frames"]
        self.collisions_frame_list = {}
        self.collisions_frame_list = config["collisions"]["list_frames"]
        self.controller_parameters = config["controller_parameters"]

        self.__servomotor_power_constants = config["joints"]["servomotor_power_constants"]
        self.__propellers_coeff_thrust_to_power = config["propellers"]["coeff_thrust_to_power"]

        print(f"\tImporting joint list...\n\t{self.joint_list}")
        print(f"\tImporting aero_frame list...\n\t{self.aero_frame_list}")
        print(f"\tImporting propellers...\n\t{self.propellers_frame_list}")
        print(f"\tImporting collisions...\n\t{self.collisions_frame_list}")

        self.kinDyn = KinDynComputations(
            urdfstring=fullpath_model + ".urdf", joints_name_list=self.joint_list, root_link=self.root_link
        )

        self.ndofs = self.kinDyn.NDoF
        self.naero = len(self.aero_frame_list)
        self.nprop = len(self.propellers_frame_list)
        self.ncoll = len(self.collisions_frame_list)

        self.limits = {
            "joint_pos": {"ub": np.array([]), "lb": np.array([])},
            "joint_vel": {"ub": np.array([]), "lb": np.array([])},
            "joint_acc": {"ub": np.array([]), "lb": np.array([])},
            "joint_tor": {"ub": np.array([]), "lb": np.array([])},
            "joint_dot_tor": {"ub": np.array([]), "lb": np.array([])},
            "prop_thrust": {"ub": np.array([]), "lb": np.array([])},
            "prop_dot_thrust": {"ub": np.array([]), "lb": np.array([])},
        }

        self.set_joint_limit(
            ub_joint_pos=config["joints"]["pos_limits_ub"],
            lb_joint_pos=config["joints"]["pos_limits_lb"],
            ub_joint_vel=config["joints"]["vel_limits_ub"],
            ub_joint_acc=config["joints"]["acc_limits_ub"],
            ub_joint_tor=config["joints"]["tor_limits_ub"],
            ub_joint_dot_tor=config["joints"]["dot_tor_limits_ub"],
        )
        self.set_propeller_limit(
            ub_thrust=config["propellers"]["thrust_limits_ub"],
            ub_dot_thrust=config["propellers"]["dot_thrust_limits_ub"],
        )

        print("\tRobot loaded.\t")
        print(f"\tmass: \t{self.kinDyn.get_total_mass():.3f} kg")

    def get_joint_consumption(self, torque: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        consumption = []
        for coeff, T, w in zip(self.__servomotor_power_constants, torque, joint_vel):
            R = coeff[0]
            kV = coeff[1]
            kI = coeff[2]
            P = w * T / (kV * kI) + R * kV / kI * T**2
            if P < 0:
                P = 0  # no energy accumulation
            consumption.append(P)
        return np.array(consumption)

    def get_propeller_consumption(self, thrust: np.ndarray) -> np.ndarray:
        consumption = []
        for coeff, T in zip(self.__propellers_coeff_thrust_to_power, thrust):
            consumption.append(coeff[0] + coeff[1] * T + coeff[2] * (T**2))
        return np.array(consumption)

    def get_global_power_consumption_fun(self):
        joint_tor = cs.SX.sym("joint_tor", self.ndofs, 1)
        joint_vel = cs.SX.sym("joint_vel", self.ndofs, 1)
        thrust = cs.SX.sym("thrust", self.nprop, 1)
        consumption = 0
        for k, coeff in enumerate(self.__servomotor_power_constants):
            T = joint_tor[k]
            w = joint_vel[k]
            R = coeff[0]
            kV = coeff[1]
            kI = coeff[2]
            P = w * T / (kV * kI) + R * kV / kI * T**2
            consumption += P * (cs.tanh(P * 1e3) + 1) / 2  # no energy accumulation
        for k, coeff in enumerate(self.__propellers_coeff_thrust_to_power):
            T = thrust[k]
            consumption += coeff[0] + coeff[1] * T + coeff[2] * (T**2)
        return cs.Function(
            "consumption",
            [thrust, joint_tor, joint_vel],
            [consumption],
            ["thrust", "joint_tor", "joint_vel"],
            ["consumption"],
        )

    def get_aerodynamic_angles_and_speed_fun(self, aero_frame):
        # base
        tform_w_b = cs.SX.sym("tform_w_b", 4, 4)
        twist_w_b = cs.SX.sym("twist_w_b", 6, 1)
        # joint
        s = cs.SX.sym("s", len(self.joint_list), 1)
        ds = cs.SX.sym("ds", len(self.joint_list), 1)
        # wind
        Vwind_w = cs.SX.sym("vwind_w", 3, 1)

        rotm_w_aero = self.kinDyn.forward_kinematics_fun(aero_frame)(tform_w_b, s)[:3, :3]
        Vbody_w = self.kinDyn.jacobian_fun(aero_frame)(tform_w_b, s) @ cs.vertcat(twist_w_b, ds)
        Vinf_w = Vwind_w - Vbody_w[:3]
        Vinf_aero = rotm_w_aero.T @ Vinf_w

        Vinf_norm = cs.norm_2(Vinf_aero)
        beta = cs.asin(Vinf_aero[1] / Vinf_norm)
        alpha = cs.atan(Vinf_aero[2] / Vinf_aero[0])

        return cs.Function(
            "aerodynamic_angles_and_speed",
            [tform_w_b, twist_w_b, s, ds, Vwind_w],
            [alpha, beta, Vinf_norm],
            ["tform_w_b", "twist_w_b", "s", "dot_s", "Vwind_w"],
            ["alpha", "beta", "Vinf_norm"],
        )

    def get_aerodynamic_wrench_fun(self, aero_frame):
        # base
        tform_w_b = cs.SX.sym("tform_w_b", 4, 4)
        # joint
        s = cs.SX.sym("s", len(self.joint_list), 1)
        # wind
        air_density = cs.SX.sym("rho", 1, 1)
        air_viscosity = cs.SX.sym("mu", 1, 1)
        alpha = cs.SX.sym("alpha", 1, 1)
        beta = cs.SX.sym("beta", 1, 1)
        Vinf_norm = cs.SX.sym("Vinf_norm", 1, 1)

        rotm_w_aero = self.kinDyn.forward_kinematics_fun(aero_frame)(tform_w_b, s)[:3, :3]

        rotm_aero_wind = (
            SO3.from_euler(cs.vertcat(0, -alpha, 0)).as_matrix() @ SO3.from_euler(cs.vertcat(0, 0, beta)).as_matrix()
        )

        A = self.aero_cross_section_list[aero_frame]
        chord = self.aero_chord_list[aero_frame]
        wingspan = A / chord

        if self.name == "fixed_wing_drone" or self.name == "fixed_wing_drone_back":
            CL = self.aero_model_list[aero_frame]["CL"](alpha, beta, Vinf_norm, s)
            CD = self.aero_model_list[aero_frame]["CD"](alpha, beta, Vinf_norm, s)
            CY = self.aero_model_list[aero_frame]["CY"](alpha, beta, Vinf_norm, s)
            L = 0.5 * A * air_density * CL * Vinf_norm**2
            D = 0.5 * A * air_density * CD * Vinf_norm**2
            Y = 0.5 * A * air_density * CY * Vinf_norm**2
            forces_wind = cs.vertcat(D, Y, L)
            forces_aero = rotm_aero_wind @ forces_wind
            Cl = self.aero_model_list[aero_frame]["Cl"](alpha, beta, Vinf_norm, s)  # roll
            Cm = self.aero_model_list[aero_frame]["Cm"](alpha, beta, Vinf_norm, s)  # pitch
            Cn = self.aero_model_list[aero_frame]["Cn"](alpha, beta, Vinf_norm, s)  # yaw
            l = 0.5 * A * air_density * Cl * Vinf_norm**2 * chord  #
            m = 0.5 * A * air_density * Cm * Vinf_norm**2 * chord
            n = 0.5 * A * air_density * Cn * Vinf_norm**2 * chord  #
            torque_aero = cs.vertcat(l, m, n)
            wrench_aero = cs.vertcat(forces_aero, torque_aero)
            wrench_world = cs.diagcat(rotm_w_aero, rotm_w_aero) @ wrench_aero
        else:
            reynolds = air_density * Vinf_norm * chord / air_viscosity
            CD = self.aero_model_list[aero_frame]["CD"](alpha, beta, reynolds)  # drag
            CL = self.aero_model_list[aero_frame]["CL"](alpha, beta, reynolds)  # lift
            CY = self.aero_model_list[aero_frame]["CY"](alpha, beta, reynolds)  # side force
            Cl = self.aero_model_list[aero_frame]["Cl"](alpha, beta, reynolds)  # roll
            Cm = self.aero_model_list[aero_frame]["Cm"](alpha, beta, reynolds)  # pitch
            Cn = self.aero_model_list[aero_frame]["Cn"](alpha, beta, reynolds)  # yaw
            D = 0.5 * A * air_density * CD * (Vinf_norm) ** 2
            L = 0.5 * A * air_density * CL * (Vinf_norm) ** 2
            Y = 0.5 * A * air_density * CY * (Vinf_norm) ** 2
            l = 0.5 * A * air_density * Cl * wingspan * (Vinf_norm) ** 2
            m = 0.5 * A * air_density * Cm * chord * (Vinf_norm) ** 2
            n = 0.5 * A * air_density * Cn * wingspan * (Vinf_norm) ** 2
            wrench_wind = cs.vertcat(D, Y, L, l, m, n)

            rotm_w_wind = rotm_w_aero @ rotm_aero_wind
            wrench_world = cs.diagcat(rotm_w_wind, rotm_w_wind) @ wrench_wind

        return cs.Function(
            "aero_wrench_w",
            [tform_w_b, s, air_density, air_viscosity, alpha, beta, Vinf_norm],
            [wrench_world],
            ["tform_w_b", "s", "air_density", "air_viscosity", "alpha", "beta", "Vinf_norm"],
            ["wrench_world"],
        )

    def get_aerodynamic_cop_fun(self):
        tform_w_b = cs.SX.sym("tform_w_b", 4, 4)
        s = cs.SX.sym("s", self.ndofs, 1)
        vcat_wrenchAero_w = cs.SX.sym("vcat_wrenchAero_w", 6 * self.naero, 1)
        pos_w_b = tform_w_b[:3, 3]
        rotm_b_w = tform_w_b[:3, :3].T

        A = np.zeros((2, 2))
        b = np.zeros(2)
        for i, aero_frame in enumerate(self.aero_frame_list):
            pos_w_aero = self.kinDyn.forward_kinematics_fun(aero_frame)(tform_w_b, s)[:3, 3]
            pos_b_aero = rotm_b_w @ (pos_w_aero - pos_w_b)
            wrenchAero_w = vcat_wrenchAero_w[6 * i : 6 * i + 6]
            forceAero_b = rotm_b_w @ wrenchAero_w[:3]
            torqueAero_b = rotm_b_w @ wrenchAero_w[3:]
            A += cs.skew(forceAero_b)[:2, :2]
            b += cs.skew(forceAero_b)[:2, :2] @ pos_b_aero[:2] + torqueAero_b[:2]
        pos_b_cop = cs.vertcat(cs.solve(A, b), cs.SX(0))
        return cs.Function(
            "pos_b_cop",
            [tform_w_b, s, vcat_wrenchAero_w],
            [pos_b_cop],
            ["tform_w_b", "s", "vcat_wrenchAero_w"],
            ["pos_b_cop"],
        )

    def get_casadi_var_robot(self):
        # base
        posquat_w_b = cs.SX.sym("posquat_w_b", 7, 1)
        twist_w_b = cs.SX.sym("twist_w_b", 6, 1)
        # joint
        s = cs.SX.sym("s", self.ndofs, 1)
        dot_s = cs.SX.sym("ds", self.ndofs, 1)
        torque = cs.SX.sym("τ", self.ndofs, 1)
        prop_thrust = cs.SX.sym("T", self.nprop, 1)
        return posquat_w_b, twist_w_b, s, dot_s, torque, prop_thrust

    def get_casadi_var_wind(self):
        Vwind_w = cs.SX.sym("vwind_w", 3, 1)
        air_density = cs.SX.sym("rho", 1, 1)
        return Vwind_w, air_density

    def get_dot_nu_fullDyn_fun(self):
        # robot
        posquat_w_b, twist_w_b, s, dot_s, torque, prop_thrust = self.get_casadi_var_robot()
        # wind
        vcat_wrenchAero_w = cs.SX.sym("vcat_wrenchAero_w", 6 * self.naero, 1)

        tform_w_b = SE3(pos=posquat_w_b[:3], xyzw=posquat_w_b[3:]).as_matrix()

        M = self.kinDyn.mass_matrix_fun()(tform_w_b, s)
        h = self.kinDyn.bias_force_fun()(tform_w_b, s, twist_w_b, dot_s)
        B = cs.vertcat(cs.SX(6, self.ndofs), cs.SX.eye(self.ndofs))

        sum_Jt_wrenchAero_w = 0
        for aero_frame in self.aero_frame_list:
            jacobianAero_w = self.kinDyn.jacobian_fun(aero_frame)(tform_w_b, s)
            wrenchAero_w = vcat_wrenchAero_w[i * 6 : i * 6 + 6]
            sum_Jt_wrenchAero_w += jacobianAero_w.T @ wrenchAero_w

        sum_Jt_wrenchPro_w = 0
        for i, prop_frame in enumerate(self.propellers_frame_list):
            jacobianProp_w = self.kinDyn.jacobian_fun(prop_frame)(tform_w_b, s)
            rotm_w_prop = self.kinDyn.forward_kinematics_fun(prop_frame)(tform_w_b, s)[:3, :3]
            wrenchProp_prop = cs.vertcat(cs.SX(2, 1), prop_thrust[i], cs.SX(3, 1))
            wrenchProp_w = cs.diagcat(rotm_w_prop, rotm_w_prop) @ wrenchProp_prop
            sum_Jt_wrenchPro_w += jacobianProp_w.T @ wrenchProp_w

        dot_nu = cs.solve(M, (sum_Jt_wrenchAero_w + sum_Jt_wrenchPro_w + B @ torque - h))

        return cs.Function(
            "dot_nu",
            [posquat_w_b, twist_w_b, s, dot_s, torque, prop_thrust, vcat_wrenchAero_w],
            [dot_nu],
            ["posquat_w_b", "twist_w_b", "s", "dot_s", "torque", "prop_thrust", "vcat_wrenchAero_w"],
            ["dot_nu"],
        )

    def get_joint_torque_static_fun(self):
        # robot
        posquat_w_b, twist_w_b, s, dot_s, dummy_torque, prop_thrust = self.get_casadi_var_robot()
        # wind
        vcat_wrenchAero_w = cs.SX.sym("vcat_wrenchAero_w", 6 * self.naero, 1)

        tform_w_b = SE3(pos=posquat_w_b[:3], xyzw=posquat_w_b[3:]).as_matrix()

        G = self.kinDyn.gravity_term_fun()(tform_w_b, s)

        sum_Jt_wrenchAero_w = 0
        for i, aero_frame in enumerate(self.aero_frame_list):
            jacobianAero_w = self.kinDyn.jacobian_fun(aero_frame)(tform_w_b, s)
            wrenchAero_w = vcat_wrenchAero_w[i * 6 : i * 6 + 6]
            sum_Jt_wrenchAero_w += jacobianAero_w.T @ wrenchAero_w

        sum_Jt_wrenchPro_w = 0
        for i, prop_frame in enumerate(self.propellers_frame_list):
            jacobianProp_w = self.kinDyn.jacobian_fun(prop_frame)(tform_w_b, s)
            rotm_w_prop = self.kinDyn.forward_kinematics_fun(prop_frame)(tform_w_b, s)[:3, :3]
            wrenchProp_prop = cs.vertcat(cs.SX(2, 1), prop_thrust[i], cs.SX(3, 1))
            wrenchProp_w = cs.diagcat(rotm_w_prop, rotm_w_prop) @ wrenchProp_prop
            sum_Jt_wrenchPro_w += jacobianProp_w.T @ wrenchProp_w

        JtW = sum_Jt_wrenchAero_w + sum_Jt_wrenchPro_w

        G1, G2 = [G[:6], G[6:]]
        JtW1, JtW2 = [JtW[:6], JtW[6:]]

        torque = G2 - JtW2

        return cs.Function(
            "static_torque",
            [posquat_w_b, s, prop_thrust, vcat_wrenchAero_w],
            [torque],
            ["posquat_w_b", "s", "prop_thrust", "vcat_wrenchAero_w"],
            ["torque"],
        )

    def get_joint_torque_fun(self):
        # robot
        posquat_w_b, twist_w_b, s, dot_s, dummy_torque, prop_thrust = self.get_casadi_var_robot()
        ddot_s = cs.SX.sym("dds", self.ndofs, 1)
        # wind
        vcat_wrenchAero_w = cs.SX.sym("vcat_wrenchAero_w", 6 * self.naero, 1)

        tform_w_b = SE3(pos=posquat_w_b[:3], xyzw=posquat_w_b[3:]).as_matrix()

        M = self.kinDyn.mass_matrix_fun()(tform_w_b, s)
        h = self.kinDyn.bias_force_fun()(tform_w_b, s, twist_w_b, dot_s)

        sum_Jt_wrenchAero_w = 0
        for i, aero_frame in enumerate(self.aero_frame_list):
            jacobianAero_w = self.kinDyn.jacobian_fun(aero_frame)(tform_w_b, s)
            wrenchAero_w = vcat_wrenchAero_w[i * 6 : i * 6 + 6]
            sum_Jt_wrenchAero_w += jacobianAero_w.T @ wrenchAero_w

        sum_Jt_wrenchPro_w = 0
        for i, prop_frame in enumerate(self.propellers_frame_list):
            jacobianProp_w = self.kinDyn.jacobian_fun(prop_frame)(tform_w_b, s)
            rotm_w_prop = self.kinDyn.forward_kinematics_fun(prop_frame)(tform_w_b, s)[:3, :3]
            wrenchProp_prop = cs.vertcat(cs.SX(2, 1), prop_thrust[i], cs.SX(3, 1))
            wrenchProp_w = cs.diagcat(rotm_w_prop, rotm_w_prop) @ wrenchProp_prop
            sum_Jt_wrenchPro_w += jacobianProp_w.T @ wrenchProp_w

        JtW = sum_Jt_wrenchAero_w + sum_Jt_wrenchPro_w

        M11, M12, M21, M22 = [M[:6, :6], M[:6, 6:], M[6:, :6], M[6:, 6:]]
        h1, h2 = [h[:6], h[6:]]
        JtW1, JtW2 = [JtW[:6], JtW[6:]]

        invM11 = cs.inv(M11)

        torque = (M22 - M21 @ invM11 @ M12) @ ddot_s + (h2 - M21 @ invM11 @ h1) - (JtW2 - M21 @ invM11 @ JtW1)

        return cs.Function(
            "torque",
            [posquat_w_b, twist_w_b, s, dot_s, ddot_s, prop_thrust, vcat_wrenchAero_w],
            [torque],
            ["posquat_w_b", "twist_w_b", "s", "dot_s", "ddot_s", "prop_thrust", "vcat_wrenchAero_w"],
            ["torque"],
        )

    def get_dot_momentum_centDyn_fun(self):
        # robot
        posquat_w_b, twist_w_b, s, dot_s, torque, prop_thrust = self.get_casadi_var_robot()
        # wind
        vcat_wrenchAero_w = cs.SX.sym("vcat_wrenchAero_w", 6 * self.naero, 1)

        tform_w_b = SE3(pos=posquat_w_b[:3], xyzw=posquat_w_b[3:]).as_matrix()

        pos_w_com = self.kinDyn.CoM_position_fun()(tform_w_b, s)
        m = self.kinDyn.get_total_mass()
        I3 = cs.SX.eye(3)
        O3 = cs.SX(3, 3)

        sum_ext_forces_w = m * self.kinDyn.g

        for i, aero_frame in enumerate(self.aero_frame_list):
            wrenchAero_w = vcat_wrenchAero_w[i * 6 : i * 6 + 6]
            pos_w_frame = self.kinDyn.forward_kinematics_fun(aero_frame)(tform_w_b, s)[:3, 3]
            S = cs.vertcat(cs.horzcat(I3, O3), cs.horzcat(cs.skew(pos_w_frame - pos_w_com), I3))
            sum_ext_forces_w += S @ wrenchAero_w

        for i, prop_frame in enumerate(self.propellers_frame_list):
            rotm_w_prop = self.kinDyn.forward_kinematics_fun(prop_frame)(tform_w_b, s)[:3, :3]
            wrenchProp_prop = cs.vertcat(cs.SX(2, 1), prop_thrust[i], cs.SX(3, 1))
            wrenchProp_w = cs.diagcat(rotm_w_prop, rotm_w_prop) @ wrenchProp_prop
            pos_w_frame = self.kinDyn.forward_kinematics_fun(prop_frame)(tform_w_b, s)[:3, 3]
            S = cs.vertcat(cs.horzcat(I3, O3), cs.horzcat(cs.skew(pos_w_frame - pos_w_com), I3))
            sum_ext_forces_w += S @ wrenchProp_w

        return cs.Function(
            "dotMomentum",
            [posquat_w_b, s, prop_thrust, vcat_wrenchAero_w],
            [sum_ext_forces_w],
            ["posquat_w_b", "s", "prop_thrust", "vcat_wrenchAero_w"],
            ["dotMomentum"],
        )

    def set_joint_limit(
        self,
        ub_joint_pos: np.ndarray = None,
        lb_joint_pos: np.ndarray = None,
        ub_joint_vel: np.ndarray = None,
        ub_joint_acc: np.ndarray = None,
        ub_joint_tor: np.ndarray = None,
        ub_joint_dot_tor: np.ndarray = None,
    ) -> None:
        if ub_joint_pos is not None:
            self.limits["joint_pos"]["ub"] = np.array(ub_joint_pos)
        if lb_joint_pos is not None:
            self.limits["joint_pos"]["lb"] = np.array(lb_joint_pos)
        if ub_joint_vel is not None:
            self.limits["joint_vel"]["ub"] = +np.array(ub_joint_vel)
            self.limits["joint_vel"]["lb"] = -np.array(ub_joint_vel)
        if ub_joint_acc is not None:
            self.limits["joint_acc"]["ub"] = +np.array(ub_joint_acc)
            self.limits["joint_acc"]["lb"] = -np.array(ub_joint_acc)
        if ub_joint_tor is not None:
            self.limits["joint_tor"]["ub"] = +np.array(ub_joint_tor)
            self.limits["joint_tor"]["lb"] = -np.array(ub_joint_tor)
        if ub_joint_dot_tor is not None:
            self.limits["joint_dot_tor"]["ub"] = +np.array(ub_joint_dot_tor)
            self.limits["joint_dot_tor"]["lb"] = -np.array(ub_joint_dot_tor)

    def set_propeller_limit(self, ub_thrust: np.ndarray = None, ub_dot_thrust: np.ndarray = None) -> None:
        self.limits["prop_thrust"]["lb"] = np.zeros(self.nprop)
        if ub_thrust is not None:
            self.limits["prop_thrust"]["ub"] = +np.array(ub_thrust)
        if ub_dot_thrust is not None:
            self.limits["prop_dot_thrust"]["ub"] = +np.array(ub_dot_thrust)
            self.limits["prop_dot_thrust"]["lb"] = -np.array(ub_dot_thrust)


if __name__ == "__main__":
    robot = Robot(f"{utils_muav.get_repository_tree(relative_path=True)['urdf']}/drone_nsga_46295d0_2")
    posquat_w_b = np.concatenate(
        (
            np.array([0, 0, 0]).T,
            np.squeeze(SO3.from_euler(np.array([0, np.deg2rad(0), np.deg2rad(0)])).as_quat().coeffs().full()),
        )
    )
    tform_w_b = SE3(posquat_w_b[:3], posquat_w_b[3:]).transform()
    s = np.zeros((robot.ndofs, 1))
    ds = np.zeros((robot.ndofs, 1))
    twist_w_b = np.array([10, 0, 0, 0, 0, 0])
    Vwind_w = np.array([0, 0, 0])
    air_density = 1.225
    air_viscosity = 1.8375e-5
    prop_thrust = np.zeros((robot.nprop, 1))
    print()
    print("total mass")
    print(robot.kinDyn.get_total_mass())
    print()
    print("CoM")
    print(robot.kinDyn.CoM_position_fun()(tform_w_b, s).full().T)
    print()
    vcat_wrenchAero_w = []
    for aero_frame in robot.aero_frame_list:
        print(aero_frame)
        alpha, beta, Vinf_norm = robot.get_aerodynamic_angles_and_speed_fun(aero_frame)(
            tform_w_b, twist_w_b, s, ds, Vwind_w
        )
        print([math.degrees(alpha), math.degrees(beta), Vinf_norm.full()])
        wrenchAero_w = robot.get_aerodynamic_wrench_fun(aero_frame)(
            tform_w_b, s, air_density, air_viscosity, alpha, beta, Vinf_norm
        )
        vcat_wrenchAero_w = cs.vcat((vcat_wrenchAero_w, wrenchAero_w))
        print(wrenchAero_w)
        print()
    print("centroidal")
    print(robot.get_dot_momentum_centDyn_fun()(posquat_w_b, s, prop_thrust, vcat_wrenchAero_w))
