from core.urdf_morphing_uav import (
    URDF_drone_generator,
    Link_UAV,
    Revolute,
    Propeller_UAV,
    Fixed_joint,
    Controller_Parameters_UAV,
)
import utils_muav
import math
import os
import copy
import numpy as np
import casadi as cs
import pickle


class URDF_fixed_wing(URDF_drone_generator):
    def build_drone(self):
        self._set_colors()
        self._root_link()
        self._fuselage(self._drone["fuselage"])
        self._wing(self._drone["aileron_left"], self._drone["joints"]["aileron_left"], tag="aileron")
        self._wing(self._drone["aileron_right"], self._drone["joints"]["aileron_right"], tag="right_aileron")
        self._wing(self._drone["rudder"], self._drone["joints"]["rudder"], tag="rudder")
        self._wing(self._drone["elevator"], self._drone["joints"]["elevator"], tag="elevator")

        # propellers
        for propeller_param in self._drone["propellers"]:
            self._propeller(propeller_param)

        # collision
        self._add_collision_frame(parent="fuselage")
        self._add_collision_frame(parent="fuselage", xyz="-0.850 0 0")
        self._add_collision_frame(parent="fuselage", xyz="-0.410 0.700 -0.100")
        self._add_collision_frame(parent="fuselage", xyz="-0.410 -0.700 -0.100")

        print(self.odio_urdf_robot)


def create_pickle_aerodynamic_function(name_pickle: str = None):
    CL0 = 0.3900
    CL_alpha = 4.5321
    CL_q = 0.3180
    CL_del_e = 0.527
    CD0 = 0.0765
    CD_alpha = 0.3346
    CD_q = 0.354
    CD_del_e = 0.004
    CY0 = 0
    CY_beta = 0.033
    CY_p = -0.100
    CY_r = 0.039
    CY_del_r = 0.225
    Cl0 = 0
    Cl_beta = -0.081
    Cl_p = -0.529
    Cl_r = 0.159
    Cl_del_a = -0.453
    Cl_del_r = 0.005
    Cm0 = 0.0200
    Cm_alpha = -1.4037
    Cm_q = -0.1324
    Cm_del_e = -0.4236
    Cn0 = 0
    Cn_beta = 0.189
    Cn_p = -0.083
    Cn_r = -0.948
    Cn_del_a = -0.041
    Cn_del_r = 0.077

    alpha = cs.SX.sym("alpha")
    beta = cs.SX.sym("beta")
    V = cs.SX.sym("V")
    chord = cs.SX.sym("chord")
    span = cs.SX.sym("span")
    joint_pos = cs.SX.sym("joint_pos", 3, 1)
    del_a = joint_pos[0]
    del_r = joint_pos[1]
    del_e = joint_pos[2]

    fun = {}
    fun["CL"] = CL0 + CL_alpha * alpha + CL_del_e * del_e  # lift force (wind)
    fun["CD"] = CD0 + CD_alpha * alpha + CD_del_e * del_e  # drag force (wind)
    fun["CY"] = CY0 + CY_beta * beta + CY_del_r * del_r  # side force (wind)
    fun["Cl"] = Cl0 + Cl_del_a * del_a  # roll moment (body)
    fun["Cm"] = Cm0 + Cm_alpha * alpha + Cm_del_e * del_e  # pitch moment (body)
    fun["Cn"] = Cn0 + Cn_beta * beta + Cn_del_a * del_a + Cn_del_r * del_r  # yaw moment (body)

    for key in fun:
        fun[key] = cs.Function(key, [alpha, beta, V, joint_pos], [fun[key]], ["alpha", "beta", "V", "joint_pos"], [key])

    # save aerodynamic model
    with open(name_pickle, "wb") as f:
        pickle.dump(fun, f)


if __name__ == "__main__":
    repo_tree = utils_muav.get_repository_tree()
    name_aerodymic_model = os.path.join(repo_tree["database_aerodynamic_models"], "fixed_wing_fuselage_experiment.p")

    create_pickle_aerodynamic_function(name_pickle=name_aerodymic_model)

    drone = {}

    # fuselage
    drone["fuselage"] = Link_UAV(
        mass=1.01 - 0.135,
        inertia=[0.05, 0.05, 0.096],  # [5.9e-2, 1.2e-1, 1.7e-1],  #
        com_pos=[-3.2e-01, 0, -0.02],
        mesh="package://ros_muav/meshes/fixed_wing_fuselage.stl",
        chord=0.185,
        span=0.77,
        pos_p_c=[0, 0, 0],
        rpy_p_c=[math.pi, 0, 0],
    )
    drone["fuselage"].set_aerodynamics(
        cross_section=0.276,
        alpha_limits=[-12, 12],
        beta_limits=[-30, 30],
        name_pickle=os.path.join(repo_tree["database_aerodynamic_models"], "fixed_wing_fuselage_experiment.p"),
        pos_b_aero=[-3.2e-01, 0, -0.02],
        rpy_b_aero=[0, math.pi, 0],
    )

    # aileron
    drone["aileron_left"] = Link_UAV(
        mass=0.013,
        inertia=[9e-5, 2e-6, 9e-5],
        com_pos=[-0.02, 0, 0],
        mesh="package://ros_muav/meshes/fixed_wing_left_aileron.stl",
        chord=0.06,
        span=0.250,
        pos_p_c=[-0.364, -0.275, -0.042],
        rpy_p_c=[0, 0, 0],
    )
    # drone["aileron_left"].set_aerodynamics(
    #     cross_section=drone["aileron_left"].chord * drone["aileron_left"].span * 3,
    #     alpha_limits=[-40, 40],
    #     beta_limits=[-30, 30],
    #     name_pickle=os.path.join(repo_tree["database_aerodynamic_models"], "fixed_wing_aileron_221219_191344.p"),
    #     pos_b_aero=[-0.015, -0.130, 0],
    #     rpy_b_aero=[0, math.pi, 0],
    # )
    drone["aileron_right"] = copy.deepcopy(drone["aileron_left"])
    drone["aileron_right"].mesh = "package://ros_muav/meshes/fixed_wing_right_aileron.stl"
    drone["aileron_right"].pos_p_c[1] = -drone["aileron_right"].pos_p_c[1]
    # drone["aileron_right"].aerodynamics.pos_b_aero[1] = -drone["aileron_right"].aerodynamics.pos_b_aero[1]

    # rudder
    drone["rudder"] = Link_UAV(
        mass=0.003,
        inertia=[6e-6, 6e-6, 2e-5],
        com_pos=[-0.013, 0, -0.075],
        mesh="package://ros_muav/meshes/fixed_wing_rudder.stl",
        chord=0.030,
        span=0.180,
        pos_p_c=[-0.820, 0, 0.010],
        rpy_p_c=[0, 0, 0],
    )
    # drone["rudder"].set_aerodynamics(
    #     cross_section=drone["rudder"].chord * drone["rudder"].span * 50,
    #     alpha_limits=[-20, 20],
    #     beta_limits=[-60, 60],
    #     name_pickle=os.path.join(repo_tree["database_aerodynamic_models"], "fixed_wing_rudder_221219_215707.p"),
    #     pos_b_aero=[-0.500, 0, -0.090],
    #     rpy_b_aero=[0, math.pi, 0],
    # )

    # elevator
    drone["elevator"] = Link_UAV(
        mass=0.014,
        inertia=[2e-4, 2e-6, 2e-4],
        com_pos=[-0.02, 0, 0],
        mesh="package://ros_muav/meshes/fixed_wing_elevator.stl",
        chord=0.04,
        span=0.360,
        pos_p_c=[-0.820, 0, 0.015],
        rpy_p_c=[0, 0, 0],
    )
    # drone["elevator"].set_aerodynamics(
    #     cross_section=drone["elevator"].chord * drone["elevator"].span * 13,
    #     alpha_limits=[-40, 40],
    #     beta_limits=[-30, 30],
    #     name_pickle=os.path.join(repo_tree["database_aerodynamic_models"], "fixed_wing_elevator_221219_202531.p"),
    #     pos_b_aero=[-0.008, 0, 0],
    #     rpy_b_aero=[0, math.pi, 0],
    # )

    # joints
    drone["joints"] = {}
    drone["joints"]["aileron_left"] = []
    drone["joints"]["aileron_left"].append(
        Revolute(
            rom=[np.deg2rad(-20), np.deg2rad(20)],
            speed_limit=5.4,
            acceleration_limit=8,
            torque_limit=0.26,
            dot_torque_limit=2 * 0.26,
            name="",
            axis=[0, 1, 0],
        ).set_motor_param(
            mass=0.018, inertia=[2.13e-06, 2.13e-06, 2.13e-06], servomotor_power_constants=[3.4, 2.15, 0.35]
        )
    )
    # drone["joints"]["aileron_right"] = copy.deepcopy(drone["joints"]["aileron_left"])
    drone["joints"]["aileron_right"] = [Fixed_joint()]
    drone["joints"]["rudder"] = []
    drone["joints"]["rudder"].append(
        Revolute(
            rom=[np.deg2rad(-20), np.deg2rad(20)],
            speed_limit=5.4,
            acceleration_limit=8,
            torque_limit=0.26,
            dot_torque_limit=2 * 0.26,
            name="",
            axis=[0, 0, 1],
        ).set_motor_param(
            mass=0.018, inertia=[2.13e-06, 2.13e-06, 2.13e-06], servomotor_power_constants=[3.4, 2.15, 0.35]
        )
    )
    drone["joints"]["elevator"] = []
    drone["joints"]["elevator"].append(
        Revolute(
            rom=[np.deg2rad(-20), np.deg2rad(20)],
            speed_limit=5.4,
            acceleration_limit=8,
            torque_limit=0.26,
            dot_torque_limit=2 * 0.26,
            name="",
            axis=[0, 1, 0],
        ).set_motor_param(
            mass=0.018, inertia=[2.13e-06, 2.13e-06, 2.13e-06], servomotor_power_constants=[3.4, 2.15, 0.35]
        )
    )

    # propellers
    drone["propellers"] = []
    drone["propellers"].append(
        Propeller_UAV(
            mass=0.04,
            inertia=[3.7625e-05, 3.7625e-05, 7.5e-05],
            mesh="package://ros_muav/meshes/propeller.stl",
            parent_link="fuselage",
            pos=[0, 0, 0],
            rpy=[math.pi, -math.pi / 2, 0],
            tag="",
            thrust_limit=6.142,
            dot_thrust_limit=2 * 6.142,
            coeff_thrust_to_power=[0.0, 22.36, 2.367],
        )
    )
    drone["controller_parameters"] = Controller_Parameters_UAV()
    drone["name_robot"] = "drone"
    drone["fullpath_model"] = f"{utils_muav.get_repository_tree(relative_path=True)['urdf']}/fixed_wing_drone"
    udg = URDF_fixed_wing(drone)
    udg.generate_urdf()
    udg.generate_toml()

    drone["propellers"][0].set_pos([-0.458, 0.0, -0.076])
    drone["propellers"][0].set_rpy([math.pi, -math.pi / 2 - 25 * math.pi / 180, 0])
    drone["fullpath_model"] = f"{utils_muav.get_repository_tree(relative_path=True)['urdf']}/fixed_wing_drone_back"
    udg = URDF_fixed_wing(drone)
    udg.generate_urdf()
    udg.generate_toml()
