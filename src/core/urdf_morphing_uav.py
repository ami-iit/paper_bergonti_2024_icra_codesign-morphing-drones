from odio_urdf import *
import numpy as np
import math
import toml
import utils_muav
import os
import xml.etree.ElementTree as ET
from typing import Dict, List
from dataclasses import dataclass, field, asdict
import copy


class Drone_generic_joint:
    def define_rom(self, rom):
        self.rom = {"lower": min(rom), "upper": max(rom)}

    def reverse_rotation_axis(self):
        self.axis = list(-1 * np.array(self.axis))
        return self

    def get_axis_obj(self):
        return Axis(xyz="{} {} {}".format(self.axis[0], self.axis[1], self.axis[2]))

    def get_limit_obj(self):
        return Limit(
            effort=self.torque_limit, lower=self.rom["lower"], upper=self.rom["upper"], velocity=self.speed_limit
        )

    def get_dynamics_obj(self):
        return Dynamics(damping=self.damping, friction=self.friction)

    def get_inertial_obj(self):
        return Inertial(
            Mass(value=self.motor["mass"]),
            Origin(xyz=self.motor["pos_b_com"], rpy=self.motor["rpy_b_com"]),
            Inertia(
                ixx=self.motor["inertia"][0],
                iyy=self.motor["inertia"][1],
                izz=self.motor["inertia"][2],
                ixy="0",
                ixz="0",
                iyz="0",
            ),
        )

    def set_motor_param(
        self,
        mass: float,
        inertia: list,
        xyz: list = [0, 0, 0],
        rpy: list = [0, 0, 0],
        servomotor_power_constants: list = [1, 1, 1],  # [R,kV,kI]
        viscous_friction: float = 0,
    ):
        self.motor = {
            "mass": mass,
            "inertia": inertia,
            "pos_b_com": xyz,
            "rpy_b_com": rpy,
            "servomotor_power_constants": servomotor_power_constants,
            "viscous_friction": viscous_friction,
        }
        return self


class Fixed_joint(Drone_generic_joint):
    def __init__(self, rom=None, torque_limit=None, speed_limit=None, acceleration_limit=None, dot_torque_limit=None):
        super().__init__()
        self.type = "fixed"
        self.axis = [0, 0, 0]
        self.name = "fixed"
        self = self.set_motor_param()

    def get_limit_obj(self):
        return 0

    def get_dynamics_obj(self):
        return 0

    def set_motor_param(
        self,
        mass: float = 0,
        inertia: list = [],
        xyz: list = [],
        rpy: list = [],
        servomotor_power_constants: list = [],
        viscous_friction: float = 0,
    ):
        self.motor = {
            "mass": 0,
            "inertia": [0, 0, 0],
            "pos_b_com": [0, 0, 0],
            "rpy_b_com": [0, 0, 0],
            "servomotor_power_constants": [0, 0, 0],
            "viscous_friction": 0,
        }
        return self


class Revolute(Drone_generic_joint):
    def __init__(
        self,
        rom: list = [np.deg2rad(-10), np.deg2rad(10)],
        damping: float = 1,
        friction: float = 0,
        torque_limit: float = 50000,
        speed_limit: float = 50000,
        acceleration_limit: float = 50000,
        dot_torque_limit: float = 50000,
        name: str = "revolute",
        axis: list = [0, 0, 0],
    ):
        super().__init__()
        self.define_rom(rom)
        self.type = "revolute"
        self.axis = axis
        self.name = name
        self.damping = damping
        self.friction = friction
        self.torque_limit = abs(torque_limit)
        self.speed_limit = abs(speed_limit)
        self.acceleration_limit = abs(acceleration_limit)
        self.dot_torque_limit = abs(dot_torque_limit)


class Dihedral(Revolute):
    def __init__(
        self,
        rom: list = [np.deg2rad(-10), np.deg2rad(10)],
        damping: float = 1,
        friction: float = 0,
        torque_limit: float = 50000,
        speed_limit: float = 50000,
        acceleration_limit: float = 50000,
        dot_torque_limit: float = 50000,
    ):
        super().__init__(
            rom=rom,
            damping=damping,
            friction=friction,
            torque_limit=torque_limit,
            speed_limit=speed_limit,
            acceleration_limit=acceleration_limit,
            dot_torque_limit=dot_torque_limit,
            name="dihedral",
            axis=[1, 0, 0],
        )


class Sweep(Revolute):
    def __init__(
        self,
        rom: list = [np.deg2rad(-10), np.deg2rad(10)],
        damping: float = 1,
        friction: float = 0,
        torque_limit: float = 50000,
        speed_limit: float = 50000,
        acceleration_limit: float = 50000,
        dot_torque_limit: float = 50000,
    ):
        super().__init__(
            rom=rom,
            damping=damping,
            friction=friction,
            torque_limit=torque_limit,
            speed_limit=speed_limit,
            acceleration_limit=acceleration_limit,
            dot_torque_limit=dot_torque_limit,
            name="sweep",
            axis=[0, 0, 1],
        )


class Twist(Revolute):
    def __init__(
        self,
        rom: list = [np.deg2rad(-10), np.deg2rad(10)],
        damping: float = 1,
        friction: float = 0,
        torque_limit: float = 50000,
        speed_limit: float = 50000,
        acceleration_limit: float = 50000,
        dot_torque_limit: float = 50000,
    ):
        super().__init__(
            rom=rom,
            damping=damping,
            friction=friction,
            torque_limit=torque_limit,
            speed_limit=speed_limit,
            acceleration_limit=acceleration_limit,
            dot_torque_limit=dot_torque_limit,
            name="twist",
            axis=[0, 1, 0],
        )


class URDF_drone_generator:
    def __init__(self, drone, print_urdf: bool = True):
        self._drone = drone
        self.list_actuated_joints = []
        self.list_pos_limit_actuated_joints = {"ub": [], "lb": []}
        self.list_vel_limit_actuated_joints = []
        self.list_acc_limit_actuated_joints = []
        self.list_tor_limit_actuated_joints = []
        self.list_dot_tor_limit_actuated_joints = []
        self.list_thrust_limit = []
        self.list_dot_thrust_limit = []
        self.list_coeff_thrust_to_power = []
        self.list_ratio_torque_thrust = []
        self.list_servomotor_power_constants = []
        self.list_viscous_friction = []
        self.aero_frames = []
        self.aero_pickle_model = []
        self.aero_cross_section = []
        self.aero_chord = []
        self.aero_beta_limits = {"ub": [], "lb": []}
        self.aero_alpha_limits = {"ub": [], "lb": []}
        self.prop_frames = []
        self.collision_frames = []
        self.collision_id = 0
        self.controller_parameters = self._drone["controller_parameters"]
        self.world = "world"
        self.root_link = "root_link"
        self.__empty_inertial_obj = Inertial(
            Origin(xyz="0 0 0", rpy="0 0 0"),
            Mass(value="0"),
            Inertia(ixx="0", iyy="0", izz="0", ixy="0", ixz="0", iyz="0"),
        )
        self.odio_urdf_robot = Robot(self._drone["name_robot"])
        self._print_urdf = print_urdf
        self.build_drone()

    def build_drone(self):
        self._set_colors()
        self._root_link()
        self._fuselage(self._drone["fuselage"])

        self._wing(self._drone["wing"]["right"], self._drone["joints"]["right_wing"], "right")
        self._wing(self._drone["wing"]["left"], self._drone["joints"]["left_wing"], "left")

        # collision
        self._add_collision_frame(parent="fuselage")
        self._add_collision_frame(parent="fuselage", xyz=f'{-self._drone["fuselage"].chord} 0 0')
        self._add_collision_frame(
            parent="right_wing",
            xyz=f'{+self._drone["wing"]["right"].chord * 3 / 4} {-self._drone["wing"]["right"].span} 0',
        )
        self._add_collision_frame(
            parent="right_wing",
            xyz=f'{-self._drone["wing"]["right"].chord * 1 / 4} {-self._drone["wing"]["right"].span} 0',
        )
        self._add_collision_frame(
            parent="left_wing",
            xyz=f'{+self._drone["wing"]["left"].chord * 3 / 4} {-self._drone["wing"]["left"].span} 0',
        )
        self._add_collision_frame(
            parent="left_wing",
            xyz=f'{-self._drone["wing"]["left"].chord * 1 / 4} {-self._drone["wing"]["left"].span} 0',
        )

        # propellers
        for propeller_param in self._drone["propellers"]:
            self._propeller(propeller_param)

        if self._print_urdf:
            print(self.odio_urdf_robot)

    def _add_collision_frame(self, parent: str, xyz: str = "0 0 0"):
        child = "collision_frame_{}".format(self.collision_id)
        self.odio_urdf_robot(
            Joint(
                "fixed_joint_collision_{}".format(self.collision_id),
                Parent(parent),
                Child(child),
                Origin(rpy="0 0 0", xyz=xyz),
                type="fixed",
            ),
            Link(child),
        )
        self.collision_id += 1
        self.collision_frames.append(child)

    def _root_link(self):
        self.odio_urdf_robot(Link(self.root_link, self.__empty_inertial_obj))

    def _fuselage(self, fuselage_param: "Link_UAV"):
        self.odio_urdf_robot(
            Joint(
                "fixed_joint_base_fuselage",
                Parent(self.root_link),
                Child("fuselage"),
                Origin(rpy=fuselage_param.rpy_p_c, xyz=fuselage_param.pos_p_c),
                type="fixed",
            ),
            Link(
                "fuselage",
                Inertial(
                    Origin(xyz=fuselage_param.com_pos, rpy="0 0 0"),
                    Mass(value=fuselage_param.mass),
                    Inertia(
                        ixx=fuselage_param.inertia[0],
                        iyy=fuselage_param.inertia[1],
                        izz=fuselage_param.inertia[2],
                        ixy=0,
                        ixz=0,
                        iyz=0,
                    ),
                ),
                Visual(
                    Origin(rpy="0 0 0", xyz="0 0 0"),
                    Geometry(Mesh(filename=fuselage_param.mesh, scale=fuselage_param.mesh_scale)),
                    Material(name="grey"),
                ),
            ),
        )
        self.odio_urdf_robot(
            # frame: aero_frame_fuselage
            Joint(
                "fixed_joint_aero_frame_fuselage",
                Parent("fuselage"),
                Child("aero_frame_fuselage"),
                Origin(rpy=fuselage_param.aerodynamics.rpy_b_aero, xyz=fuselage_param.aerodynamics.pos_b_aero),
                type="fixed",
            ),
            Link("aero_frame_fuselage"),
        )
        # aerodynamics
        if fuselage_param.aerodynamics is not None:
            self.aero_frames.append("aero_frame_fuselage")
            self.aero_pickle_model.append(fuselage_param.aerodynamics.name_pickle)
            self.aero_cross_section.append(fuselage_param.aerodynamics.cross_section)
            self.aero_chord.append(fuselage_param.aerodynamics.chord)
            self.aero_alpha_limits["ub"].append(max(fuselage_param.aerodynamics.alpha_limits))
            self.aero_alpha_limits["lb"].append(min(fuselage_param.aerodynamics.alpha_limits))
            self.aero_beta_limits["ub"].append(max(fuselage_param.aerodynamics.beta_limits))
            self.aero_beta_limits["lb"].append(min(fuselage_param.aerodynamics.beta_limits))

    def _wing(self, wing_param: "Link_UAV", joint_param: list, tag: str):
        n_joint = len(joint_param)

        for i in range(0, n_joint):
            if joint_param[i].type != "fixed":
                self.list_actuated_joints.append(f"joint_{i}_{joint_param[i].name}_{tag}_wing")
                self.list_pos_limit_actuated_joints["ub"].append(joint_param[i].rom["upper"])
                self.list_pos_limit_actuated_joints["lb"].append(joint_param[i].rom["lower"])
                self.list_vel_limit_actuated_joints.append(joint_param[i].speed_limit)
                self.list_acc_limit_actuated_joints.append(joint_param[i].acceleration_limit)
                self.list_tor_limit_actuated_joints.append(joint_param[i].torque_limit)
                self.list_dot_tor_limit_actuated_joints.append(joint_param[i].dot_torque_limit)
                self.list_servomotor_power_constants.append(joint_param[i].motor["servomotor_power_constants"])
                self.list_viscous_friction.append(joint_param[i].motor["viscous_friction"])

        i = 0
        self.odio_urdf_robot(
            Joint(
                f"fixed_joint_{tag}_wing",
                Parent("fuselage"),
                Child(f"fuselage_{tag}_{i}"),
                Origin(rpy=wing_param.rpy_p_c, xyz=wing_param.pos_p_c),
                type="fixed",
            ),
            Link(f"fuselage_{tag}_{i}", joint_param[i].get_inertial_obj()),
        )

        for i in range(1, n_joint):
            self.odio_urdf_robot(
                Joint(
                    f"joint_{i - 1}_{joint_param[i - 1].name}_{tag}_wing",
                    Parent(f"fuselage_{tag}_{i - 1}"),
                    Child(f"fuselage_{tag}_{i}"),
                    Origin(rpy="0 0 0", xyz="0 0 0"),
                    joint_param[i - 1].get_axis_obj(),
                    joint_param[i - 1].get_limit_obj(),
                    joint_param[i - 1].get_dynamics_obj(),
                    type=joint_param[i - 1].type,
                ),
                Link(f"fuselage_{tag}_{i}", joint_param[i].get_inertial_obj()),
            )

        i = n_joint - 1
        self.odio_urdf_robot(
            Joint(
                f"joint_{i}_{joint_param[i].name}_{tag}_wing",
                Parent(f"fuselage_{tag}_{i}"),
                Child(f"{tag}_wing"),
                Origin(rpy="0 0 0", xyz="0 0 0"),
                joint_param[i].get_axis_obj(),
                joint_param[i].get_limit_obj(),
                joint_param[i].get_dynamics_obj(),
                type=joint_param[i].type,
            ),
            Link(
                f"{tag}_wing",
                Inertial(
                    Origin(xyz=wing_param.com_pos, rpy="0 0 0"),
                    Mass(value=wing_param.mass),
                    Inertia(
                        ixx=wing_param.inertia[0],
                        iyy=wing_param.inertia[1],
                        izz=wing_param.inertia[2],
                        ixy=0,
                        ixz=0,
                        iyz=0,
                    ),
                ),
                Visual(
                    Origin(rpy="0 0 0", xyz="0 0 0"),
                    Geometry(Mesh(filename=wing_param.mesh, scale=wing_param.mesh_scale)),
                    Material(name="red"),
                ),
            ),
        )
        if wing_param.aerodynamics is not None:
            self.odio_urdf_robot(
                # frame: aero_frame_{}_wing
                Joint(
                    f"fixed_joint_aero_frame_{tag}_wing",
                    Parent(f"{tag}_wing"),
                    Child(f"aero_frame_{tag}_wing"),
                    Origin(rpy=wing_param.aerodynamics.rpy_b_aero, xyz=wing_param.aerodynamics.pos_b_aero),
                    type="fixed",
                ),
                Link(f"aero_frame_{tag}_wing"),
            )

            self.aero_frames.append(f"aero_frame_{tag}_wing")
            self.aero_pickle_model.append(wing_param.aerodynamics.name_pickle)
            self.aero_cross_section.append(wing_param.aerodynamics.cross_section)
            self.aero_chord.append(wing_param.aerodynamics.chord)
            self.aero_alpha_limits["ub"].append(max(wing_param.aerodynamics.alpha_limits))
            self.aero_alpha_limits["lb"].append(min(wing_param.aerodynamics.alpha_limits))
            self.aero_beta_limits["ub"].append(max(wing_param.aerodynamics.beta_limits))
            self.aero_beta_limits["lb"].append(min(wing_param.aerodynamics.beta_limits))

    def _propeller(self, propeller_param: "Propeller_UAV"):
        name_joint = "fixed_joint_prop_frame_" + propeller_param.parent_link + propeller_param.tag
        name_link = "prop_frame_" + propeller_param.parent_link + propeller_param.tag
        self.odio_urdf_robot(
            # frame: prop_frame_{}
            Joint(
                name_joint,
                Parent(propeller_param.parent_link),
                Child(name_link),
                Origin(rpy=propeller_param.rpy, xyz=propeller_param.pos),
                type="fixed",
            ),
            Link(
                name_link,
                Inertial(
                    Mass(value=propeller_param.mass),
                    Origin(xyz="0 0 0", rpy="0 0 0"),
                    Inertia(
                        ixx=propeller_param.inertia[0],
                        iyy=propeller_param.inertia[1],
                        izz=propeller_param.inertia[2],
                        ixy="0",
                        ixz="0",
                        iyz="0",
                    ),
                ),
                Visual(
                    Origin(rpy="0 0 0", xyz="0 0 0"),
                    Geometry(Mesh(filename=propeller_param.mesh, scale=propeller_param.mesh_scale)),
                    Material(name="black"),
                ),
            ),
        )
        self.prop_frames.append(name_link)
        self.list_thrust_limit.append(propeller_param.thrust_limit)
        self.list_dot_thrust_limit.append(propeller_param.dot_thrust_limit)
        self.list_coeff_thrust_to_power.append(propeller_param.coeff_thrust_to_power)
        self.list_ratio_torque_thrust.append(propeller_param.ratio_torque_thrust)

    def _set_colors(self):
        self.odio_urdf_robot(Material("grey", Color(rgba="0.7 0.7 0.7 1")))
        self.odio_urdf_robot(Material("red", Color(rgba="0.7 0 0 1")))
        self.odio_urdf_robot(Material("black", Color(rgba="0.2 0.2 0.2 1")))

    def generate_urdf(self):
        print("urdf generation: started")
        tree = ET.ElementTree(ET.fromstring(str(self.odio_urdf_robot)))
        for elem1 in tree.iter():
            for elem2 in elem1.iter():
                for elem3 in elem2.iter():
                    elem3.attrib = dict(sorted(elem3.attrib.items(), key=lambda x: x[0].lower()))
        with open(self._drone["fullpath_model"] + ".urdf", "wb") as file:
            tree.write(file)
        print("urdf generation: concluded | {} generated".format(self._drone["fullpath_model"]))

    def generate_toml(self):
        out = {}
        out["world"] = self.world
        out["root_link"] = self.root_link
        out["joints"] = {}
        out["joints"]["list"] = self.list_actuated_joints
        out["joints"]["pos_limits_ub"] = self.list_pos_limit_actuated_joints["ub"]
        out["joints"]["pos_limits_lb"] = self.list_pos_limit_actuated_joints["lb"]
        out["joints"]["vel_limits_ub"] = self.list_vel_limit_actuated_joints
        out["joints"]["acc_limits_ub"] = self.list_acc_limit_actuated_joints
        out["joints"]["tor_limits_ub"] = self.list_tor_limit_actuated_joints
        out["joints"]["dot_tor_limits_ub"] = self.list_dot_tor_limit_actuated_joints
        out["joints"]["servomotor_power_constants"] = self.list_servomotor_power_constants
        out["joints"]["servomotor_viscous_friction"] = self.list_viscous_friction
        out["aerodynamics"] = {}
        out["aerodynamics"]["list_frames"] = self.aero_frames
        out["aerodynamics"]["list_model"] = self.aero_pickle_model
        out["aerodynamics"]["cross_section"] = self.aero_cross_section
        out["aerodynamics"]["chord"] = self.aero_chord
        out["aerodynamics"]["alpha_limits_lb"] = self.aero_alpha_limits["lb"]
        out["aerodynamics"]["alpha_limits_ub"] = self.aero_alpha_limits["ub"]
        out["aerodynamics"]["beta_limits_lb"] = self.aero_beta_limits["lb"]
        out["aerodynamics"]["beta_limits_ub"] = self.aero_beta_limits["ub"]
        out["propellers"] = {}
        out["propellers"]["list_frames"] = self.prop_frames
        out["propellers"]["thrust_limits_ub"] = self.list_thrust_limit
        out["propellers"]["dot_thrust_limits_ub"] = self.list_dot_thrust_limit
        out["propellers"]["coeff_thrust_to_power"] = self.list_coeff_thrust_to_power
        out["propellers"]["ratio_torque_thrust"] = self.list_ratio_torque_thrust
        out["collisions"] = {}
        out["collisions"]["list_frames"] = self.collision_frames
        out["controller_parameters"] = asdict(self.controller_parameters)
        f = open(self._drone["fullpath_model"] + ".toml", "w")
        toml.dump(out, f, encoder=toml.TomlNumpyEncoder())
        f.close()


@dataclass
class Link_UAV:
    mass: float
    inertia: list
    com_pos: list
    chord: float
    span: float
    pos_p_c: list
    rpy_p_c: list
    mesh: str
    mesh_scale: List[float] = field(default_factory=lambda: [0.001, 0.001, 0.001])

    def __post_init__(self):
        self.aerodynamics = None

    def set_aerodynamics(
        self,
        alpha_limits: list,
        beta_limits: list,
        name_pickle: str,
        cross_section: float = None,
        pos_b_aero: list = [0, 0, 0],
        rpy_b_aero: list = [0, 0, 0],
    ):
        if cross_section is None:
            cross_section = self.span * self.chord
        self.aerodynamics = Aerodynamics_Link_UAV(
            alpha_limits=alpha_limits,
            beta_limits=beta_limits,
            name_pickle=name_pickle,
            cross_section=cross_section,
            chord=self.chord,
            pos_b_aero=pos_b_aero,
            rpy_b_aero=rpy_b_aero,
        )


@dataclass
class Aerodynamics_Link_UAV:
    alpha_limits: float
    beta_limits: float
    cross_section: float
    chord: float
    name_pickle: str
    pos_b_aero: list
    rpy_b_aero: list


@dataclass
class Propeller_UAV:
    mass: float
    inertia: List
    mesh: str
    pos: List
    rpy: List
    parent_link: str
    tag: str
    thrust_limit: float = 50000
    dot_thrust_limit: float = 50000
    coeff_thrust_to_power: List = field(default_factory=lambda: [0, 20, 2])
    ratio_torque_thrust: float = 0
    mesh_scale: List[float] = field(default_factory=lambda: [0.001, 0.001, 0.001])

    def set_pos(self, pos):
        self.pos = pos
        return self

    def set_rpy(self, rpy):
        self.rpy = rpy
        return self

    def set_tag(self, tag):
        self.tag = tag
        return self


@dataclass
class Controller_Parameters_UAV:
    weight_time_energy: float = 10


if __name__ == "__main__":
    for list_joints in [[Fixed_joint], [Dihedral, Sweep, Twist], [Sweep, Twist], [Twist], [Sweep], [Dihedral]]:
        for type_prop in [1, 4]:
            repo_tree = utils_muav.get_repository_tree()
            drone = {}

            # fuselage
            drone["fuselage"] = Link_UAV(
                mass=0.2 + 0.170,  # fuselage + battery
                inertia=[2.9015955e-04, 7.4080147e-03, 7.4511885e-03],
                com_pos=[-3.44e-01, 0, 0],
                mesh="package://ros_muav/meshes/fuselage0014_tail0009.stl",
                chord=0.75,
                span=0.1,
                pos_p_c=[0, 0, 0],
                rpy_p_c=[math.pi, 0, 0],
            )
            drone["fuselage"].set_aerodynamics(
                alpha_limits=[-12, 12],
                beta_limits=[-30, 30],
                name_pickle=os.path.join(
                    repo_tree["database_aerodynamic_models"], "fuselage0014_tail0009_221202_101041.p"
                ),
                pos_b_aero=drone["fuselage"].com_pos,
                rpy_b_aero=[0, math.pi, 0],
            )

            # wings
            drone["wing"] = {}
            drone["wing"]["right"] = Link_UAV(
                mass=0.1,
                inertia=[3.0041278e-03, 4.9835443e-04, 3.4942271e-03],  # [ixx, iyy, izz]
                com_pos=[5.16e-02, -3e-01, 0],  # [x, y, z]
                mesh="package://ros_muav/meshes/wing_naca0009.stl",
                chord=0.3,
                span=0.6,
                pos_p_c=[-0.25, 0.05, 0],
                rpy_p_c=[0, 0, math.pi],
            )
            drone["wing"]["right"].set_aerodynamics(
                alpha_limits=[-12, 12],
                beta_limits=[-120, 120],
                name_pickle=os.path.join(repo_tree["database_aerodynamic_models"], "wing0009_2_0_230426.p"),
                pos_b_aero=drone["wing"]["right"].com_pos,
                rpy_b_aero=[math.pi, 0, 0],
            )
            drone["wing"]["left"] = copy.deepcopy(drone["wing"]["right"])
            drone["wing"]["left"].pos_p_c = [
                drone["wing"]["right"].pos_p_c[0],
                -drone["wing"]["right"].pos_p_c[1],
                drone["wing"]["right"].pos_p_c[2],
            ]
            drone["wing"]["left"].rpy_p_c = [0, math.pi, 0]
            drone["wing"]["left"].aerodynamics.rpy_b_aero = [0, 0, 0]

            # joints
            drone["joints"] = {}
            drone["joints"]["right_wing"] = []
            nj = ""
            for joint_obj in list_joints:
                drone["joints"]["right_wing"].append(
                    joint_obj(
                        rom=[np.deg2rad(-30), np.deg2rad(30)],
                        speed_limit=np.deg2rad(30),
                        acceleration_limit=8,
                        torque_limit=1.5,
                        dot_torque_limit=2 * 1.5,
                    ).set_motor_param(
                        mass=0.082,
                        inertia=[1.8e-05, 1.8e-05, 1.8e-05],
                        servomotor_power_constants=[5.2, 0.67, 1.30],
                        viscous_friction=1e-5,
                    )
                )
                nj += joint_obj().name[0]
            drone["joints"]["left_wing"] = copy.deepcopy(drone["joints"]["right_wing"])
            for joint_obj in drone["joints"]["left_wing"]:
                if joint_obj.name == "dihedral" or joint_obj.name == "twist":
                    joint_obj.reverse_rotation_axis()

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
                    thrust_limit=4.7,
                    dot_thrust_limit=2 * 4.7,
                    coeff_thrust_to_power=[0.0, 20.092, 1.632],
                )
            )

            if type_prop == 4:
                drone["propellers"].append(copy.deepcopy(drone["propellers"][0]))
                drone["propellers"].append(copy.deepcopy(drone["propellers"][0]))
                drone["propellers"].append(copy.deepcopy(drone["propellers"][0]))
                drone["propellers"][0].set_pos([-0.1, -0.12, +0.06]).set_tag("left_low")
                drone["propellers"][1].set_pos([-0.1, +0.12, +0.06]).set_tag("right_low")
                drone["propellers"][2].set_pos([-0.1, -0.12, -0.06]).set_tag("left_up")
                drone["propellers"][3].set_pos([-0.1, +0.12, -0.06]).set_tag("right_up")

            drone["controller_parameters"] = Controller_Parameters_UAV()

            drone["name_robot"] = "drone"
            utils_muav.get_repository_tree()["urdf"]
            drone[
                "fullpath_model"
            ] = f"{utils_muav.get_repository_tree(relative_path=True)['urdf']}/drone_{nj}_t{type_prop}"
            udg = URDF_drone_generator(drone)
            udg.generate_urdf()
            udg.generate_toml()
