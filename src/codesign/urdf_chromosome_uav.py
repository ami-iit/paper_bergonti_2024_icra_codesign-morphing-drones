import numpy as np
from core.urdf_morphing_uav import (
    Fixed_joint,
    Dihedral,
    Sweep,
    Twist,
    Propeller_UAV,
    Link_UAV,
    URDF_drone_generator,
    Controller_Parameters_UAV,
)
import utils_muav
import math
import os
import pandas as pd
from typing import Dict, Tuple, List
import copy
import multiprocessing

# ---------- Generic ---------- #


class Chromosome_Generic:
    def __init__(self, name: str = "name") -> None:
        self.key = name
        self._list_genes = None

    def get_length(self) -> float:
        length = 0
        for gene in self._list_genes:
            length += gene.get_length()
        return length

    def get_type(self) -> List:
        type = []
        for gene in self._list_genes:
            type += gene.get_type()
        assert len(type) == self.get_length()
        return type

    def get_space(self) -> List:
        space = []
        for gene in self._list_genes:
            gene_space = gene.get_space()
            if (type(gene_space[0]) == dict) | (type(gene_space[0]) == list):
                space += gene_space
            else:
                space += [gene_space]
        assert len(space) == self.get_length()
        return space

    def diff(self, list_chromosome_1: List[float], list_chromosome_2: List[float]) -> List[float]:
        d = []
        for i, gene in enumerate(self._list_genes):
            gene_list_1 = list_chromosome_1[: gene.get_length()]
            gene_list_2 = list_chromosome_2[: gene.get_length()]
            d += gene.diff(gene_list_1, gene_list_2)
            list_chromosome_1 = list_chromosome_1[gene.get_length() :]
            list_chromosome_2 = list_chromosome_2[gene.get_length() :]
        return d

    def max(self):
        max_values = []
        for gene in self._list_genes:
            max_values += gene.max()
        return max_values

    def min(self):
        min_values = []
        for gene in self._list_genes:
            min_values += gene.min()
        return min_values

    def get_random(self, clean_Chromosome=True):
        space = self.get_space()
        length = self.get_length()
        max_chromosome = np.array(self.max())
        min_chromosome = np.array(self.min())
        chromosome = min_chromosome + np.random.random(length) * (max_chromosome - min_chromosome)
        for i, gene_space in enumerate(space):
            if type(gene_space) == dict:
                gene_possible_values = np.arange(
                    gene_space["low"], gene_space["high"] + gene_space["step"], gene_space["step"]
                ).round(5)
                gene_possible_values = gene_possible_values[gene_possible_values >= gene_space["low"]]
                gene_possible_values = gene_possible_values[gene_possible_values <= gene_space["high"]]
                chromosome[i] = np.random.choice(gene_possible_values)
            elif type(gene_space) == list:
                chromosome[i] = np.random.choice(gene_space)
        if clean_Chromosome:
            chromosome = self.clean_list(chromosome)
        return chromosome

    def compute_chromosomes_distance(
        self, chromosome_1: List[float], chromosome_2: List[float], clean_Chromosomes=False
    ):
        if clean_Chromosomes:
            chromosome_1 = self.clean_list(chromosome_1)
            chromosome_2 = self.clean_list(chromosome_2)
        diff = np.array(self.diff(chromosome_1, chromosome_2))
        diff_max = np.array(self.diff(self.max(), self.min()))
        distance = np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(diff_max**2))
        return distance

    def compute_population_distance_matrix(
        self, population: List[List[float]], n_processors=12, clean_population=False
    ):
        N = len(population)
        # clean population
        if clean_population:
            with multiprocessing.Pool(processes=n_processors) as pool:
                pop = pool.map(Chromosome_Drone().clean_list, population)
        else:
            pop = population
        # create list of chromosome couples
        list_chromosome_couples = np.zeros((N * (N - 1) // 2, 2), dtype=object)
        I, J = np.triu_indices(N, k=1)
        for k, ij in enumerate(zip(I, J)):
            i = ij[0]
            j = ij[1]
            list_chromosome_couples[k, :] = [pop[i], pop[j]]
        # compute distance matrix
        distance_matrix = np.zeros((N, N))
        with multiprocessing.Pool(processes=n_processors) as pool:
            temp = pool.starmap(
                Chromosome_Drone().compute_chromosomes_distance,
                [[chromosome_couples[0], chromosome_couples[1]] for chromosome_couples in list_chromosome_couples],
            )
        for k, ij in enumerate(zip(I, J)):
            i = ij[0]
            j = ij[1]
            distance_matrix[i, j] = temp[k]
            distance_matrix[j, i] = temp[k]
        return distance_matrix

    def compute_population_diversity(self, population: List[List[float]], n_processors=12, clean_population=False):
        N = len(population)
        distance_matrix = self.compute_population_distance_matrix(population, n_processors, clean_population)
        diversity = np.sum(distance_matrix) / (N**2)
        return diversity

    def from_list_to_dict(self, chromo_list: List):
        chromo_dict = {}
        for gene in self._list_genes:
            gene_list = chromo_list[: gene.get_length()]
            if issubclass(type(gene), Chromosome_Generic):
                chromo_dict[gene.key] = gene.from_list_to_dict(gene_list)
            else:
                chromo_dict[gene.key] = gene.set_given_value_encoded(gene_list).value
            chromo_list = chromo_list[gene.get_length() :]
        return chromo_dict

    def from_dict_to_list(self, chromo_dict: Dict):
        chromo_list = []
        for gene in self._list_genes:
            if issubclass(type(gene), Chromosome_Generic):
                chromo_list += gene.from_dict_to_list(chromo_dict[gene.key])
            else:
                chromo_list += gene.set_value(chromo_dict[gene.key]).get_value_encoded()
        return chromo_list

    def clean_list(self, chromo_list: np.ndarray) -> List:
        new_chromo_list = []
        for gene in self._list_genes:
            gene_list = chromo_list[: gene.get_length()]
            if issubclass(type(gene), Chromosome_Generic):
                new_chromo_list += gene.clean_list(gene_list)
            else:
                new_chromo_list += self.round_list(gene_list)
            chromo_list = chromo_list[gene.get_length() :]
        return new_chromo_list

    @staticmethod
    def round_list(gene_array: np.ndarray) -> List:
        gene_list = list(gene_array)
        for i, g in enumerate(gene_list):
            float_g = np.round(g, 5)
            int_g = int(float_g)
            gene_list[i] = int_g if float_g == int_g else float_g
        return gene_list


class Base_Gene:
    def __init__(self) -> None:
        self.value = None
        self.type = None
        self.space = None
        self.key = "base"

    def set_value(self, value):
        self.value = value
        return self

    def get_length(self):
        return len(self.type)

    def get_space(self):
        return self.space

    def get_type(self):
        return self.type

    def get_value_encoded(self) -> List:
        return self.value

    def set_given_value_encoded(self, value: list) -> "Base_Gene":
        self.value = value
        return self

    def diff(self, list_gene_1, list_gene_2):
        range_values = np.array(self.min()) - np.array(self.max())
        range_values[range_values == 0] = 1
        gene_1_norm = np.array(list_gene_1) / range_values
        gene_2_norm = np.array(list_gene_2) / range_values
        return list(gene_1_norm - gene_2_norm)

    def max(self):
        space = self.get_space()
        max_values = []
        for s in space:
            if type(s) == dict:
                max_values.append(s["high"])
            else:
                max_values.append(np.array(s).max())
        return self.round_list(max_values)

    def min(self):
        space = self.get_space()
        min_values = []
        for s in space:
            if type(s) == dict:
                min_values.append(s["low"])
            else:
                min_values.append(np.array(s).min())
        return self.round_list(min_values)

    @staticmethod
    def round_list(gene_array: np.ndarray) -> List:
        gene_list = list(gene_array)
        for i, g in enumerate(gene_list):
            float_g = np.round(g, 5)
            int_g = int(float_g)
            gene_list[i] = int_g if float_g == int_g else float_g
        return gene_list


class Base_Gene_Enum(Base_Gene):
    def __init__(self) -> None:
        super().__init__()
        self.type = [int]
        self.key = "base_enum"
        self._possible_values = [""]

    def get_space(self):
        low = 0
        high = len(self._possible_values) - 1
        return [{"low": low, "high": high, "step": 1}] if high != low else [low]

    def get_value_encoded(self) -> List:
        return [self._possible_values.index(self.value)]

    def set_given_value_encoded(self, value_encoded: list) -> "Base_Gene_Enum":
        self.value = self._possible_values[int(value_encoded[0])]
        return self

    def get_list_possible_values(self):
        return self._possible_values


# ---------- Wing ---------- #


class Chromosome_Wing(Chromosome_Generic):
    def __init__(self, name_wing: str = "wing") -> None:
        self.key = name_wing
        self._list_genes = [
            Gene_Wing_Position(),
            Gene_Wing_Orientation(),
            Gene_Wing_Airfoil(),
            Gene_Wing_Chord(),
            Gene_Wing_Aspect_Ratio(),
        ]


class Gene_Wing_Position(Base_Gene):
    def __init__(self) -> None:
        super().__init__()
        self.key = "position"
        self.type = [float, float, float]
        self.space = [{"low": -0.4, "high": -0.1, "step": 0.05}, [0.05], {"low": -0.03, "high": 0.03, "step": 0.01}]


class Gene_Wing_Orientation(Base_Gene):
    def __init__(self) -> None:
        super().__init__()
        self.key = "orientation"
        self.type = [float, float, float]
        self.space = [
            {"low": -10, "high": 10, "step": 2},
            {"low": -10, "high": 10, "step": 2},
            {"low": -10, "high": 10, "step": 2},
        ]


class Gene_Wing_Airfoil(Base_Gene_Enum):
    def __init__(self) -> None:
        super().__init__()
        self.key = "airfoil"
        self._possible_values = ["NACA 0009"]  # , "NACA 2409"]


class Gene_Wing_Chord(Base_Gene):
    def __init__(self) -> None:
        super().__init__()
        self.key = "chord"
        self.type = [float]
        self.space = [{"low": 0.1, "high": 0.4, "step": 0.05}]


class Gene_Wing_Aspect_Ratio(Base_Gene):
    def __init__(self) -> None:
        super().__init__()
        self.key = "aspect_ratio"
        self.type = [float]
        self.space = [{"low": 2, "high": 5, "step": 0.5}]


# ---------- Joint ---------- #


class Chromosome_Joint(Chromosome_Generic):
    def __init__(self, name_joint: str = "joint") -> None:
        self.key = name_joint
        self._list_genes = [Gene_Joint_Type(), Gene_Joint_ServomotorModel()]

    def clean_list(self, chromo_array: np.ndarray) -> List:
        chromo_list = self.round_list(chromo_array)
        chromo_dict = self.from_list_to_dict(chromo_list)
        if chromo_dict[Gene_Joint_Type().key] == Fixed_joint:
            chromo_dict[Gene_Joint_ServomotorModel().key] = int(Gene_Joint_ServomotorModel().get_space()[0]["high"] / 2)
        chromo_list = self.from_dict_to_list(chromo_dict)
        return chromo_list


class Gene_Joint_Type(Base_Gene_Enum):
    def __init__(self) -> None:
        super().__init__()
        self.key = "type"
        self._possible_values = [Fixed_joint, Dihedral, Sweep, Twist]

    def diff(self, list_gene_1, list_gene_2):
        d = [0] if list_gene_1[0] == list_gene_2[0] else [1]
        return d


class Gene_Joint_ServomotorModel(Base_Gene_Enum):
    def __init__(self) -> None:
        super().__init__()
        self.key = "servomotor_model"
        df = pd.read_csv(f'{utils_muav.get_repository_tree()["database_servomotor"]}/db_servomotor.csv')
        self._possible_values = list(df["id"].values)


# ---------- Propeller ---------- #


class Chromosome_Propeller(Chromosome_Generic):
    def __init__(self, name_propeller: str = "propeller") -> None:
        self.key = name_propeller
        self._list_genes = [Gene_Propeller_Active(), Gene_Propeller_Position(), Gene_Propeller_Model()]

    def clean_list(self, chromo_array: np.ndarray) -> List:
        chromo_list = self.round_list(chromo_array)
        chromo_dict = self.from_list_to_dict(chromo_list)
        if chromo_dict[Gene_Propeller_Active().key][0] == False:
            chromo_dict[Gene_Propeller_Position().key] = [0, 0, 0]
            chromo_dict[Gene_Propeller_Model().key] = int(Gene_Propeller_Model().get_space()[0]["high"] / 2)
        chromo_list = self.from_dict_to_list(chromo_dict)
        return chromo_list


class Gene_Propeller_Active(Base_Gene):
    def __init__(self) -> None:
        super().__init__()
        self.key = "active"
        self.type = [int]
        self.space = [[1]]


class Gene_Propeller_Position(Base_Gene):
    def __init__(self) -> None:
        super().__init__()
        self.key = "position"
        self.type = [float, float, float]
        self.space = [[0], [0], [0]]


class Gene_Propeller_Model(Base_Gene_Enum):
    def __init__(self) -> None:
        super().__init__()
        self.key = "propeller_model"
        df = pd.read_csv(f'{utils_muav.get_repository_tree()["database_propeller"]}/db_propeller.csv')
        self._possible_values = list(df["id"].values)


# ---------- Controller ---------- #


class Chromosome_Controller_Parameters(Chromosome_Generic):
    def __init__(self, name_wing: str = "controller_parameters") -> None:
        self.key = name_wing
        self._list_genes = [Gene_Weight_Time_Energy()]


class Gene_Weight_Time_Energy(Base_Gene_Enum):
    def __init__(self) -> None:
        super().__init__()
        self.key = "weight_time_energy"
        self._possible_values = [1, 5, 10, 20, 50, 75, 100]


# ---------- Drone ---------- #


class Chromosome_Drone(Chromosome_Generic):
    def __init__(self, name_drone: str = "drone") -> None:
        self.key = name_drone
        self._list_genes = [
            Chromosome_Wing(),
            Chromosome_Joint("joint0"),
            Chromosome_Joint("joint1"),
            Chromosome_Joint("joint2"),
            Chromosome_Propeller(),
            Chromosome_Controller_Parameters(),
        ]

    def clean_list(self, chromo_array: np.ndarray) -> List:
        chromo_list = super().clean_list(chromo_array)
        # reorder fixed joints
        new_chromo_list = []
        list_fixed_joint = []
        for i, gene in enumerate(self._list_genes):
            if isinstance(gene, Chromosome_Propeller):
                new_chromo_list += list_fixed_joint
                list_fixed_joint = []
            gene_list = chromo_list[: gene.get_length()]
            if isinstance(gene, Chromosome_Joint):
                if gene.from_list_to_dict(gene_list)[Gene_Joint_Type().key] == Fixed_joint:
                    list_fixed_joint += gene_list
                    gene_list = []
            chromo_list = chromo_list[gene.get_length() :]
            new_chromo_list += gene_list
        return new_chromo_list


def create_urdf_model(chromosome: list, overwrite: bool) -> str:
    repo_tree = utils_muav.get_repository_tree(relative_path=True)

    drone = {}
    drone["name_robot"] = "drone_ea"

    if not os.path.exists(f"{repo_tree['urdf']}/{drone['name_robot']}"):
        os.makedirs(f"{repo_tree['urdf']}/{drone['name_robot']}")
    drone["fullpath_model"] = f"{repo_tree['urdf']}/{drone['name_robot']}/{str(chromosome)}"

    if os.path.exists(f"{drone['fullpath_model']}.urdf") == 0 or overwrite:
        chromosome_dict = Chromosome_Drone().from_list_to_dict(chromosome)
        df_servomotor = pd.read_csv(f'{repo_tree["database_servomotor"]}/db_servomotor.csv')
        df_propeller = pd.read_csv(f'{repo_tree["database_propeller"]}/db_propeller.csv')

        # fuselage
        drone["fuselage"] = Link_UAV(
            mass=0.200 + 0.350,  # fuselage + battery and electronics
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
            name_pickle=os.path.join(repo_tree["database_aerodynamic_models"], "fuselage0014_tail0009_221202_101041.p"),
            pos_b_aero=drone["fuselage"].com_pos,
            rpy_b_aero=[0, math.pi, 0],
        )

        # wing
        chord = chromosome_dict["wing"]["chord"][0]
        span = chord * chromosome_dict["wing"]["aspect_ratio"][0]
        # material parameters
        density = 30.3441  # kg/m^3
        # naca 0009 parameters assuming chord = 1 and inertia = 1
        area_naca_0009 = 0.061027448  # m^2
        com_pos_naca_0009_x = 0.17218332  # m
        if chromosome_dict["wing"]["aspect_ratio"][0] == 1:
            inertia = [5.1136115e-03, 3.3792763e-03, 8.4369061e-03]
            name_aerodynamic_model_wing = "wing0009_1_0_230426.p"
        elif chromosome_dict["wing"]["aspect_ratio"][0] == 1.5:
            inertia = [1.7205956e-02, 5.0689144e-03, 2.2190898e-02]
            name_aerodynamic_model_wing = "wing0009_1_5_230426.p"
        elif chromosome_dict["wing"]["aspect_ratio"][0] == 2:
            inertia = [4.0740947e-02, 6.7585525e-03, 4.7387536e-02]
            name_aerodynamic_model_wing = "wing0009_2_0_230426.p"
        elif chromosome_dict["wing"]["aspect_ratio"][0] == 2.5:
            inertia = [7.9532800e-02, 8.4481906e-03, 8.7841036e-02]
            name_aerodynamic_model_wing = "wing0009_2_5_230426.p"
        elif chromosome_dict["wing"]["aspect_ratio"][0] == 3:
            inertia = [1.3739573e-01, 1.0137829e-02, 1.4736561e-01]
            name_aerodynamic_model_wing = "wing0009_3_0_230426.p"
        elif chromosome_dict["wing"]["aspect_ratio"][0] == 3.5:
            inertia = [2.1814395e-01, 1.1827467e-02, 2.2977548e-01]
            name_aerodynamic_model_wing = "wing0009_3_5_230426.p"
        elif chromosome_dict["wing"]["aspect_ratio"][0] == 4:
            inertia = [3.2559169e-01, 1.3517105e-02, 3.3888486e-01]
            name_aerodynamic_model_wing = "wing0009_4_0_230426.p"
        elif chromosome_dict["wing"]["aspect_ratio"][0] == 4.5:
            inertia = [4.6355314e-01, 1.5206743e-02, 4.7850797e-01]
            name_aerodynamic_model_wing = "wing0009_4_5_230426.p"
        elif chromosome_dict["wing"]["aspect_ratio"][0] == 5:
            inertia = [6.3584254e-01, 1.6896381e-02, 6.5245901e-01]
            name_aerodynamic_model_wing = "wing0009_5_0_230426.p"
        else:
            raise ValueError("Aspect ratio not available")

        # wings
        drone["wing"] = {}
        drone["wing"]["right"] = Link_UAV(
            mass=density * (area_naca_0009 * chord**2) * span,
            inertia=[
                inertia[0] * density * chord**5,  # ixx
                inertia[1] * density * chord**5,  # iyy
                inertia[2] * density * chord**5,  # izz
            ],
            com_pos=[com_pos_naca_0009_x * chord, -span / 2, 0],
            mesh="package://ros_muav/meshes/wing0009.stl",
            mesh_scale=[chord, span, chord],
            chord=chord,
            span=span,
            pos_p_c=chromosome_dict["wing"]["position"],
            rpy_p_c=[
                chromosome_dict["wing"]["orientation"][0] * math.pi / 180,
                chromosome_dict["wing"]["orientation"][1] * math.pi / 180,
                chromosome_dict["wing"]["orientation"][2] * math.pi / 180 + math.pi,
            ],
        )
        drone["wing"]["right"].set_aerodynamics(
            alpha_limits=[-12, 12],
            beta_limits=[-120, 120],
            name_pickle=os.path.join(repo_tree["database_aerodynamic_models"], name_aerodynamic_model_wing),
            pos_b_aero=drone["wing"]["right"].com_pos,
            rpy_b_aero=[math.pi, 0, 0],
        )
        drone["wing"]["left"] = copy.deepcopy(drone["wing"]["right"])
        drone["wing"]["left"].pos_p_c = [
            drone["wing"]["right"].pos_p_c[0],
            -drone["wing"]["right"].pos_p_c[1],
            drone["wing"]["right"].pos_p_c[2],
        ]
        drone["wing"]["left"].rpy_p_c = [
            -chromosome_dict["wing"]["orientation"][0] * math.pi / 180,
            -chromosome_dict["wing"]["orientation"][1] * math.pi / 180 + math.pi,
            -chromosome_dict["wing"]["orientation"][2] * math.pi / 180,
        ]
        drone["wing"]["left"].aerodynamics.rpy_b_aero = [0, 0, 0]

        # joints
        drone["joints"] = {}
        drone["joints"]["right_wing"] = []
        nj = ""
        for key in ["joint0", "joint1", "joint2"]:
            servomotor_model = df_servomotor[df_servomotor["id"] == chromosome_dict[key]["servomotor_model"]]
            R = servomotor_model["resistance"].values[0]
            kV = servomotor_model["motor_velocity_constant"].values[0]
            kI = servomotor_model["motor_torque_constant"].values[0]
            joint_obj = chromosome_dict[key]["type"]
            drone["joints"]["right_wing"].append(
                joint_obj(
                    rom=[np.deg2rad(-30), np.deg2rad(30)],
                    speed_limit=servomotor_model["speed_limit"].values[0],
                    acceleration_limit=8,
                    torque_limit=servomotor_model["torque_limit"].values[0],
                    dot_torque_limit=2 * servomotor_model["torque_limit"].values[0],
                ).set_motor_param(
                    mass=servomotor_model["mass"].values[0],
                    inertia=[
                        servomotor_model["inertia_xx"].values[0],
                        servomotor_model["inertia_yy"].values[0],
                        servomotor_model["inertia_zz"].values[0],
                    ],
                    servomotor_power_constants=[R, kV, kI],
                )
            )
            nj += joint_obj().name[0]
        drone["joints"]["left_wing"] = copy.deepcopy(drone["joints"]["right_wing"])
        for joint_obj in drone["joints"]["left_wing"]:
            if joint_obj.name == "dihedral" or joint_obj.name == "twist":
                joint_obj.reverse_rotation_axis()

        # propellers
        drone["propellers"] = []

        for i, key in enumerate(["propeller"]):
            if chromosome_dict[key]["active"][0]:
                propeller_model = df_propeller[df_propeller["id"] == chromosome_dict[key]["propeller_model"]]

                drone["propellers"].append(
                    Propeller_UAV(
                        mass=propeller_model["mass"].values[0],
                        inertia=[3.7625e-05, 3.7625e-05, 7.5e-05],
                        mesh="package://ros_muav/meshes/propeller.stl",
                        parent_link="fuselage",
                        pos=chromosome_dict[key]["position"],
                        rpy=[math.pi, -math.pi / 2, 0],
                        tag="",
                        thrust_limit=propeller_model["max_thrust"].values[0],
                        dot_thrust_limit=2 * propeller_model["max_thrust"].values[0],
                        coeff_thrust_to_power=[
                            propeller_model["power_coeff_c0"].values[0],
                            propeller_model["power_coeff_c1"].values[0],
                            propeller_model["power_coeff_c2"].values[0],
                        ],
                        ratio_torque_thrust=propeller_model["torque_thrust_ratio"].values[0],
                    ).set_tag(f"_{i}")
                )

        try:
            drone["controller_parameters"] = Controller_Parameters_UAV(
                weight_time_energy=chromosome_dict["controller_parameters"]["weight_time_energy"]
            )
        except:
            drone["controller_parameters"] = Controller_Parameters_UAV()

        udg = URDF_drone_generator(drone, print_urdf=False)
        udg.generate_urdf()
        udg.generate_toml()

    return drone["fullpath_model"]


if __name__ == "__main__":
    print("\nwing space ...")
    print(Chromosome_Wing().get_space())
    print("\nwing type ...")
    print(Chromosome_Wing().get_type())
    print("\nwing length ...")
    print(Chromosome_Wing().get_length())

    print("\njoint space ...")
    print(Chromosome_Joint().get_space())
    print("\njoint type ...")
    print(Chromosome_Joint().get_type())
    print("\njoint length ...")
    print(Chromosome_Joint().get_length())

    print("\npropeller space ...")
    print(Chromosome_Propeller().get_space())
    print("\npropeller type ...")
    print(Chromosome_Propeller().get_type())
    print("\npropeller length ...")
    print(Chromosome_Propeller().get_length())

    print("\ndrone space ...")
    print(Chromosome_Drone().get_space())
    print("\ndrone type ...")
    print(Chromosome_Drone().get_type())
    print("\ndrone length ...")
    print(Chromosome_Drone().get_length())

    l = [1, 2, 3, 2, 0, 0, 0, 0.3, 2]
    assert Chromosome_Wing().from_dict_to_list(Chromosome_Wing().from_list_to_dict(l)) == l
    l = [0, 2]
    assert Chromosome_Joint().from_dict_to_list(Chromosome_Joint().from_list_to_dict(l)) == l
    l = [True, 1.2, 0.5, 3, 5]
    assert Chromosome_Propeller().from_dict_to_list(Chromosome_Propeller().from_list_to_dict(l)) == l
    l = [-0.3, 0.05, 0.02, 0, 0, 0, 0, 0.3, 2] + [1, 2] + [2, 3] + [3, 4] + [1, 1.2, 0.5, 3, 5] + [3]
    assert Chromosome_Drone().from_dict_to_list(Chromosome_Drone().from_list_to_dict(l)) == l

    print("\nclean list ...")
    print(f"{l}\n{Chromosome_Drone().clean_list(l)}")

    print("\ncreate urdf model ...")
    create_urdf_model(l, overwrite=True)
