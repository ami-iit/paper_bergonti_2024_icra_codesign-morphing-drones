from traj.trajectory import Goal, Postprocess, Obstacle_Sphere, Obstacle_InfiniteCylinder, Gains_Trajectory, Task
from traj.trajectory_wholeboby import Trajectory_WholeBody_Planner
from core.robot import Robot
import numpy as np
import math
from liecasadi import SO3
import utils_muav
import time
import multiprocessing
from typing import List, Dict
import pandas as pd
import os, datetime


class Database_results:
    def __init__(self, name_database: str, columns: List[str] = [""]) -> None:
        self.name_database_with_ext = name_database + ".csv"
        try:
            self.df = pd.read_csv(self.name_database_with_ext)
        except:
            self.create_empty_csv_database(columns=columns)

    def create_empty_csv_database(self, columns: List[str]) -> None:
        self.df = pd.DataFrame(columns=columns)
        self.df.to_csv(self.name_database_with_ext, index=False)

    def update(self, list_to_be_added: List) -> None:
        df = pd.DataFrame(columns=self.df.columns)
        df.loc[len(self.df)] = list_to_be_added
        df.to_csv(self.name_database_with_ext, index=False, mode="a", header=False)

    def rename(self, new_name: str) -> None:
        os.rename(self.name_database_with_ext, new_name + ".csv")
        self.name_database_with_ext = new_name + ".csv"


def define_task(
    goal_dist: float,
    goal_angl: float,
    obst_radius: float,
    obst_type: str,
    ic_speed_x: float,
    ic_roll: float,
    ic_pitch: float,
    ic_yaw: float,
) -> Task:
    knots = 2 * goal_dist + 30
    list_goals = []
    list_obstacles = []
    list_goals.append(
        Goal(index=int(knots / 2)).set_position(
            xyz=[goal_dist * np.cos(np.deg2rad(goal_angl)), goal_dist * np.sin(np.deg2rad(goal_angl)), 0],
            isStrict=True,
            param=0.1,
        )
    )
    list_goals.append(
        Goal(index=knots)
        .set_position(xyz=[2 * goal_dist * np.cos(np.deg2rad(goal_angl)), 0, 0], isStrict=True, param=0.1)
        .set_linearVelocityBody(xyz=[10, 0, 0], isStrict=False, param=5)
    )
    obs1 = [goal_dist / 2 * np.cos(np.deg2rad(goal_angl)), goal_dist / 2 * np.sin(np.deg2rad(goal_angl)), 0]
    obs2 = [3 * goal_dist / 2 * np.cos(np.deg2rad(goal_angl)), goal_dist / 2 * np.sin(np.deg2rad(goal_angl)), 0]
    if obst_radius > 0:
        if obst_type == "sphere":
            list_obstacles.append(Obstacle_Sphere(xyz=obs1, r=obst_radius))
            list_obstacles.append(Obstacle_Sphere(xyz=obs2, r=obst_radius))
        elif obst_type == "cylinder":
            list_obstacles.append(Obstacle_InfiniteCylinder(xy=obs1[:2], r=obst_radius))
            list_obstacles.append(Obstacle_InfiniteCylinder(xy=obs2[:2], r=obst_radius))
    task = Task(
        name="mt",
        knots=knots,
        list_goals=list_goals,
        list_obstacles=list_obstacles,
        ic_twist_w_b=np.array([ic_speed_x, 0, 0, 0, 0, 0]),
        ic_position_w_b=np.zeros(3),
        ic_quat_w_b=np.squeeze(
            SO3.from_euler(np.array([ic_roll, ic_pitch, ic_yaw]) * math.pi / 180).as_quat().coeffs().full()
        ),
    )
    return task


def solve_s_trajectory(
    robot_name: str,
    goal_dist: float,
    goal_angl: float,
    obst_radius: float,
    obst_type: str,
    ic_speed_x: float,
    ic_roll: float,
    ic_pitch: float,
    ic_yaw: float,
    str_date: str,
):
    t0_fitness_func = time.time()
    task = define_task(goal_dist, goal_angl, obst_radius, obst_type, ic_speed_x, ic_roll, ic_pitch, ic_yaw)
    robot = Robot(f"{utils_muav.get_repository_tree(relative_path=True)['urdf']}/{robot_name}")
    robot.set_joint_limit()
    robot.set_propeller_limit()
    traj = Trajectory_WholeBody_Planner(
        robot=robot, knots=task.knots, time_horizon=None, regularize_control_input_variations=True
    )
    traj.set_gains(
        Gains_Trajectory(
            cost_function_weight_time=robot.controller_parameters["weight_time_energy"], cost_function_weight_energy=1
        )
    )
    traj.set_wind_parameters(air_density=1.225, air_viscosity=1.8375e-5, vel_w_wind=np.array([-1, 0, 0]))
    traj.set_initial_condition(
        s=np.zeros(robot.ndofs),
        dot_s=np.zeros(robot.ndofs),
        ddot_s=np.zeros(robot.ndofs),
        pos_w_b=task.ic_position_w_b,
        quat_w_b=task.ic_quat_w_b,
        twist_w_b=task.ic_twist_w_b,
    )
    [traj.add_goal(goal) for goal in task.list_goals]
    [traj.add_obstacle(obstacle) for obstacle in task.list_obstacles]
    traj.create()

    try:
        out = traj.solve()
        traj.save(out, folder_name=f"result/{str_date}")
        pp = Postprocess(out)
    except:
        pp = Postprocess()
        out = {}

    Database_results("multiple_trajectories").update(
        [
            datetime.datetime.timestamp(datetime.datetime.now()),
            robot_name,
            goal_dist,
            goal_angl,
            obst_radius,
            obst_type,
            ic_speed_x,
            ic_roll,
            ic_pitch,
            ic_yaw,
            True if pp.out is not None else False,
            pp.stats["energy"]["global"]["joint"] + pp.stats["energy"]["global"]["propeller"],
            pp.stats["time"]["trajectory"],
            time.time() - t0_fitness_func,
            pp.stats["energy"]["global"]["propeller"],
            pp.stats["energy"]["global"]["joint"],
            traj.name_trajectory,
        ]
    )


def run_validation(dict: Dict):
    str_date = utils_muav.get_date_str()
    Database_results("multiple_trajectories").create_empty_csv_database(
        columns=[
            "timestamp",
            "name_drone",
            "distance",
            "angle",
            "radius",
            "type_obstacle",
            "initial_speed_x",
            "initial_roll",
            "initial_pitch",
            "initial_yaw",
            "success",
            "energy",
            "time",
            "computational_time",
            "energy_propeller",
            "energy_joint",
            "traj_name",
        ]
    )
    db_ff = Database_results(name_database="multiple_trajectories")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(
            solve_s_trajectory,
            [
                (
                    robot_name,
                    goal_dist,
                    goal_angl,
                    obst_radius,
                    type_obstacle,
                    ic_speed_x,
                    ic_roll,
                    ic_pitch,
                    ic_yaw,
                    str_date,
                )
                for robot_name in dict["robot"]
                for goal_dist in dict["goal_dist"]
                for goal_angl in dict["goal_angl"]
                for obst_radius in dict["obst_radius"]
                for type_obstacle in dict["obst_type"]
                for ic_speed_x in dict["ic_speed_x"]
                for ic_roll in dict["ic_roll"]
                for ic_pitch in dict["ic_pitch"]
                for ic_yaw in dict["ic_yaw"]
            ],
        )
    db_ff.rename(f"result/mt_{str_date}")


if __name__ == "__main__":
    # Script for running the validation of the co-design methodology (see section VI.B of the paper).
    # If you leave the code unchanged, it will run with the parameters from the paper.
    # If you want to run your own validation, change the dictionary `dict` below.
    validation_dict = {}
    # Drone names to be tested
    validation_dict["robot"] = [
        "fixed_wing_drone_back",  # bix3
        "drone_nsga_46295d0_1",  # opt1
        "drone_nsga_46295d0_2",  # opt2
        "drone_nsga_46295d0_3",  # opt3
        "drone_nsga_46295d0_4",  # opt4
    ]
    # List of distances to be tested (see Fig. 8 of the paper)
    validation_dict["goal_dist"] = [30, 40, 50]
    # List of angles to be tested (see Fig. 8 of the paper)
    validation_dict["goal_angl"] = [0, 10, 20, 30, 40, 50]
    # List of radii to be tested (see Fig. 8 of the paper)
    validation_dict["obst_radius"] = [0, 0.5, 2, 4, 6, 8]
    # List of obstacle types to be tested (see Fig. 8 of the paper)
    validation_dict["obst_type"] = ["sphere"]
    # List of initial speed along X to be tested (see Fig. 8 of the paper)
    validation_dict["ic_speed_x"] = [8, 10, 12]
    # List of initial drone orientation to be tested (see Fig. 8 of the paper)
    validation_dict["ic_roll"] = [0]
    validation_dict["ic_pitch"] = [-5, 0, 5]
    validation_dict["ic_yaw"] = [0]

    run_validation(validation_dict)
