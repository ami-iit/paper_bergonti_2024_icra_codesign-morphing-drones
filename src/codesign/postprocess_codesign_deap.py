import pickle
from traj.trajectory import Postprocess
from core.visualizer import Visualize_robot
from core.robot import Robot
import numpy as np
import glob
import os
import pandas as pd
from codesign.urdf_chromosome_uav import Chromosome_Drone, create_urdf_model
from codesign.codesign_deap import Stats_Codesign
from typing import Dict, Tuple


def get_trajectory_output(
    name_deap: str = None, chromosome: list = None
) -> Tuple[Dict, str, str, Stats_Codesign, pd.DataFrame]:
    if name_deap is None:
        name_deap = max(glob.glob("result/deap_*.pkl"), key=os.path.getctime)[:-4]
        print(name_deap)
    else:
        name_deap = name_deap[:-4]
    stats_deap = Stats_Codesign.load(name_deap)
    df = pd.read_csv(f"{name_deap}.csv")
    if chromosome is None:
        print(
            f"`` | `{name_deap}` | {stats_deap.git_info['commit']} | {stats_deap.stgs['n_pop']} |  {stats_deap.stgs['crossover_prob']:.2f} | {stats_deap.stgs['mutation_prob']:.2f} | {stats_deap.stgs['n_gen']} | -- |"
        )
        print(f"`` | `{name_deap}` | {(df['timestamp'].max()-df['timestamp'].min())/3600:.2f}h | {len(df)}")

        for chromosome in [
            stats_deap.populations_front[-1][0],
            stats_deap.populations_front[-1][int(len(stats_deap.populations_front[-1]) / 2)],
            stats_deap.populations_front[-1][-1],
        ]:
            energy = df[df["chromosome"] == str(chromosome)]["energy"].values[0]
            time = df[df["chromosome"] == str(chromosome)]["time"].values[0]
            chromosome_dict = Chromosome_Drone().from_list_to_dict(chromosome)
            print(
                f" `` | `{chromosome[:9]}` | `{chromosome[9:15]}` | `[{chromosome[-2]}]` | `[{chromosome_dict['controller_parameters']['weight_time_energy']}]` | {energy:.1f} | {time:.2f}  "
            )
    fullpath_model = create_urdf_model(chromosome, overwrite=False)
    index_task = 0
    traj_name = f'{eval(df[df["chromosome"]==str(chromosome)]["traj_name"].values[0])[index_task]}.p'
    with open(traj_name, "rb") as f:
        s = pickle.load(f)
    print(traj_name)
    print(chromosome_dict)
    return s["out"], traj_name, fullpath_model, stats_deap, df


def clean(fullpath_model):
    os.remove(f"{fullpath_model}.urdf")
    os.remove(f"{fullpath_model}.toml")


if __name__ == "__main__":
    out, traj_name, fullpath_model, stats_deap, df = get_trajectory_output(name_deap=None, chromosome=None)

    pp = Postprocess(out)
    pp.save_pics(name_directory_images=traj_name[:-2], figsize=(16, 9))
    pp.print_stats()

    robot = Robot(fullpath_model=out["robot"]["fullpath_model"])
    robot.set_joint_limit(
        ub_joint_pos=out["robot"]["limits"]["joint_pos"]["ub"],
        lb_joint_pos=out["robot"]["limits"]["joint_pos"]["lb"],
        ub_joint_vel=out["robot"]["limits"]["joint_vel"]["ub"],
        ub_joint_acc=out["robot"]["limits"]["joint_acc"]["ub"],
        ub_joint_tor=out["robot"]["limits"]["joint_tor"]["ub"],
        ub_joint_dot_tor=out["robot"]["limits"]["joint_dot_tor"]["ub"],
    )
    robot.set_propeller_limit(
        ub_thrust=out["robot"]["limits"]["prop_thrust"]["ub"],
        ub_dot_thrust=out["robot"]["limits"]["prop_dot_thrust"]["ub"],
    )

    viz = Visualize_robot(robot)
    viz.set_goals(out["constant"]["goals"])
    viz.set_obstacles(out["constant"]["obstacles"])
    viz.run(
        t=out["time"],
        base_state=np.concatenate((out["optivar"]["pos_w_b"], out["optivar"]["quat_w_b"])),
        joint_values=out["optivar"]["s"],
        prop_values=out["optivar"]["thrust"],
        aero_values=out["optivar"]["vcat_wrenchAero_w"],
        base_twist_values=out["optivar"]["twist_w_b"],
    )

    stats_deap.plot_fitness(save_dir="fitness", fill="sem")
    stats_deap.plot_pareto(save_dir="pareto_gen", plot_all_individuals=False)
    stats_deap.plot_pareto(save_dir="pareto_tot")
    stats_deap.plot_pareto_energy_time(df, save_dir="pareto_ET_gen")
    stats_deap.plot_pareto_video(save_dir="pareto_video")
    stats_deap.plot_count_unique_chromosomes(save_dir="count_unique_chromosomes")
    stats_deap.plot_genes_scaled(save_dir="genes")
    stats_deap.plot_diversity(save_dir="diversity")
    stats_deap.plot_chromosome_repetition(
        save_dir="chromosome_repetition", figsize=[0.106 * stats_deap.stgs["n_gen"], 0.048 * stats_deap.stgs["n_pop"]]
    )

    pp.base()
    pp.base_3d()
    pp.thrust()
    pp.joint()
    pp.centroidal()
    pp.com_bodies()
    pp.frame_bodies()
    pp.wind()
    pp.wind_body()
    pp.show()

    viz.close_rviz()

    clean(fullpath_model)
