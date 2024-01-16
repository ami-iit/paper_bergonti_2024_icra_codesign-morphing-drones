from codesign.codesign_deap import Stats_Codesign
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from deap import creator, base, tools
from codesign.urdf_chromosome_uav import Chromosome_Drone, create_urdf_model
import os
import utils_muav
from codesign.codesign_deap import Codesign_DEAP
from core.robot import Robot
from traj.trajectory import Gains_Trajectory, Postprocess, Obstacle_Plane
from traj.trajectory_wholeboby import Trajectory_WholeBody_Planner
import pandas as pd
from core.robot import Robot
import seaborn as sns
from matplotlib.animation import FuncAnimation
import multiprocessing


def compute_total_pareto_front(list_result_nsga) -> tools.support.ParetoFront:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Chromosome", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("get_chromosome", tools.initIterate, creator.Chromosome, Chromosome_Drone().get_random)
    toolbox.register("population", tools.initRepeat, list, toolbox.get_chromosome)
    pareto = tools.ParetoFront()
    fitness_fronts = {0: [], 1: []}
    populations_fronts = []
    for name_result_deap in list_result_nsga:
        stats_deap = Stats_Codesign.load(name_result_deap["pkl"])
        fitness_fronts[0] = fitness_fronts[0] + list(stats_deap.fitness_front[0][-1])
        fitness_fronts[1] = fitness_fronts[1] + list(stats_deap.fitness_front[1][-1])
        populations_fronts = populations_fronts + stats_deap.populations_front[-1]
    pop = toolbox.population(n=len(populations_fronts))
    for i, chromosome in enumerate(populations_fronts):
        pop[i] = creator.Chromosome(chromosome)
        pop[i].fitness.values = (fitness_fronts[0][i], fitness_fronts[1][i])
    pareto.update(pop)
    return pareto


def select_nsga_drones(pareto, tag="") -> np.ndarray:
    fitness_first_front = np.array([chromo.fitness.values for chromo in pareto])
    N = len(fitness_first_front)
    sel = [4, int(N / 3) + 3, 2 * int(N / 3), N - 4]
    fitness_sel = fitness_first_front[sel, :]
    for i in range(len(sel)):
        fullpath_model = create_urdf_model(pareto[sel[i]], overwrite=False)
        repo_tree = utils_muav.get_repository_tree(relative_path=True)
        os.rename(f"{fullpath_model}.urdf", f"{repo_tree['urdf']}/drone_nsga_{tag}_{i+1}.urdf")
        os.rename(f"{fullpath_model}.toml", f"{repo_tree['urdf']}/drone_nsga_{tag}_{i+1}.toml")
    return fitness_sel


def select_ga_drones(list_result_ga) -> np.ndarray:
    fitness_sel = np.zeros((len(list_result_ga), 2))
    for i, result_ga in enumerate(list_result_ga):
        import pandas as pd

        df = pd.read_csv(result_ga["pkl"] + ".csv")
        # find row with best fitness
        idx = df["fitness_function"].idxmax()
        fitness_sel[i, :] = (df.iloc[idx][["energy", "time"]].values) / 5
        chromo = eval(df.iloc[idx]["chromosome"])
        chromo[-1] = 2  # because ga tested are executed with a fixed controller gain
        fullpath_model = create_urdf_model(chromo, overwrite=False)
        repo_tree = utils_muav.get_repository_tree(relative_path=True)
        os.rename(f"{fullpath_model}.urdf", f"{repo_tree['urdf']}/drone_ga_{i+1}.urdf")
        os.rename(f"{fullpath_model}.toml", f"{repo_tree['urdf']}/drone_ga_{i+1}.toml")
    return fitness_sel


def plot_pareto_front_evolution(list_result_nsga, list_gen):
    for i, result_deap in enumerate(list_result_nsga):
        stats_deap = Stats_Codesign.load(result_deap["pkl"])
        pal = sns.color_palette("GnBu_d", len(list_gen))
        plt.figure(figsize=(7, 3))
        for j, gen in enumerate(list_gen):
            plt.plot(
                stats_deap.fitness_front[0][gen],
                stats_deap.fitness_front[1][gen],
                c=pal[j],
                marker="",
                label=f"after {gen} generations",
            )
        plt.legend()
        plt.xlabel("Energy [J]")
        plt.ylabel("Time [s]")
        plt.grid(color="0.9")
        plt.gca().set_axisbelow(True)
        plt.xlim([52.2, 236])
        plt.ylim([4.85, 7.1])
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"pareto_front_ev{i}.png")
        plt.savefig(f"pareto_front_ev{i}.pdf")


def plot_pareto_front_evolution_video(list_result_nsga):
    pal = list(mcolors.TABLEAU_COLORS) + sns.color_palette("Blues", 10)

    fig, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=5000)
    ax1.set_xlim((56.76203784720779, 316.3087760589073))
    ax1.set_ylim((4.9087881235901545, 7.187344461696856))
    stats_deap = Stats_Codesign.load(list_result_nsga[0]["pkl"])
    gen = len(stats_deap.fitness_front[0]) - 1
    ax1.plot(stats_deap.fitness_front[0][gen], stats_deap.fitness_front[1][gen], c=pal[-0 - 1])
    ax1.set_xlabel("energy consumption")
    ax1.set_ylabel("agility")
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.tight_layout()
    plt.savefig("pareto_front_small.png")

    fps = 4
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.3), dpi=500)
    pal = list(mcolors.TABLEAU_COLORS) + sns.color_palette("Blues", 10)
    ax1.grid(color="0.9")
    ax1.set_xlim((56.76203784720779, 316.3087760589073))
    ax1.set_ylim((4.9087881235901545, 7.187344461696856))
    ax1.set_xlabel("energy consumption [J]")
    ax1.set_ylabel("mission completion time [s]")

    def update(frame):
        result_deap = list_result_nsga[frame]
        stats_deap = Stats_Codesign.load(result_deap["pkl"])
        gen = len(stats_deap.fitness_front[0]) - 1
        ax1.plot(stats_deap.fitness_front[0][gen], stats_deap.fitness_front[1][gen], c=pal[-frame - 1])

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(list_result_nsga), interval=1000 / fps)
    video_writer = ani.save("pareto_front_evolution.mp4", fps=fps, extra_args=["-vcodec", "libx264"], dpi=600)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.3), dpi=500)
    for i, result_deap in enumerate(list_result_nsga):
        stats_deap = Stats_Codesign.load(result_deap["pkl"])
        gen = len(stats_deap.fitness_front[0]) - 1
        pal = list(mcolors.TABLEAU_COLORS) + sns.color_palette("Blues", 10)
        ax1.plot(stats_deap.fitness_front[0][gen], stats_deap.fitness_front[1][gen], c=pal[-i - 1])
    ax1.grid(color="0.9")
    ax1.set_xlim((56.76203784720779, 316.3087760589073))
    ax1.set_ylim((4.9087881235901545, 7.187344461696856))
    ax1.set_xlabel("energy consumption [J]")
    ax1.set_ylabel("mission completion time [s]")
    plt.savefig("pareto_front_video.png")


def plot_pareto_front(list_result_nsga, fitness_sel_nsga, fitness_sel_ga, fitness_bix3):
    arrowprops = dict(arrowstyle="-|>", fc="w", connectionstyle="arc3")
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 2.105), gridspec_kw={"width_ratios": [1, 0.1]})
    if len(fitness_bix3) > 0:
        ax1.plot(
            fitness_bix3[0],
            fitness_bix3[1],
            c="k",
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=colors_drones[0],
            # markeredgecolor="brown",
        )
        ax2.plot(
            fitness_bix3[0],
            fitness_bix3[1],
            c="k",
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=colors_drones[0],
            # markeredgecolor="brown",
        )
        ax2.annotate(
            f"bix3", xy=fitness_bix3, xytext=(fitness_bix3[0] - 10, fitness_bix3[1] - 0.5), arrowprops=arrowprops
        )
    for i, result_deap in enumerate(list_result_nsga):
        stats_deap = Stats_Codesign.load(result_deap["pkl"])
        gen = len(stats_deap.fitness_front[0]) - 1
        # plt.scatter(stats_deap.fitness[0][:gen, :], stats_deap.fitness[1][:gen, :], c="g", marker=".")
        # plt.scatter(stats_deap.fitness_offspring[0][:gen, :], stats_deap.fitness_offspring[1][:gen, :], c="g", marker=".")
        pal = list(mcolors.TABLEAU_COLORS) + sns.color_palette("Blues", 10)
        ax1.plot(
            stats_deap.fitness_front[0][gen],
            stats_deap.fitness_front[1][gen],
            c=pal[-i - 1],
            # marker=".",
            # label=result_deap["name"],
        )
    for i, sel_nsga in enumerate(fitness_sel_nsga):
        ax1.plot(
            sel_nsga[0],
            sel_nsga[1],
            c="k",
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=colors_drones[i + 1],
            # markeredgecolor="brown",
        )
    ax1.annotate(
        f"opt1",
        xy=fitness_sel_nsga[0, :],
        xytext=(fitness_sel_nsga[0, 0], fitness_sel_nsga[0, 1] - 0.5),
        arrowprops=arrowprops,
    )
    ax1.annotate(
        f"opt2",
        xy=fitness_sel_nsga[1, :],
        xytext=(fitness_sel_nsga[1, 0] - 5, fitness_sel_nsga[1, 1] + 0.5),
        arrowprops=arrowprops,
    )
    ax1.annotate(
        f"opt3",
        xy=fitness_sel_nsga[2, :],
        xytext=(fitness_sel_nsga[2, 0], fitness_sel_nsga[2, 1] + 0.5),
        arrowprops=arrowprops,
    )
    ax1.annotate(
        f"opt4",
        xy=fitness_sel_nsga[3, :],
        xytext=(fitness_sel_nsga[3, 0], fitness_sel_nsga[3, 1] + 0.5),
        arrowprops=arrowprops,
    )
    if len(fitness_sel_ga) > 0:
        plt.plot(
            fitness_sel_ga[:, 0],
            fitness_sel_ga[:, 1],
            c="k",
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor="tab:green",
            # markeredgecolor="brown",
            label="SOF best individuals",
        )
        plt.annotate(
            f"sof1",
            xy=fitness_sel_ga[0, :],
            xytext=(fitness_sel_ga[0, 0] + 20, fitness_sel_ga[0, 1]),
            arrowprops=arrowprops,
        )
        plt.annotate(
            f"sof2",
            xy=fitness_sel_ga[1, :],
            xytext=(fitness_sel_ga[1, 0] + 20, fitness_sel_ga[1, 1]),
            arrowprops=arrowprops,
        )
    # plt.title(f"Pareto Front")
    # set xlabel centered between subplots
    ax1.set_xlim((65, 170))
    ax2.set_xlim((280, 320))
    ax2.set_xticks((280, 320))
    ax1.grid(color="0.9")
    ax2.grid(color="0.9")
    plt.tight_layout()

    ax1.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.tick_params(left=False)
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

    # reduce subplot spacing (https://stackoverflow.com/a/65411281/5288758)
    # add xlabel centered between subplots
    fig.text(0.5, 0.02, "energy [J]", ha="center")
    ax1.set_ylabel("time [s]")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03)
    # plt.legend()
    plt.savefig("pareto_front.png")
    plt.savefig("pareto_front.pdf")


def plot_genes_variability(list_result_nsga):
    # plot genes
    df = pd.read_csv(f'{utils_muav.get_repository_tree()["database_servomotor"]}/db_servomotor.csv')
    df["torque_limit"]
    df["speed_limit"]
    pop_paretos = {0: [], 1: [], 2: [], 3: []}
    robot_param = {"mass": {0: [], 1: [], 2: [], 3: []}, "com": {0: [], 1: [], 2: [], 3: []}}
    servo_param = {"torque": {0: [], 1: [], 2: [], 3: []}, "speed": {0: [], 1: [], 2: [], 3: []}}
    for name_result_deap in list_result_nsga:
        stats_deap = Stats_Codesign.load(name_result_deap["pkl"])
        for chromo, fit0, fit1 in zip(
            stats_deap.populations_front[-1], stats_deap.fitness_front[0][-1], stats_deap.fitness_front[1][-1]
        ):
            robot = Robot(create_urdf_model(chromo, overwrite=True))
            mass = robot.kinDyn.get_total_mass()
            com = robot.kinDyn.CoM_position_fun()(np.eye(4), np.zeros(robot.ndofs)).full().flatten()
            motor_torque = [np.nan, np.nan, np.nan, np.nan]
            motor_speed = [np.nan, np.nan, np.nan, np.nan]
            for i in range(3):
                motor_speed[chromo[9 + 2 * i]] = df[df["id"] == chromo[10 + 2 * i]]["speed_limit"].values[0]
                motor_torque[chromo[9 + 2 * i]] = df[df["id"] == chromo[10 + 2 * i]]["torque_limit"].values[0]
            if fit0 < 85:
                pop_paretos[1].append(chromo)
                robot_param["mass"][1].append(mass)
                robot_param["com"][1].append(com)
                servo_param["torque"][1].append(motor_torque)
                servo_param["speed"][1].append(motor_speed)
            elif fit0 < 105:
                pop_paretos[2].append(chromo)
                robot_param["mass"][2].append(mass)
                robot_param["com"][2].append(com)
                servo_param["torque"][2].append(motor_torque)
                servo_param["speed"][2].append(motor_speed)
            else:
                pop_paretos[3].append(chromo)
                robot_param["mass"][3].append(mass)
                robot_param["com"][3].append(com)
                servo_param["torque"][3].append(motor_torque)
                servo_param["speed"][3].append(motor_speed)
            pop_paretos[0].append(chromo)
            robot_param["mass"][0].append(mass)
            robot_param["com"][0].append(com)
            servo_param["torque"][0].append(motor_torque)
            servo_param["speed"][0].append(motor_speed)
    ngr = len(pop_paretos)
    for i in range(ngr):
        pop_paretos[i] = np.array(pop_paretos[i])
        robot_param["mass"][i] = np.array(robot_param["mass"][i])
        robot_param["com"][i] = np.array(robot_param["com"][i])
        servo_param["torque"][i] = np.array(servo_param["torque"][i])
        servo_param["speed"][i] = np.array(servo_param["speed"][i])

    from codesign.urdf_chromosome_uav import Gene_Weight_Time_Energy

    df = pd.read_csv(f'{utils_muav.get_repository_tree()["database_propeller"]}/db_propeller.csv')
    enum_param = {"prop": {0: [], 1: [], 2: [], 3: []}, "contr": {0: [], 1: [], 2: [], 3: []}}
    for i in range(ngr):
        enum_param["prop"][i] = [df[df["id"] == j]["max_thrust"].values[0] for j in list(pop_paretos[i][:, 19])]
        enum_param["contr"][i] = [
            Gene_Weight_Time_Energy().get_list_possible_values()[int(j)] for j in list(pop_paretos[i][:, 20])
        ]
    print(
        "prop thrust"
        + f" \t | all: {np.mean(enum_param['prop'][0]):.3f}±{np.std(enum_param['prop'][0]):.3f}"
        + f' \t | slow: {np.mean(enum_param["prop"][1]):.3f}±{np.std(enum_param["prop"][1]):.3f}'
        + f' \t | int: {np.mean(enum_param["prop"][2]):.3f}±{np.std(enum_param["prop"][2]):.3f}'
        + f' \t | fast: {np.mean(enum_param["prop"][3]):.3f}±{np.std(enum_param["prop"][3]):.3f}'
    )
    print(
        "controller weight"
        + f" \t | all: {np.mean(enum_param['contr'][0]):.3f}±{np.std(enum_param['contr'][0]):.3f}"
        + f' \t | slow: {np.mean(enum_param["contr"][1]):.3f}±{np.std(enum_param["contr"][1]):.3f}'
        + f' \t | int: {np.mean(enum_param["contr"][2]):.3f}±{np.std(enum_param["contr"][2]):.3f}'
        + f' \t | fast: {np.mean(enum_param["contr"][3]):.3f}±{np.std(enum_param["contr"][3]):.3f}'
    )

    list_plt_details = [
        {"idx": 0, "ylabel": "position [m]", "title": "wing horizontal location", "figsize": (6 * 16 / 9, 6)},
        {"idx": 2, "ylabel": "position [m]", "title": "wing vertical location", "figsize": (6 * 16 / 9, 6)},
        {"idx": 3, "ylabel": "angle [deg]", "title": "wing dihedral angle (roll)", "figsize": (6 * 16 / 9, 6)},
        {"idx": 4, "ylabel": "angle [deg]", "title": "wing incidence angle (pitch)", "figsize": (6 * 16 / 9, 6)},
        {"idx": 5, "ylabel": "angle [deg]", "title": "wing sweep angle (yaw)", "figsize": (6 * 16 / 9, 6)},
        {"idx": 7, "ylabel": "length [m]", "title": "wing chord", "figsize": (6 * 16 / 9, 6)},
        {"idx": 8, "ylabel": "length [m]", "title": "wing aspect ratio", "figsize": (6 * 16 / 9, 6)},
        # {"idx": 9, "ylabel": "[]", "title": "joint1 - type", "figsize": (6 * 16 / 9, 6)},
        # {"idx": 10, "ylabel": "[]", "title": "joint1 - servo", "figsize": (6 * 16 / 9, 6)},
        # {"idx": 11, "ylabel": "[]", "title": "joint2 - type", "figsize": (6 * 16 / 9, 6)},
        # {"idx": 12, "ylabel": "[]", "title": "joint2 - servo", "figsize": (6 * 16 / 9, 6)},
        # {"idx": 13, "ylabel": "[]", "title": "joint3 - type", "figsize": (6 * 16 / 9, 6)},
        # {"idx": 14, "ylabel": "[]", "title": "joint3 - servo", "figsize": (6 * 16 / 9, 6)},
        {"idx": 19, "ylabel": "[]", "title": "propeller model", "figsize": (6 * 16 / 9, 6)},
        {"idx": 20, "ylabel": "[]", "title": "controller gain", "figsize": (6 * 16 / 9, 6)},
    ]
    for plt_details in list_plt_details:
        print(
            f'{plt_details["title"]} '
            + f" \t | range: [{Chromosome_Drone().min()[plt_details['idx']]:.3f},{Chromosome_Drone().max()[plt_details['idx']]:.3f}]"
            + f' \t | all: {np.mean(pop_paretos[0][:, plt_details["idx"]]):.3f}±{np.std(pop_paretos[0][:, plt_details["idx"]]):.3f}'
            + f' \t | slow: {np.mean(pop_paretos[1][:, plt_details["idx"]]):.3f}±{np.std(pop_paretos[1][:, plt_details["idx"]]):.3f}'
            + f' \t | int: {np.mean(pop_paretos[2][:, plt_details["idx"]]):.3f}±{np.std(pop_paretos[2][:, plt_details["idx"]]):.3f}'
            + f' \t | fast: {np.mean(pop_paretos[3][:, plt_details["idx"]]):.3f}±{np.std(pop_paretos[3][:, plt_details["idx"]]):.3f}'
        )
        plt.figure(figsize=plt_details["figsize"])
        plt.boxplot([pop_paretos[i][:, plt_details["idx"]] for i in range(ngr)], patch_artist=True)
        plt.ylim([Chromosome_Drone().min()[plt_details["idx"]], Chromosome_Drone().max()[plt_details["idx"]]])
        plt.ylabel(plt_details["ylabel"])
        plt.title(plt_details["title"])
        plt.savefig(f"gene_{plt_details['idx']}.png")
    # mass
    plt.figure(figsize=(6 * 16 / 9, 6))
    plt.boxplot([robot_param["mass"][i] for i in range(ngr)], patch_artist=True)
    plt.ylabel("mass [kg]")
    plt.title("mass")
    plt.savefig("gene_mass.png")
    print(
        "mass"
        + f" \t | all: {np.mean(robot_param['mass'][0]):.3f}±{np.std(robot_param['mass'][0]):.3f}"
        + f' \t | slow: {np.mean(robot_param["mass"][1]):.3f}±{np.std(robot_param["mass"][1]):.3f}'
        + f' \t | int: {np.mean(robot_param["mass"][2]):.3f}±{np.std(robot_param["mass"][2]):.3f}'
        + f' \t | fast: {np.mean(robot_param["mass"][3]):.3f}±{np.std(robot_param["mass"][3]):.3f}'
    )
    # com
    for i in range(3):
        plt.figure(figsize=(6 * 16 / 9, 6))
        plt.boxplot([robot_param["com"][j][:, i] for j in range(ngr)], patch_artist=True)
        plt.ylabel("position [m]")
        plt.title(f"com - {['x','y','z'][i]}")
        plt.savefig(f"gene_com_{i}.png")
        print(
            f"com - {['x','y','z'][i]} "
            + f" \t | all: {np.mean(robot_param['com'][0][:, i]):.3f}±{np.std(robot_param['com'][0][:, i]):.3f}"
            + f' \t | slow: {np.mean(robot_param["com"][1][:, i]):.3f}±{np.std(robot_param["com"][1][:, i]):.3f}'
            + f' \t | int: {np.mean(robot_param["com"][2][:, i]):.3f}±{np.std(robot_param["com"][2][:, i]):.3f}'
            + f' \t | fast: {np.mean(robot_param["com"][3][:, i]):.3f}±{np.std(robot_param["com"][3][:, i]):.3f}'
        )
    # motor torque
    for i, name in zip(range(1, 4), ["dihedral", "sweep", "twist"]):
        plt.figure(figsize=(6 * 16 / 9, 6))
        plt.boxplot([servo_param["torque"][j][:, i] for j in range(ngr)], patch_artist=True)
        plt.ylabel("torque [Nm]")
        plt.title(f"motor - {name}")
        plt.savefig(f"gene_motor_torque_{name}.png")
        print(
            f"motor torque - {name} "
            + f" \t | all: {np.nanmean(servo_param['torque'][0][:, i]):.3f}±{np.nanstd(servo_param['torque'][0][:, i]):.3f}"
            + f' \t | slow: {np.nanmean(servo_param["torque"][1][:, i]):.3f}±{np.nanstd(servo_param["torque"][1][:, i]):.3f}'
            + f' \t | int: {np.nanmean(servo_param["torque"][2][:, i]):.3f}±{np.nanstd(servo_param["torque"][2][:, i]):.3f}'
            + f' \t | fast: {np.nanmean(servo_param["torque"][3][:, i]):.3f}±{np.nanstd(servo_param["torque"][3][:, i]):.3f}'
        )
    # motor speed
    for i, name in zip(range(1, 4), ["dihedral", "sweep", "twist"]):
        plt.figure(figsize=(6 * 16 / 9, 6))
        plt.boxplot([servo_param["speed"][j][:, i] for j in range(ngr)], patch_artist=True)
        plt.ylabel("speed [rad/s]")
        plt.title(f"motor speed - {name}")
        plt.savefig(f"gene_motor_speed_{name}.png")
        print(
            f"motor speed - {name} "
            + f" \t | all: {np.nanmean(servo_param['speed'][0][:, i]):.3f}±{np.nanstd(servo_param['speed'][0][:, i]):.3f}"
            + f' \t | slow: {np.nanmean(servo_param["speed"][1][:, i]):.3f}±{np.nanstd(servo_param["speed"][1][:, i]):.3f}'
            + f' \t | int: {np.nanmean(servo_param["speed"][2][:, i]):.3f}±{np.nanstd(servo_param["speed"][2][:, i]):.3f}'
            + f' \t | fast: {np.nanmean(servo_param["speed"][3][:, i]):.3f}±{np.nanstd(servo_param["speed"][3][:, i]):.3f}'
        )

    plt.figure(figsize=(6 * 16 / 9, 6))
    for i in range(ngr):
        plt.subplot(1, ngr, i + 1)
        unq, cnt = np.unique(pop_paretos[i][:, [9, 11, 13]], return_counts=True, axis=0)
        plt.pie(cnt, labels=unq, autopct="%1.1f%%")
    plt.savefig("gene_joint.png")

    pop_paretos = []
    for name_result_deap in list_result_nsga:
        stats_deap = Stats_Codesign.load(name_result_deap["pkl"])
        pop_paretos = pop_paretos + stats_deap.populations_front[-1]
    pop_paretos = np.array(pop_paretos)

    max_value = np.array(Chromosome_Drone().max())
    min_value = np.array(Chromosome_Drone().min())
    rang = max_value - min_value
    rang[rang == 0] = 1
    chromosomes_scaled = (pop_paretos - min_value) / rang
    plt.figure(figsize=(6 * 16 / 9, 6))
    plt.boxplot(chromosomes_scaled, patch_artist=True)
    plt.title("unique chromosomes")
    plt.xlabel("Generation")
    plt.ylabel("unique chromosomes")
    plt.savefig("gene_paretos.png")


def plot_fitness_ga(list_result_ga):
    if len(list_result_ga) > 0:
        max_n_gen = 101
        best = np.zeros((len(list_result_ga), max_n_gen))
        for i, result_ga in enumerate(list_result_ga):
            best[i, :] = -np.array(pygad.load(filename=result_ga["pkl"]).best_solutions_fitness)
        plt.figure(figsize=(4, 2.5))
        plt.plot(range(max_n_gen), best.mean(axis=0), color="tab:green")
        plt.fill_between(range(max_n_gen), best.min(axis=0), best.max(axis=0), alpha=0.2, color="tab:green")
        # plt.title("Fitness function")
        plt.ylim([np.min(best), np.max(best[best < 1e6])])
        plt.xlim([0, max_n_gen])
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid(color="0.9")
        plt.gca().set_axisbelow(True)
        plt.tight_layout()
        plt.savefig("fitness_ga.png")
        plt.savefig("fitness_ga.pdf", transparent=True)


def plot_diversity_ga(list_result_ga):
    if len(list_result_ga) > 0:
        max_n_gen = 100
        div = np.zeros((len(list_result_ga), max_n_gen))
        print("plot_diversity started")
        for i, result_ga in enumerate(list_result_ga):
            div[i, :] = pygad.load(filename=result_ga["pkl"]).diversity
        plt.figure(figsize=(4, 2.5))
        plt.plot(range(max_n_gen), div.mean(axis=0), label="Mean", color="tab:green")
        plt.fill_between(range(max_n_gen), div.min(axis=0), div.max(axis=0), alpha=0.2, color="tab:green")
        plt.xlim([0, max_n_gen])
        # plt.title("Diversity")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.grid(color="0.9")
        plt.gca().set_axisbelow(True)
        plt.tight_layout()
        plt.savefig("diversity_ga.png")
        plt.savefig("diversity_ga.pdf", transparent=True)


def plot_traj(name_urdf, task):
    out = solve_task(name_urdf, task)
    # joint_pos
    plt.figure(figsize=(3, 2))
    plt.plot(out["time"], out["optivar"]["s"].T * 180 / np.pi)
    plt.xlim([0, out["time"].max()])
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.legend(["r.sweep", "r.incid.", "l.sweep", "l.incid."], loc="upper right")
    plt.grid(color="0.9")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("traj_jointpos.png", bbox_inches="tight")
    plt.savefig("traj_jointpos.pdf", transparent=True, bbox_inches="tight")
    # joint_tor
    plt.figure(figsize=(7, 3))
    plt.plot(out["time"], out["optivar"]["torque"].T)
    plt.xlim([0, out["time"].max()])
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.legend(["r.sweep", "r.incidence", "l.sweep", "l.incidence"], loc="upper right")
    plt.grid(color="0.9")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("traj_jointtor.png", bbox_inches="tight")
    plt.savefig("traj_jointtor.pdf", transparent=True, bbox_inches="tight")
    # thrust
    plt.figure(figsize=(3, 2))
    plt.plot(out["time"], out["optivar"]["thrust"].T)
    plt.xlim([0, out["time"].max()])
    plt.xlabel("time [s]")
    plt.ylabel("force [N]")
    plt.grid(color="0.9")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("traj_thrust.png", bbox_inches="tight")
    plt.savefig("traj_thrust.pdf", transparent=True, bbox_inches="tight")

    # plots
    def get_time(X, r):
        T = []
        for x in X:
            t = out["time"][np.abs(out["optivar"]["pos_w_b"][0, :] - x).argmin()]
            T.append(t.round(r))
        return T

    N = len(out["optivar"]["pos_w_b"][0, :])
    # sel = (0, int(1 * N / 5), int(2 * N / 5) - 1, int(3 * N / 5) - 8, int(4 * N / 5) - 13, N - 1)
    # attitude roll pitch yaw
    fig = plt.figure(figsize=(7, 2.5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlim([-2, 62])
    ax1.plot(out["optivar"]["pos_w_b"][0, :], out["pp"]["rpy_w_b"].T * 180 / np.pi)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("angle [deg]")
    ax1.legend(["roll", "pitch", "yaw"], loc="upper right")
    ax1.grid(color="0.9")
    ax2 = ax1.twiny()
    ax1Ticks = ax1.get_xticks()
    ax2Ticks = ax1Ticks
    ax2.set_xticks(ax2Ticks)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(get_time(ax2Ticks, 1))
    ax2.set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig("traj_attitude.png")
    plt.savefig("traj_attitude.pdf", transparent=True)
    # position
    fig = plt.figure(figsize=(4.5, 1.8))
    ax1 = fig.add_subplot(111)
    ax1.plot(out["optivar"]["pos_w_b"][0, :], out["optivar"]["pos_w_b"][1, :], color="black")
    ax1.plot(0, 0, marker="o", color="tab:blue")
    for goal in out["constant"]["goals"]:
        goal.plot_xy()
    for obstacle in out["constant"]["obstacles"][:-1]:
        obstacle.plot_xy()
    # ax1.scatter(out["optivar"]["pos_w_b"][0, sel], out["optivar"]["pos_w_b"][1, sel], color="black", marker="x")
    # for txt, x, y in zip(
    #     ["a", "b", "c", "d", "e", "f"], out["optivar"]["pos_w_b"][0, sel], out["optivar"]["pos_w_b"][1, sel]
    # ):
    #     ax1.annotate(txt, (x, y + 1))
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_ylim([-5.5, 5.5])
    ax1.grid(color="0.9")
    ax1.set_axisbelow(True)
    ax1.set_xlim([-1, 61])
    # ax2 = ax1.twiny()
    # ax1Ticks = ax1.get_xticks()
    # ax2Ticks = ax1Ticks
    # ax2.set_xticks(ax2Ticks)
    # ax2.set_xbound(ax1.get_xbound())
    # ax2.set_xticklabels(get_time(ax2Ticks, 1))
    # ax2.set_xlabel("time [s]")
    # plt.tight_layout()
    ax1.set_aspect("equal")
    # ax2.set_xbound(ax1.get_xbound())
    # ax2.set_xticklabels(get_time(ax2Ticks, 1))
    # ax2.set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig("traj_xy.png", bbox_inches="tight")
    plt.savefig("traj_xy.pdf", transparent=True, bbox_inches="tight")


def plot_trajs(list_name_urdf, task):
    with multiprocessing.Pool(processes=len(list_name_urdf)) as pool:
        out = pool.starmap(solve_task, [(name_urdf, task) for name_urdf in list_name_urdf])

    # increase tolerance of last goal for visualisation purpose
    out[0]["constant"]["goals"][0].position["tol"] = out[0]["constant"]["goals"][0].position["tol"] * 5

    fig = plt.figure(figsize=(10, 2.1))
    ax1 = fig.add_subplot(111)
    for i, name_urdf in enumerate(list_name_urdf):
        ax1.plot(
            out[i]["optivar"]["pos_w_b"][0, :],
            out[i]["optivar"]["pos_w_b"][1, :],
            label=name_urdf,
            linewidth=0.8,
            color=colors_drones[i],
        )
    ax1.plot(0, 0, marker="o", color="tab:blue")
    for goal in [out[0]["constant"]["goals"][0]]:
        goal.plot_xy()
    for obstacle in out[0]["constant"]["obstacles"]:
        if type(obstacle) != Obstacle_Plane:
            obstacle.plot_xy()
    plt.legend(["bix3", "opt1", "opt2", "opt3", "opt4"], loc="upper right", bbox_to_anchor=(0.8, 1))
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_ylim([-4.9, 4.7])
    ax1.grid(color="0.9")
    ax1.set_axisbelow(True)
    ax1.set_xlim([-1, 61])
    ax1.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("trajs_xy.png", bbox_inches="tight")
    plt.savefig("trajs_xy.pdf", transparent=True, bbox_inches="tight")

    fig = plt.figure(figsize=(10, 2.1))
    ax1 = fig.add_subplot(111)
    for i, name_urdf in enumerate(list_name_urdf):
        ax1.plot(
            out[i]["optivar"]["pos_w_b"][0, :],
            out[i]["optivar"]["pos_w_b"][1, :],
            label=name_urdf,
            linewidth=0.8,
            color=colors_drones[i],
        )
    ax1.plot(0, 0, marker="o", color="tab:blue")

    for goal in out[0]["constant"]["goals"]:
        goal.plot_xy(ax=ax1)
    for obstacle in out[0]["constant"]["obstacles"]:
        if type(obstacle) != Obstacle_Plane:
            obstacle.plot_xy(ax=ax1)
    plt.legend(["bix3", "opt1", "opt2", "opt3", "opt4"], loc="upper right", bbox_to_anchor=(0.8, 1))
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_ylim([-5, 5])
    ax1.grid(color="0.9")
    ax1.set_axisbelow(True)
    ax1.set_xlim([-1, 61])
    ax1.set_aspect("equal")
    plt.tight_layout()

    axins = ax1.inset_axes([0.17, -0.25, 0.3, 1])
    for i, name_urdf in enumerate(list_name_urdf):
        axins.plot(
            out[i]["optivar"]["pos_w_b"][0, :],
            out[i]["optivar"]["pos_w_b"][1, :],
            label=name_urdf,
            linewidth=1,
            color=colors_drones[i],
        )
    axins.plot(0, 0, marker="o", color="tab:blue")
    for goal in out[0]["constant"]["goals"]:
        goal.plot_xy(ax=axins)
    for obstacle in out[0]["constant"]["obstacles"]:
        if type(obstacle) != Obstacle_Plane:
            obstacle.plot_xy(ax=axins)
    # ax1.set_ylim([-4.9, 4.7])
    # axins.grid(color="0.9")
    axins.set_xlim([17.5, 22])
    axins.set_ylim([3.8, 4.75])
    axins.set_aspect("equal")
    axins.set_facecolor([0.98, 0.98, 0.98])
    axins.set_xticks([])
    # axins.set_yticks([])
    axins.set_xticklabels([])
    # axins.set_yticklabels([])
    for axis in ["top", "bottom", "left", "right"]:
        axins.spines[axis].set_color("grey")
    axins.spines["bottom"].set_color("grey")
    plt.tight_layout()
    ax1.indicate_inset_zoom(axins, edgecolor="black", alpha=0.2, facecolor="grey", linewidth=0.1)
    plt.savefig("trajs_xy.png", bbox_inches="tight")
    plt.savefig("trajs_xy.pdf", transparent=True, bbox_inches="tight")

    fig = plt.figure(figsize=(6, 2))
    ax2 = fig.add_subplot(111)
    for i, name_urdf in enumerate(list_name_urdf):
        ax2.plot(
            out[i]["optivar"]["pos_w_b"][0, :],
            (out[i]["optivar"]["twist_w_b"][:3, :] ** 2).sum(0) ** 0.5,
            label=name_urdf,
            linewidth=1.5,
            color=colors_drones[i],
        )
    plt.legend(["bix3", "opt1", "opt2", "opt3", "opt4"], loc="upper right")
    ax2.set_xlabel("x [m]")
    # write ylabel with interpreter latex
    ax2.set_ylabel(r"$|| \dot{p}_B ||_2$ [m/s]")
    ax2.grid(color="0.9")
    ax2.set_axisbelow(True)
    ax2.set_xlim([0, 60])
    plt.tight_layout()
    plt.savefig("trajs_speed_x.png", bbox_inches="tight")
    plt.savefig("trajs_speed_x.pdf", transparent=True, bbox_inches="tight")


def plot_trajs_moving(list_name_urdf, task):
    with multiprocessing.Pool(processes=len(list_name_urdf)) as pool:
        out = pool.starmap(solve_task, [(name_urdf, task) for name_urdf in list_name_urdf])

    # increase tolerance of last goal for visualisation purpose
    out[0]["constant"]["goals"][0].position["tol"] = out[0]["constant"]["goals"][0].position["tol"] * 5

    fps = 60

    max_time = max([out[i]["time"].max() for i in range(len(list_name_urdf))])
    time = np.arange(0, max_time, 1 / 60)

    X = np.zeros((len(time), len(list_name_urdf)))
    Y = np.zeros((len(time), len(list_name_urdf)))
    V = np.zeros((len(time), len(list_name_urdf)))

    for i, name_urdf in enumerate(list_name_urdf):
        X[:, i] = np.interp(time, out[i]["time"], out[i]["optivar"]["pos_w_b"][0, :]).T
        Y[:, i] = np.interp(time, out[i]["time"], out[i]["optivar"]["pos_w_b"][1, :]).T
        V[:, i] = np.interp(time, out[i]["time"], (out[i]["optivar"]["twist_w_b"][:3, :] ** 2).sum(0) ** 0.5).T

    def update_XY(frame):
        x = X[:frame, :]
        y = Y[:frame, :]
        for i, l in enumerate(line):
            l.set_xdata(x[:, i])
            l.set_ydata(y[:, i])
        for i, m in enumerate(mark):
            if frame == 0:
                m.set_xdata(X[0, i])
                m.set_ydata(Y[0, i])
            else:
                m.set_xdata(x[-1, i])
                m.set_ydata(y[-1, i])
        return line + mark

    fig, ax = plt.subplots(figsize=(12, 2.8))
    line = plt.plot(X, Y, linewidth=1.5)
    for i, l in enumerate(line):
        l.set_color(colors_drones[i])
    mark = [None] * len(list_name_urdf)
    for i in range(len(list_name_urdf)):
        mark[i] = plt.plot(X[0, i], Y[0, i], marker="o", color=colors_drones[i])[0]
    ax.plot(0, 0, marker="o", color="tab:blue")
    for goal in out[0]["constant"]["goals"]:
        goal.plot_xy(ax=ax)
    for obstacle in out[0]["constant"]["obstacles"]:
        if type(obstacle) != Obstacle_Plane:
            obstacle.plot_xy(ax=ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim([-1, 61])
    ax.set_ylim([-5.5, 5.5])
    ax.grid(color="0.9")
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    plt.tight_layout()
    ani = FuncAnimation(fig=fig, func=update_XY, frames=len(time), blit=True, interval=1000 / fps)
    video_writer = ani.save("trajs_xy.mp4", fps=fps / 2, extra_args=["-vcodec", "libx264"], dpi=600)

    def update_XV(frame):
        x = X[:frame, :]
        v = V[:frame, :]
        for i, l in enumerate(line2):
            l.set_xdata(x[:, i])
            l.set_ydata(v[:, i])
        for i, m in enumerate(mark2):
            if frame == 0:
                m.set_xdata(X[0, i])
                m.set_ydata(V[0, i])
            else:
                m.set_xdata(x[-1, i])
                m.set_ydata(v[-1, i])
        return line2 + mark2

    fig, ax = plt.subplots(figsize=(12, 2.5))
    line2 = plt.plot(X, V, linewidth=1.5)
    for i, l in enumerate(line2):
        l.set_color(colors_drones[i])
    mark2 = [None] * len(list_name_urdf)
    for i in range(len(list_name_urdf)):
        mark2[i] = plt.plot(X[0, i], V[0, i], marker="o", color=colors_drones[i])[0]
    ax.set_ylabel(r"$|| \dot{p}_B ||_2$ [m/s]")
    ax.set_xlabel("x [m]")
    ax.set_xlim([-1, 61])
    ax.grid(color="0.9")
    ax.set_axisbelow(True)
    plt.tight_layout()
    ani = FuncAnimation(fig=fig, func=update_XV, frames=len(time), blit=True, interval=1000 / fps)
    video_writer = ani.save("trajs_xv.mp4", fps=fps / 2, extra_args=["-vcodec", "libx264"], dpi=600)


def print_computational_time(list_result_nsga, list_result_ga):
    print("Computational time")
    print("\tNSGA")
    avg_time = 0
    avg_n_chromo = 0
    for result_deap in list_result_nsga:
        df = pd.read_csv(result_deap["pkl"] + ".csv")
        time = df["timestamp"].max() - df["timestamp"].min()
        n_chromo = len(df["chromosome"].unique())
        print(f"\t\t{result_deap['name']} \t | {time/3600:.2f} h | {n_chromo} chromosomes analysed")
        avg_time += time / 3600 / len(list_result_nsga)
        avg_n_chromo += n_chromo / len(list_result_nsga)
    print(f"\t\tavg \t | {avg_time:.2f} h | {avg_n_chromo:.0f} chromosomes analysed")
    avg_time = 0
    avg_n_chromo = 0
    print("\tGA")
    for result_ga in list_result_ga:
        df = pd.read_csv(result_ga["pkl"] + ".csv")
        time = df["timestamp"].max() - df["timestamp"].min()
        n_chromo = len(df["chromosome"].unique())
        print(f"\t\t{result_ga['name']} \t | {time/3600:.2f} h | {n_chromo} chromosomes analysed")
        avg_time += time / 3600 / len(list_result_nsga)
        avg_n_chromo += n_chromo / len(list_result_nsga)
    print(f"\t\tavg \t | {avg_time:.2f} h | {avg_n_chromo:.0f} chromosomes analysed")


def evaluate_bixler():
    with multiprocessing.Pool(processes=len(Codesign_DEAP.define_tasks())) as pool:
        list_out = pool.starmap(
            solve_task,
            [
                (name_urdf, task)
                for name_urdf in ["src/ros_muav/urdf/fixed_wing_drone_back"]
                for task in Codesign_DEAP.define_tasks()
            ],
        )
    N = len(Codesign_DEAP.define_tasks())
    list_ff = [0] * 2
    for out in list_out:
        traj_specs = Postprocess(out)
        t = traj_specs.stats["time"]["trajectory"]
        energy = traj_specs.stats["energy"]["global"]["propeller"] + traj_specs.stats["energy"]["global"]["joint"]
        list_ff[0] += (energy) / N
        list_ff[1] += (t) / N
    return list_ff


def solve_task(name_urdf, task):
    robot = Robot(name_urdf)
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
    out = traj.solve()
    traj.save(out, folder_name=f"result")
    pp = Postprocess(out)
    return out


if __name__ == "__main__":
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fitness_bix3 = evaluate_bixler()

    plot_traj("src/ros_muav/urdf/drone_nsga_46295d0_4", Codesign_DEAP.define_tasks()[1])

    colors_drones = ["#D62728", "#FF7F0E", "#CBBF5F", "#15B7C3", "#2CA02C"]

    list_result_nsga = [
        {"pkl": "result/deap_2023-07-08_09h43m49s", "name": "A"},
        {"pkl": "result/deap_2023-07-09_19h54m55s", "name": "B"},
        {"pkl": "result/deap_2023-07-10_23h54m13s", "name": "C"},
        {"pkl": "result/deap_2023-07-12_09h12m55s", "name": "D"},
        {"pkl": "result/deap_2023-07-13_19h04m08s", "name": "E"},
        {"pkl": "result/deap_2023-07-25_11h13m36s", "name": "F"},
        {"pkl": "result/deap_2023-07-29_22h52m27s", "name": "G"},
        {"pkl": "result/deap_2023-07-31_08h35m26s", "name": "H"},
    ]

    list_result_ga = []

    pareto = compute_total_pareto_front(list_result_nsga)
    fitness_sel_nsga = select_nsga_drones(
        pareto, Stats_Codesign.load(list_result_nsga[-1]["pkl"]).git_info["commit"][:7]
    )
    fitness_sel_ga = select_ga_drones(list_result_ga)
    print_computational_time(list_result_nsga, list_result_ga)

    plot_trajs(
        [
            "src/ros_muav/urdf/fixed_wing_drone_back",
            "src/ros_muav/urdf/drone_nsga_46295d0_1",
            "src/ros_muav/urdf/drone_nsga_46295d0_2",
            "src/ros_muav/urdf/drone_nsga_46295d0_3",
            "src/ros_muav/urdf/drone_nsga_46295d0_4",
        ],
        Codesign_DEAP.define_tasks()[1],
    )

    plot_trajs_moving(
        [
            "src/ros_muav/urdf/fixed_wing_drone_back",
            "src/ros_muav/urdf/drone_nsga_46295d0_1",
            "src/ros_muav/urdf/drone_nsga_46295d0_2",
            "src/ros_muav/urdf/drone_nsga_46295d0_3",
            "src/ros_muav/urdf/drone_nsga_46295d0_4",
        ],
        Codesign_DEAP.define_tasks()[1],
    )

    plot_fitness_ga(list_result_ga)
    plot_diversity_ga(list_result_ga)
    plot_pareto_front(list_result_nsga, fitness_sel_nsga, fitness_sel_ga, fitness_bix3)
    # plot_pareto_front_evolution(list_result_nsga, [20, 40, 60, 80, 100])
    plt.show()
    plot_genes_variability(list_result_nsga)

    plot_traj("src/ros_muav/urdf/drone_nsga_46295d0_2", Codesign_DEAP.define_tasks()[1])
