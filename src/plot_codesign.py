from codesign.codesign_deap import Stats_Codesign
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from deap import creator, base, tools
from codesign.urdf_chromosome_uav import Chromosome_Drone, create_urdf_model, Gene_Weight_Time_Energy
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


def evaluate_optimal_pareto_front(list_result_nsga) -> tools.support.ParetoFront:
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


def select_four_nsga_drones(tag="") -> np.ndarray:
    pareto = evaluate_optimal_pareto_front(list_result_nsga)
    fitness_first_front = np.array([chromo.fitness.values for chromo in pareto])
    N = len(fitness_first_front)
    sel = [
        np.random.randint(0, 10),
        int(N / 3) + np.random.randint(-5, 5),
        2 * int(N / 3) + np.random.randint(-5, 5),
        N - 1 + np.random.randint(-10, 0),
    ]
    list_optimal_drones = []
    for i in range(len(sel)):
        fullpath_model = create_urdf_model(pareto[sel[i]], overwrite=False)
        repo_tree = utils_muav.get_repository_tree(relative_path=True)
        os.rename(f"{fullpath_model}.urdf", f"{repo_tree['urdf']}/drone_nsga_{tag}_{i+1}.urdf")
        os.rename(f"{fullpath_model}.toml", f"{repo_tree['urdf']}/drone_nsga_{tag}_{i+1}.toml")
        list_optimal_drones.append(f"drone_nsga_{tag}_{i+1}")
    return list_optimal_drones


def plot_pareto_front(list_result_nsga, fitness, colors_drones, list_short_name):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    arrowprops = dict(arrowstyle="-|>", fc="w", connectionstyle="arc3")
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 2.105), gridspec_kw={"width_ratios": [1, 0.1]})
    if len(fitness[0]) > 0:
        ax1.plot(
            fitness[0][0],
            fitness[0][1],
            c="k",
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=colors_drones[0],
            # markeredgecolor="brown",
        )
        ax2.plot(
            fitness[0][0],
            fitness[0][1],
            c="k",
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=colors_drones[0],
            # markeredgecolor="brown",
        )
        ax2.annotate(list_short_name[0], xy=fitness[0], xytext=(fitness[0][0] - 10, fitness[0][1] - 0.5), arrowprops=arrowprops)
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
    for i, sel_nsga in enumerate(fitness[1:]):
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
        list_short_name[1],
        xy=fitness[1, :],
        xytext=(fitness[1, 0], fitness[1, 1] - 0.5),
        arrowprops=arrowprops,
    )
    ax1.annotate(
        list_short_name[2],
        xy=fitness[2, :],
        xytext=(fitness[2, 0] - 5, fitness[2, 1] + 0.5),
        arrowprops=arrowprops,
    )
    ax1.annotate(
        list_short_name[3],
        xy=fitness[3, :],
        xytext=(fitness[3, 0], fitness[3, 1] + 0.5),
        arrowprops=arrowprops,
    )
    ax1.annotate(
        list_short_name[4],
        xy=fitness[4, :],
        xytext=(fitness[4, 0], fitness[4, 1] + 0.5),
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


def plot_pareto_front_evolution_video(list_result_nsga):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
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


def plot_trajectories(traj_state, colors_drones, list_robot_name):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # increase tolerance of last goal for visualisation purpose
    for key, value in traj_state.items():
        value["constant"]["goals"][0].position["tol"] = value["constant"]["goals"][0].position["tol"] * 5

    fig = plt.figure(figsize=(10, 2.1))
    ax1 = fig.add_subplot(111)

    i = 0
    for key, value in traj_state.items():
        ax1.plot(
            value["optivar"]["pos_w_b"][0, :],
            value["optivar"]["pos_w_b"][1, :],
            label=list_robot_name[i],
            linewidth=0.8,
            color=colors_drones[i],
        )
        i += 1
    ax1.plot(0, 0, marker="o", color="tab:blue")
    for goal in [next(iter(traj_state.values()))["constant"]["goals"][0]]:
        goal.plot_xy()
    for obstacle in next(iter(traj_state.values()))["constant"]["obstacles"]:
        if type(obstacle) != Obstacle_Plane:
            obstacle.plot_xy()
    plt.legend(list_short_name, loc="upper right", bbox_to_anchor=(0.8, 1))
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
    i = 0
    for key, value in traj_state.items():
        ax1.plot(
            value["optivar"]["pos_w_b"][0, :],
            value["optivar"]["pos_w_b"][1, :],
            label=list_robot_name[i],
            linewidth=0.8,
            color=colors_drones[i],
        )
        i += 1
    ax1.plot(0, 0, marker="o", color="tab:blue")

    for goal in next(iter(traj_state.values()))["constant"]["goals"]:
        goal.plot_xy(ax=ax1)
    for obstacle in next(iter(traj_state.values()))["constant"]["obstacles"]:
        if type(obstacle) != Obstacle_Plane:
            obstacle.plot_xy(ax=ax1)
    plt.legend(list_short_name, loc="upper right", bbox_to_anchor=(0.8, 1))
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_ylim([-5, 5])
    ax1.grid(color="0.9")
    ax1.set_axisbelow(True)
    ax1.set_xlim([-1, 61])
    ax1.set_aspect("equal")
    plt.tight_layout()

    axins = ax1.inset_axes([0.17, -0.25, 0.3, 1])
    i = 0
    for key, value in traj_state.items():
        axins.plot(
            value["optivar"]["pos_w_b"][0, :],
            value["optivar"]["pos_w_b"][1, :],
            label=list_robot_name[i],
            linewidth=1,
            color=colors_drones[i],
        )
        i += 1
    axins.plot(0, 0, marker="o", color="tab:blue")
    for goal in next(iter(traj_state.values()))["constant"]["goals"]:
        goal.plot_xy(ax=axins)
    for obstacle in next(iter(traj_state.values()))["constant"]["obstacles"]:
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
    i = 0
    for key, value in traj_state.items():
        ax2.plot(
            value["optivar"]["pos_w_b"][0, :],
            (value["optivar"]["twist_w_b"][:3, :] ** 2).sum(0) ** 0.5,
            label=list_robot_name[i],
            linewidth=1.5,
            color=colors_drones[i],
        )
        i += 1
    plt.legend(list_short_name, loc="upper right")
    ax2.set_xlabel("x [m]")
    # write ylabel with interpreter latex
    ax2.set_ylabel(r"$|| \dot{p}_B ||_2$ [m/s]")
    ax2.grid(color="0.9")
    ax2.set_axisbelow(True)
    ax2.set_xlim([0, 60])
    plt.tight_layout()
    plt.savefig("trajs_speed_x.png", bbox_inches="tight")
    plt.savefig("trajs_speed_x.pdf", transparent=True, bbox_inches="tight")


def video_trajectories(traj_state, colors_drones, list_robot_name):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    max_time = max(value["time"].max() for value in traj_state.values())
    # increase tolerance of last goal for visualisation purpose
    for key, value in traj_state.items():
        value["constant"]["goals"][0].position["tol"] = value["constant"]["goals"][0].position["tol"] * 5

    fps = 60
    time = np.arange(0, max_time, 1 / 60)
    X = np.zeros((len(time), len(list_robot_name)))
    Y = np.zeros((len(time), len(list_robot_name)))
    V = np.zeros((len(time), len(list_robot_name)))

    for i, (key, value) in enumerate(traj_state.items()):
        X[:, i] = np.interp(time, value["time"], value["optivar"]["pos_w_b"][0, :]).T
        Y[:, i] = np.interp(time, value["time"], value["optivar"]["pos_w_b"][1, :]).T
        V[:, i] = np.interp(time, value["time"], (value["optivar"]["twist_w_b"][:3, :] ** 2).sum(0) ** 0.5).T

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
    mark = [None] * len(list_robot_name)
    for i in range(len(list_robot_name)):
        mark[i] = plt.plot(X[0, i], Y[0, i], marker="o", color=colors_drones[i])[0]
    ax.plot(0, 0, marker="o", color="tab:blue")
    for goal in next(iter(traj_state.values()))["constant"]["goals"]:
        goal.plot_xy(ax=ax)
    for obstacle in next(iter(traj_state.values()))["constant"]["obstacles"]:
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
    mark2 = [None] * len(list_robot_name)
    for i in range(len(list_robot_name)):
        mark2[i] = plt.plot(X[0, i], V[0, i], marker="o", color=colors_drones[i])[0]
    ax.set_ylabel(r"$|| \dot{p}_B ||_2$ [m/s]")
    ax.set_xlabel("x [m]")
    ax.set_xlim([-1, 61])
    ax.grid(color="0.9")
    ax.set_axisbelow(True)
    plt.tight_layout()
    ani = FuncAnimation(fig=fig, func=update_XV, frames=len(time), blit=True, interval=1000 / fps)
    video_writer = ani.save("trajs_xv.mp4", fps=fps / 2, extra_args=["-vcodec", "libx264"], dpi=600)


def print_computational_time_nsga(list_result_nsga):
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


def evaluate_drones(drones_to_be_evaluated):
    n_tasks = len(Codesign_DEAP.define_tasks())
    fitness = np.zeros((len(drones_to_be_evaluated), 2))
    traj_specs = [{drone_name: [] for drone_name in drones_to_be_evaluated} for _ in range(n_tasks)]
    traj_state = [{drone_name: [] for drone_name in drones_to_be_evaluated} for _ in range(n_tasks)]
    for i, drone_name in enumerate(drones_to_be_evaluated):
        fullpath_model = f"{utils_muav.get_repository_tree(relative_path=True)['urdf']}/{drone_name}"
        with multiprocessing.Pool(processes=n_tasks) as pool:
            output_map = pool.starmap(
                Codesign_DEAP.solve_trajectory,
                [(fullpath_model, task, f"result") for task in Codesign_DEAP.define_tasks()],
            )
        temp_traj_name = [t[0] for t in output_map]
        temp_traj_specs = [t[1] for t in output_map]
        temp_traj_state = [t[2] for t in output_map]
        fitness[i, :] = Codesign_DEAP.compute_fitness_MO_given_trajectories(temp_traj_specs, temp_traj_state)
        for j in range(n_tasks):
            traj_specs[j][drone_name] = temp_traj_specs[j]
            traj_state[j][drone_name] = temp_traj_state[j]
    return fitness, traj_specs, traj_state


if __name__ == "__main__":
    # script for plotting the results of `run_codesign.py`
    # if you leave the code unchanged, it will plot the results of the paper.
    # if you want to plot your own results, change the path of the csv files in `list_result_nsga`
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

    # setting use_paper_optimal_drones = True will select the opt drones of the paper
    use_paper_optimal_drones = True

    # select the task to be plotted
    index_task = 1
    # 0, 1, 2, 3, 4

    if use_paper_optimal_drones:
        list_optimal_drones = [
            "drone_nsga_46295d0_1",
            "drone_nsga_46295d0_2",
            "drone_nsga_46295d0_3",
            "drone_nsga_46295d0_4",
        ]
    else:
        list_optimal_drones = select_four_nsga_drones()

    drones_to_be_evaluated = ["fixed_wing_drone_back"] + list_optimal_drones

    fitness, traj_specs, traj_state = evaluate_drones(drones_to_be_evaluated)

    drones_colors = ["#D62728", "#FF7F0E", "#CBBF5F", "#15B7C3", "#2CA02C"]
    list_short_name = ["bix3", "opt1", "opt2", "opt3", "opt4"]

    print_computational_time_nsga(list_result_nsga)
    plot_pareto_front(list_result_nsga, fitness, drones_colors, list_short_name)
    plot_trajectories(traj_state[index_task], drones_colors, list_short_name)
    video_trajectories(traj_state[index_task], drones_colors, list_short_name)

    plt.show()