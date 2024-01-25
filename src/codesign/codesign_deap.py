from deap import base, creator, tools
from codesign.urdf_chromosome_uav import create_urdf_model, Chromosome_Drone
from core.robot import Robot
from traj.trajectory import Postprocess, Task, Gains_Trajectory
from traj.trajectory_wholeboby import Trajectory_WholeBody_Planner
from typing import Dict, Tuple, List
import numpy as np
import utils_muav
import os, shutil
import pandas as pd
import datetime, time
import multiprocessing
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pickle
import subprocess
import copy


class Database_fitness_function:
    def __init__(self, name_database: str, n_objectives: int = 1) -> None:
        self.n_objectives = n_objectives
        self.name_database_with_ext = name_database + ".csv"
        try:
            self.df = pd.read_csv(self.name_database_with_ext)
        except:
            self.create_empty_csv_database()

    def create_empty_csv_database(self) -> None:
        self.df = Database_fitness_function.get_empty_dataframe(self.n_objectives)
        self.df.to_csv(self.name_database_with_ext, index=False)

    def get_fitness_value(self, chromosome: List[float]) -> List[float]:
        if len(self.df["chromosome"]) > 0 and len(self.df[self.df["chromosome"] == str(chromosome)]) > 0:
            list_ff = []
            for i in range(self.n_objectives):
                list_ff.append(self.df[self.df["chromosome"] == str(chromosome)][f"ff_{i}"].min())
        else:
            list_ff = None
        return list_ff

    def get_value(self, chromosome: List[float], key: str) -> float:
        value = self.df[self.df["chromosome"] == str(chromosome)][key].min()
        return value

    def update(
        self,
        chromosome: list,
        list_ff: List[float],
        list_traj_name: List[str],
        list_traj_specs: List[Dict],
        list_traj_state: List[Dict],
        t0_fitness_func: float,
    ) -> None:
        assert len(list_ff) == self.n_objectives
        timestamp = datetime.datetime.timestamp(datetime.datetime.now())
        chromosome_dict = ""
        t = np.nansum(np.array([traj_specs["time"]["trajectory"] for traj_specs in list_traj_specs]))
        energy_prop = np.nansum(
            np.array([traj_specs["energy"]["global"]["propeller"] for traj_specs in list_traj_specs])
        )
        energy_joint = np.nansum(np.array([traj_specs["energy"]["global"]["joint"] for traj_specs in list_traj_specs]))
        energy = energy_prop + energy_joint
        success_rate = 0
        for traj_state in list_traj_state:
            success_rate += 1 / len(list_traj_state) if traj_state["nlp_output"]["success"] else 0
        df = Database_fitness_function.get_empty_dataframe(self.n_objectives)
        df.loc[len(self.df)] = (
            [timestamp, str(chromosome)]  # timestamp, chromosome
            + list_ff
            + [
                None,  # n.generation
                success_rate,  # success_rate
                energy,  # energy
                t,  # time
                time.time() - t0_fitness_func,  # computational_time
                [traj_specs["energy"]["global"]["propeller"] for traj_specs in list_traj_specs],  # energy_propeller
                [traj_specs["energy"]["global"]["joint"] for traj_specs in list_traj_specs],  # energy_joint
                list_traj_name,  # traj_name
                chromosome_dict,  # chromosome_dict
            ]
        )
        df.to_csv(self.name_database_with_ext, index=False, mode="a", header=False)

    def set_chromosome_generation(self, n_generation: int) -> None:
        self.df["n.generation"] = self.df["n.generation"].fillna(n_generation)
        self.df.to_csv(self.name_database_with_ext, index=False)

    @staticmethod
    def get_empty_dataframe(n_objectives):
        df = pd.DataFrame(
            columns=["timestamp", "chromosome"]
            + [f"ff_{i}" for i in range(n_objectives)]
            + [
                "n.generation",
                "success_rate",
                "energy",
                "time",
                "computational_time",
                "energy_propeller",
                "energy_joint",
                "traj_name",
                "chromosome_dict",
            ]
        )
        return df

    def rename(self, new_name: str) -> None:
        os.rename(self.name_database_with_ext, new_name + ".csv")
        self.name_database_with_ext = new_name + ".csv"


class Stats_Codesign:
    def __init__(self, n_pop: int, n_gen: int, n_obj: int, stgs: Dict) -> None:
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.n_obj = n_obj
        self.stgs = stgs
        self.populations = []
        self.populations_offspring = []
        self.fitness = {}
        for i in range(self.n_obj):
            self.fitness[i] = np.zeros((self.n_gen + 1, self.n_pop))
        self.populations_front = []
        self.fitness_front = {}
        for i in range(self.n_obj):
            self.fitness_front[i] = []
        self.fitness_offspring = {}
        for i in range(self.n_obj):
            self.fitness_offspring[i] = np.zeros((self.n_gen + 1, self.n_pop))
        self.diversity = np.zeros((self.n_gen + 1, self.n_pop))
        self.success_rate = np.zeros((self.n_gen + 1, self.n_pop))
        self.git_info = {"commit": subprocess.getoutput("git rev-parse HEAD"), "diff": subprocess.getoutput("git diff")}

    def record(
        self,
        gen: int,
        pop: List[List[float]],
        pop_front: List[List[float]],
        offspring: List[List[float]],
        diversity_matrix: List[float],
        success_rate: List[float],
    ):
        fitness = np.array([chromo.fitness.values for chromo in pop])
        fitness_front = np.array([chromo.fitness.values for chromo in pop_front])
        fitness_offspring = np.array([chromo.fitness.values for chromo in offspring])
        self.populations.append([chromo[:] for chromo in pop])
        for i in range(self.n_obj):
            self.fitness[i][gen, :] = fitness[:, i]
        self.populations_front.append([chromo[:] for chromo in pop_front])
        for i in range(self.n_obj):
            self.fitness_front[i].append(fitness_front[:, i])
        self.populations_offspring.append([chromo[:] for chromo in offspring])
        for i in range(self.n_obj):
            self.fitness_offspring[i][gen, :] = fitness_offspring[:, i]
        self.diversity[gen] = np.array(diversity_matrix.sum(axis=0) / self.n_pop)
        self.success_rate[gen] = np.array(success_rate)

    def save(self, filename: str) -> None:
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> "Stats_Codesign":
        with open(filename + ".pkl", "rb") as file:
            stats = pickle.load(file)
        return stats

    def plot_fitness(
        self,
        save_dir: str = None,
        fill: str = None,
        benchmark: float = None,
        figsize: tuple = (6.4, 4.8),
        remove_fails: bool = True,
    ):
        for i in range(self.n_obj):
            fit = np.copy(self.fitness[i])
            if remove_fails:
                fit[fit >= Codesign_DEAP.get_value_if_nlp_fails()] = np.nan
            fit.sort(axis=1)
            MEAN = fit.mean(axis=1)
            STD = fit.std(axis=1)
            SEM = STD / np.sqrt(self.n_pop)
            MIN = fit[:, 0]
            MAX = fit[:, -1]
            plt.figure(figsize=figsize)
            plt.plot(MEAN)
            plt.plot(MAX)
            plt.plot(MIN)
            if fill == "std":
                plt.fill_between(range(self.n_gen + 1), MEAN + STD, MEAN - STD, alpha=0.2)
            elif fill == "sem":
                plt.fill_between(range(self.n_gen + 1), MEAN + SEM, MEAN - SEM, alpha=0.2)
            elif fill == "iqr":
                plt.fill_between(
                    range(self.n_gen + 1), fit[:, int(self.n_pop * 0.75)], fit[:, int(self.n_pop * 0.25)], alpha=0.2
                )
            plt.legend(["Mean", "Max", "Min"], loc="upper right")
            if benchmark is not None:
                plt.plot([0, self.n_gen], [benchmark, benchmark], "--")
                plt.legend(["Mean", "Max", "Min", fill, "Benchmark"], loc="upper right")
            plt.title(f'Fitness function | {self.stgs["objectives"]["type"][i]} {self.stgs["objectives"]["name"][i]}')
            plt.xlabel("Generation")
            plt.ylabel(f'{self.stgs["objectives"]["name"][i]} [{self.stgs["objectives"]["unit"][i]}]')
            if save_dir is not None:
                plt.savefig(save_dir + f"_f{i}.png")

    def plot_diversity(self, save_dir: str = None, figsize: tuple = (6.4, 4.8)):
        print("plot_diversity started")
        plt.figure(figsize=figsize)
        plt.plot(self.diversity.mean(axis=1))
        plt.title("Diversity")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        if save_dir is not None:
            plt.savefig(save_dir + ".png")
        print("plot_diversity finished")

    def plot_count_unique_chromosomes(self, save_dir: str = None, figsize: tuple = (6.4, 4.8)):
        print("plot_count_unique_chromosomes started")
        n_gen = len(self.populations)
        # count_unique_chromosomes
        count_unique = np.zeros(n_gen)
        for g in range(n_gen):
            count_unique[g] = len(np.unique(self.populations[g], axis=0))
        # count_new_chromosomes
        populations_unique_chromo = []
        for g in range(n_gen):
            populations_unique_chromo.append(np.unique(self.populations[g], axis=0))
        count_new_chromosomes = np.zeros(n_gen)
        count_new_chromosomes[0] = len(populations_unique_chromo[0])
        for g in range(1, n_gen):
            for chromo in populations_unique_chromo[g]:
                count_new_chromosomes[g] += (np.sum(abs(populations_unique_chromo[g - 1] - chromo), axis=1) != 0).all()
        plt.figure(figsize=figsize)
        plt.plot(count_unique, label="unique")
        plt.plot(count_new_chromosomes, label="new")
        plt.legend()
        plt.title("unique/mew chromosomes")
        plt.xlabel("Generation")
        plt.ylabel("n.chromosomes")
        if save_dir is not None:
            plt.savefig(save_dir + ".png")
        print("plot_count_unique_chromosomes finished")

    def plot_chromosome_repetition(self, save_dir: str = None, figsize: tuple = (6.4, 4.8)):
        print("plot_chromosome_repetition started")
        n_gen = len(self.populations)
        pop = np.array(self.populations).reshape((n_gen * self.n_pop, -1))
        unique_chromosomes, idx = np.unique(pop, axis=0, return_index=True)
        unique_fitness = self.fitness[0].reshape(-1)[idx]
        idx = np.argsort(unique_fitness)
        unique_fitness = unique_fitness[idx]
        unique_chromosomes = unique_chromosomes[idx]
        repetition = np.zeros((len(unique_chromosomes), n_gen))
        for i, chromosome in enumerate(unique_chromosomes):
            repetition[i, :] = (pop == chromosome).all(axis=1).reshape((n_gen, self.n_pop)).sum(axis=1)
        cum_repetition = np.cumsum(repetition, axis=0)
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(len(unique_chromosomes) - 1, -1, -1):
            ax.bar(range(n_gen), cum_repetition[i, :], color=np.random.rand(3), edgecolor="black")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Population Composition")
        ax.set_title("Population Composition")
        if save_dir is not None:
            plt.savefig(save_dir + ".png")
        print("plot_chromosome_repetition finished")

    def plot_genes_scaled(self, save_dir: str = None, figsize: tuple = (6.4, 4.8)):
        print("plot_genes_scaled started")
        n_gen = len(self.populations)
        max_value = np.array(Chromosome_Drone().max())
        min_value = np.array(Chromosome_Drone().min())
        range = max_value - min_value
        range[range == 0] = 1
        pop = np.array(self.populations).reshape((n_gen * self.n_pop, -1))
        chromosomes_scaled = (pop - min_value) / range
        plt.figure(figsize=figsize)
        plt.boxplot(chromosomes_scaled, patch_artist=True)
        plt.title("unique chromosomes")
        plt.xlabel("Generation")
        plt.ylabel("unique chromosomes")
        if save_dir is not None:
            plt.savefig(save_dir + ".png")
        print("plot_genes_scaled finished")

    def plot_pareto(
        self,
        gen: int = None,
        save_dir: str = None,
        figsize: tuple = (6.4, 4.8),
        plot_all_individuals: bool = True,
        remove_fails: bool = True,
    ):
        print("plot_pareto started")
        if gen is None:
            # generation start from 0
            gen = len(self.fitness_front[0]) - 1
        fitness = copy.deepcopy(self.fitness)
        fitness_offspring = copy.deepcopy(self.fitness_offspring)
        if remove_fails:
            for i in range(len(fitness)):
                fitness[i][fitness[i] >= Codesign_DEAP.get_value_if_nlp_fails()] = np.nan
                fitness_offspring[i][fitness_offspring[i] >= Codesign_DEAP.get_value_if_nlp_fails()] = np.nan
        plt.figure(figsize=figsize)
        if plot_all_individuals:
            plt.scatter(fitness[0][: gen + 1, :], fitness[1][: gen + 1, :], c="g", marker=".")
            plt.scatter(fitness_offspring[0][: gen + 1, :], fitness_offspring[1][: gen + 1, :], c="g", marker=".")
        plt.scatter(fitness[0][gen, :], fitness[1][gen, :], c="b", marker=".")
        plt.scatter(self.fitness_front[0][gen], self.fitness_front[1][gen], c="r", marker="*")
        plt.title(f"Pareto Front | gen:{gen:.0f}")
        plt.xlabel(f'{self.stgs["objectives"]["name"][0]} [{self.stgs["objectives"]["unit"][0]}]')
        plt.ylabel(f'{self.stgs["objectives"]["name"][1]} [{self.stgs["objectives"]["unit"][1]}]')
        if save_dir is not None:
            plt.savefig(save_dir + ".png")
        print("plot_pareto finished")

    def plot_pareto_video(
        self,
        gen: int = None,
        save_dir: str = None,
        figsize: tuple = (6.4, 4.8),
        plot_all_individuals: bool = True,
        remove_fails: bool = True,
        interval: int = 200,
    ):
        print("plot_pareto_video started")
        if gen is None:
            # generation start from 0
            gen = len(self.fitness_front[0]) - 1
        fitness = copy.deepcopy(self.fitness)
        fitness_offspring = copy.deepcopy(self.fitness_offspring)
        if remove_fails:
            for i in range(len(fitness)):
                fitness[i][fitness[i] >= Codesign_DEAP.get_value_if_nlp_fails()] = np.nan
                fitness_offspring[i][fitness_offspring[i] >= Codesign_DEAP.get_value_if_nlp_fails()] = np.nan
        fig = plt.figure(figsize=figsize)
        plt.xlabel(f'{self.stgs["objectives"]["name"][0]} [{self.stgs["objectives"]["unit"][0]}]')
        plt.ylabel(f'{self.stgs["objectives"]["name"][1]} [{self.stgs["objectives"]["unit"][1]}]')
        ax = fig.add_subplot(111)
        frames = []
        for g in range(gen + 1):
            p = []
            if plot_all_individuals:
                p.append(plt.scatter(fitness[0][: g + 1, :], fitness[1][: g + 1, :], c="g", marker=".", animated=True))
                p.append(
                    plt.scatter(
                        fitness_offspring[0][: g + 1, :],
                        fitness_offspring[1][: g + 1, :],
                        c="g",
                        marker=".",
                        animated=True,
                    )
                )
            p.append(plt.scatter(fitness[0][g, :], fitness[1][g, :], c="b", marker=".", animated=True))
            fitness_front = np.array([self.fitness_front[0][g], self.fitness_front[1][g]]).reshape((2, -1))
            if remove_fails:
                fitness_front[:, fitness_front[0, :] >= Codesign_DEAP.get_value_if_nlp_fails()] = np.nan
            p.append(plt.scatter(fitness_front[0, :], fitness_front[1, :], c="r", marker="*", animated=True))
            p.append(
                plt.text(
                    0.5,
                    1.01,
                    f"Pareto Front | gen:{g:.0f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    transform=ax.transAxes,
                )
            )
            frames.append(p)
        ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True, repeat_delay=1000)
        if save_dir is not None:
            ani.save(save_dir + ".mp4")
        print("plot_pareto_video finished")

    def plot_pareto_energy_time(
        self,
        df: pd.DataFrame,
        gen: int = None,
        save_dir: str = None,
        figsize: tuple = (6.4, 4.8),
        plot_all_individuals: bool = True,
    ):
        print("plot_pareto_energy_time started")
        if gen is None:
            gen = len(self.fitness_front[0]) - 1
        plt.figure(figsize=figsize)
        if plot_all_individuals:
            energy = np.array(
                [
                    df[df["chromosome"] == str(chromosome)]["energy"].min()
                    for pop in self.populations[: gen + 1] + self.populations_offspring[: gen + 1]
                    for chromosome in pop
                ]
            )
            time = np.array(
                [
                    df[df["chromosome"] == str(chromosome)]["time"].min()
                    for pop in self.populations[: gen + 1] + self.populations_offspring[: gen + 1]
                    for chromosome in pop
                ]
            )
            success_rate = np.array(
                [
                    df[df["chromosome"] == str(chromosome)]["success_rate"].min()
                    for pop in self.populations[: gen + 1] + self.populations_offspring[: gen + 1]
                    for chromosome in pop
                ]
            )
            plt.scatter(energy[success_rate == 1], time[success_rate == 1], c="g", marker=".")
        # fitness
        energy = np.array(
            [df[df["chromosome"] == str(chromosome)]["energy"].min() for chromosome in self.populations[gen]]
        )
        time = np.array([df[df["chromosome"] == str(chromosome)]["time"].min() for chromosome in self.populations[gen]])
        success_rate = np.array(
            [df[df["chromosome"] == str(chromosome)]["success_rate"].min() for chromosome in self.populations[gen]]
        )
        plt.scatter(energy[success_rate == 1], time[success_rate == 1], c="b", marker=".")
        # fitness front
        energy = np.array(
            [df[df["chromosome"] == str(chromosome)]["energy"].min() for chromosome in self.populations_front[gen]]
        )
        time = np.array(
            [df[df["chromosome"] == str(chromosome)]["time"].min() for chromosome in self.populations_front[gen]]
        )
        success_rate = np.array(
            [
                df[df["chromosome"] == str(chromosome)]["success_rate"].min()
                for chromosome in self.populations_front[gen]
            ]
        )
        plt.scatter(energy[success_rate == 1], time[success_rate == 1], c="r", marker="*")
        plt.scatter
        plt.title(f"Pareto Front E-t | gen:{gen:.0f}")
        plt.xlabel("Energy [J]")
        plt.ylabel("Time [s]")
        if save_dir is not None:
            plt.savefig(save_dir + ".png")
        print("plot_pareto_energy_time stopped")


class Codesign_DEAP:
    def __init__(
        self,
        n_pop: int = 12,
        n_gen: int = 50,
        crossover_prob: float = 0.6,
        mutation_prob: float = 0.01,
        n_processors: int = 1,
        start_with_feasible_initial_population: bool = False,
    ) -> None:
        self.stgs = {}
        self.n_objectives = 2
        self.stgs["objectives"] = {}
        self.stgs["objectives"]["name"] = ["energy", "time"]
        self.stgs["objectives"]["type"] = ["min", "min"]
        self.stgs["objectives"]["unit"] = ["J", "s"]
        self.stgs["n_pop"] = n_pop
        self.stgs["n_gen"] = n_gen
        self.stgs["crossover_prob"] = crossover_prob
        self.stgs["mutation_prob"] = mutation_prob
        self.stgs["n_processors"] = n_processors
        self.stgs["start_with_feasible_initial_population"] = start_with_feasible_initial_population
        self.str_date = utils_muav.get_date_str()
        self.name_temp_output = f"deap_temp"
        self.stats = Stats_Codesign(self.stgs["n_pop"], self.stgs["n_gen"], self.n_objectives, self.stgs)
        self.urdf_locations = None

    def get_initial_population(self) -> np.ndarray:
        print("start get_initial_population")
        tested_chromosomes = []
        initial_population = []
        n_tasks = len(self.get_scenarios())
        while len(initial_population) < self.stgs["n_pop"]:
            candidate_chromosomes = []
            while len(candidate_chromosomes) < self.stgs["n_pop"]:
                chromosome = Chromosome_Drone().get_random()
                if chromosome not in tested_chromosomes:
                    tested_chromosomes.append(chromosome)
                    candidate_chromosomes.append(chromosome)
            fitness_values = self.compute_fitness_given_list_chromosomes(candidate_chromosomes)
            for chromosome, fit_val in zip(candidate_chromosomes, fitness_values):
                if fit_val[0] < Codesign_DEAP.get_value_if_nlp_fails() * n_tasks:
                    initial_population.append(chromosome)
        initial_population = initial_population[: self.stgs["n_pop"]]
        print("stop get_initial_population")
        Database_fitness_function(self.name_temp_output, self.n_objectives).set_chromosome_generation(-1)
        return initial_population

    def run(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Chromosome", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("get_chromosome", tools.initIterate, creator.Chromosome, Chromosome_Drone().get_random)
        toolbox.register("population", tools.initRepeat, list, toolbox.get_chromosome)

        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", self.cxOnePoint)
        toolbox.register("mutate", self.mutUniform)
        toolbox.register("select", tools.selNSGA2)
        self.pareto = tools.ParetoFront()

        # create initial population
        pop = toolbox.population(n=self.stgs["n_pop"])
        if self.stgs["start_with_feasible_initial_population"]:
            list_pop = self.get_initial_population()
            for i in range(self.stgs["n_pop"]):
                pop[i] = creator.Chromosome(list_pop[i])
        # evaluate initial population
        invalid_chromo = [chromo for chromo in pop if not chromo.fitness.valid]
        self.compute_fitness_given_list_chromosomes(invalid_chromo)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_chromo)
        for chromo, fit in zip(invalid_chromo, fitnesses):
            chromo.fitness.values = fit
        print(f"Evaluated {len(invalid_chromo)} individuals")
        # compute crowding distance
        pop = toolbox.select(pop, self.stgs["n_pop"])

        # begin the evolution
        self.on_generation(gen=0, pop=pop, offspring=pop)

        for gen in range(1, self.stgs["n_gen"] + 1):
            # select offspring (parents)
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(chromo) for chromo in offspring]
            # apply crossover and mutation on offspring
            for chromo1, chromo2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(chromo1, chromo2)
                toolbox.mutate(chromo1)
                toolbox.mutate(chromo2)
                del chromo1.fitness.values, chromo2.fitness.values
            # evaluate offspring
            invalid_chromo = [chromo for chromo in offspring if not chromo.fitness.valid]
            self.compute_fitness_given_list_chromosomes(invalid_chromo)
            fitnesses = toolbox.map(toolbox.evaluate, invalid_chromo)
            for chromo, fit in zip(invalid_chromo, fitnesses):
                chromo.fitness.values = fit
            print(f"Evaluated {len(invalid_chromo)} individuals")
            # select the next generation population
            pop = toolbox.select(pop + offspring, self.stgs["n_pop"])
            # end of generation
            self.on_generation(gen=gen, pop=pop, offspring=offspring)
        self.on_stop()
        return pop

    def on_generation(self, gen: int, pop: List[List[float]], offspring: List[List[float]]) -> None:
        self.pareto.update(pop)
        self.stats.record(
            gen=gen,
            pop=pop,
            pop_front=self.pareto.items,
            offspring=offspring,
            diversity_matrix=Chromosome_Drone().compute_population_distance_matrix(pop, self.stgs["n_processors"]),
            success_rate=[
                Database_fitness_function(self.name_temp_output, self.n_objectives).get_value(chromo, "success_rate")
                for chromo in pop
            ],
        )
        self.stats.save(filename=self.name_temp_output)
        # Save the plots.
        try:
            self.stats.plot_fitness(save_dir="fitness", fill="sem")
            self.stats.plot_pareto(save_dir="pareto_gen", gen=gen, plot_all_individuals=False)
            self.stats.plot_pareto(save_dir="pareto_tot", gen=gen)
            df = Database_fitness_function(self.name_temp_output, self.n_objectives).df
            self.stats.plot_pareto_energy_time(df, save_dir="pareto_ET_gen", gen=gen)
            self.stats.plot_count_unique_chromosomes(save_dir="count_unique_chromosomes")
            self.stats.plot_genes_scaled(save_dir="genes")
            self.stats.plot_diversity(save_dir="diversity")
            if np.mod(gen, 10) == 0:
                self.stats.plot_chromosome_repetition(
                    save_dir="chromosome_repetition", figsize=[0.106 * self.stgs["n_gen"], 0.048 * self.stgs["n_pop"]]
                )
        except:
            print("An error occured while plotting!")
        # Set chromosome generation.
        Database_fitness_function(self.name_temp_output).set_chromosome_generation(gen)
        # Delete urdfs
        shutil.rmtree(self.urdf_locations)

    def on_stop(self) -> None:
        db_ff = Database_fitness_function(f"{self.name_temp_output}", self.n_objectives)
        db_ff.rename(f"result/deap_{self.str_date}")
        self.stats.save(f"result/deap_{self.str_date}")
        try:
            self.stats.plot_fitness(save_dir=f"result/deap_{self.str_date}/fitness", fill="sem")
            self.stats.plot_pareto(save_dir=f"result/deap_{self.str_date}/pareto_gen", plot_all_individuals=False)
            self.stats.plot_pareto(save_dir=f"result/deap_{self.str_date}/pareto_tot")
            self.stats.plot_pareto_energy_time(db_ff.df, save_dir=f"result/deap_{self.str_date}/pareto_ET_gen")
            self.stats.plot_count_unique_chromosomes(save_dir=f"result/deap_{self.str_date}/count_unique_chromosomes")
            self.stats.plot_genes_scaled(save_dir=f"result/deap_{self.str_date}/genes")
            self.stats.plot_diversity(save_dir=f"result/deap_{self.str_date}/diversity")
            self.stats.plot_chromosome_repetition(
                save_dir=f"result/deap_{self.str_date}/chromosome_repetition",
                figsize=[0.106 * self.stgs["n_gen"], 0.048 * self.stgs["n_pop"]],
            )
        except:
            print("An error occured while plotting!")
        print("End")

    def evaluate(self, chromosome) -> Tuple[float]:
        db_ff = Database_fitness_function(self.name_temp_output, self.n_objectives)
        list_ff = db_ff.get_fitness_value(chromosome)
        if list_ff is None:
            list_list_ff = self.compute_fitness_given_list_chromosomes([chromosome])
            list_ff = list_list_ff[0]
        return tuple(list_ff)

    def cxOnePoint(self, chromo1, chromo2):
        assert len(chromo1) == len(chromo2)
        if np.random.rand() < self.stgs["crossover_prob"]:
            list_possible_crossover_points = [9, 15, 20]
            cxpoint = np.random.choice(list_possible_crossover_points)
            # be careful, if chromo1 and chromo2 are not lists, then the following line will not work
            # check https://deap.readthedocs.io/en/master/tutorials/advanced/numpy.html
            chromo1[cxpoint:], chromo2[cxpoint:] = chromo2[cxpoint:], chromo1[cxpoint:]
        chromo1[:] = Chromosome_Drone().clean_list(chromo1)
        chromo2[:] = Chromosome_Drone().clean_list(chromo2)
        return chromo1, chromo2

    def mutUniform(self, chromo):
        for i in range(len(chromo)):
            if np.random.rand() < self.stgs["mutation_prob"]:
                chromo[i] = Chromosome_Drone().get_random(clean_Chromosome=False)[i]
        chromo[:] = Chromosome_Drone().clean_list(chromo)
        return chromo

    @staticmethod
    def get_scenarios():
        list_tasks = [
            Task.define_slalom_trajectory(
                distance_start_obstacle=20,
                obstacles_radius=4,
                initial_velocity=[8, 0, 0, 0, 0, 0],
                initial_orientation_rpy_deg=[0, 0, 0],
            ),
            Task.define_slalom_trajectory(
                distance_start_obstacle=20,
                obstacles_radius=4,
                initial_velocity=[10, 0, 0, 0, 0, 0],
                initial_orientation_rpy_deg=[0, 0, 0],
            ),
            Task.define_slalom_trajectory(
                distance_start_obstacle=20,
                obstacles_radius=4,
                initial_velocity=[12, 0, 0, 0, 0, 0],
                initial_orientation_rpy_deg=[0, 0, 0],
            ),
            Task.define_slalom_trajectory(
                distance_start_obstacle=20,
                obstacles_radius=4,
                initial_velocity=[10, 0, 0, 0, 0, 0],
                initial_orientation_rpy_deg=[0, 5, 0],
            ),
            Task.define_slalom_trajectory(
                distance_start_obstacle=20,
                obstacles_radius=4,
                initial_velocity=[10, 0, 0, 0, 0, 0],
                initial_orientation_rpy_deg=[0, -5, 0],
            ),
        ]
        return list_tasks

    @staticmethod
    def solve_trajectory(fullpath_model: str, task: Task, folder_name: str) -> Tuple[str, Dict, Dict]:
        robot = Robot(fullpath_model)
        robot.set_joint_limit()
        robot.set_propeller_limit()
        traj = Trajectory_WholeBody_Planner(
            robot=robot, knots=task.knots, time_horizon=None, regularize_control_input_variations=True
        )
        traj.set_gains(
            Gains_Trajectory(
                cost_function_weight_time=robot.controller_parameters["weight_time_energy"],
                cost_function_weight_energy=1,
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
            traj.save(out, folder_name=folder_name)
            pp = Postprocess(out)
        except:
            pp = Postprocess()
            out = traj._get_empty_out()
        return traj.name_trajectory, pp.stats, out

    @staticmethod
    def get_value_if_nlp_fails():
        return 1e6

    @staticmethod
    def compute_fitness_MO_given_trajectories(list_traj_specs: List, list_traj_state: List) -> List[float]:
        # the fitness function is composed by two objectives: the energy and the time
        # if the nlp fails, then the fitness function is set to a very big value
        # the fitness function is aimed to be minimized
        N = len(list_traj_specs)
        n_objectives = 2
        list_ff = [0] * n_objectives
        for traj_specs, traj_state in zip(list_traj_specs, list_traj_state):
            if traj_state["nlp_output"]["success"]:
                t = traj_specs["time"]["trajectory"]
                energy = traj_specs["energy"]["global"]["propeller"] + traj_specs["energy"]["global"]["joint"]
                list_ff[0] += (energy) / N
                list_ff[1] += (t) / N
            else:
                list_ff[0] += Codesign_DEAP.get_value_if_nlp_fails()
                list_ff[1] += Codesign_DEAP.get_value_if_nlp_fails()
        return list_ff

    def compute_fitness_given_list_chromosomes(self, list_chromosomes: List[List[float]]) -> List[List[float]]:
        list_chromosomes_to_be_analysed = []
        db_ff = Database_fitness_function(self.name_temp_output, self.n_objectives)
        # remove already computed chromosomes and duplicates
        for chromosome in list_chromosomes:
            chromosome = Chromosome_Drone().clean_list(chromosome)
            list_ff = db_ff.get_fitness_value(chromosome)
            if (list_ff is None) and (chromosome not in list_chromosomes_to_be_analysed):
                fullpath_model = create_urdf_model(chromosome=chromosome, overwrite=False)
                if self.urdf_locations is None:
                    self.urdf_locations = os.path.dirname(fullpath_model)
                list_chromosomes_to_be_analysed.append(chromosome)
        # define tasks
        list_tasks = self.get_scenarios()
        n_tasks = len(list_tasks)
        # compute n_processors
        n_processors = min(
            self.stgs["n_processors"], len(list_chromosomes_to_be_analysed) * n_tasks, multiprocessing.cpu_count()
        )
        n_processors = max(n_processors, 1)
        # compute fitness function
        with multiprocessing.Pool(processes=n_processors) as pool:
            output_map = pool.starmap(
                Codesign_DEAP.solve_trajectory,
                [
                    (create_urdf_model(chromosome=chromosome, overwrite=False), task, f"result/{self.str_date}")
                    for chromosome in list_chromosomes_to_be_analysed
                    for task in list_tasks
                ],
            )
        for i, chromosome in enumerate(list_chromosomes_to_be_analysed):
            list_traj_name = [t[0] for t in output_map[n_tasks * i : n_tasks * (i + 1)]]
            list_traj_specs = [t[1] for t in output_map[n_tasks * i : n_tasks * (i + 1)]]
            list_traj_state = [t[2] for t in output_map[n_tasks * i : n_tasks * (i + 1)]]
            list_ff = Codesign_DEAP.compute_fitness_MO_given_trajectories(list_traj_specs, list_traj_state)
            db_ff.update(chromosome, list_ff, list_traj_name, list_traj_specs, list_traj_state, 0)
        db_ff = Database_fitness_function(self.name_temp_output, self.n_objectives)
        list_list_ff = [
            db_ff.get_fitness_value(Chromosome_Drone().clean_list(chromosome)) for chromosome in list_chromosomes
        ]
        return list_list_ff


if __name__ == "__main__":
    cod = Codesign_DEAP(
        n_pop=100,
        n_gen=100,
        crossover_prob=0.9,
        mutation_prob=1 / 15,
        start_with_feasible_initial_population=False,
        n_processors=multiprocessing.cpu_count(),
    )
    pop = cod.run()
