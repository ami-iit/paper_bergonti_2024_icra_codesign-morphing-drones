from codesign.codesign_deap import Codesign_DEAP
import multiprocessing

if __name__ == "__main__":
    # script for running the codesign method
    cod = Codesign_DEAP(
        n_pop=100,
        n_gen=100,
        crossover_prob=0.9,
        mutation_prob=1 / 15,
        start_with_feasible_initial_population=False,
        n_processors=multiprocessing.cpu_count(),
    )
    pop = cod.run()
