from codesign.codesign_deap import Codesign_DEAP
import multiprocessing

if __name__ == "__main__":
    # Script for running the co-design methodology (see section VI.B of the paper).
    # If you leave the code unchanged, it will run with the parameters from the paper.
    # Scenarios can be modified by updating the static method `Codesign_DEAP.get_scenarios()`.
    # The trajectory optimization problem formulation can be modified by updating the method `Trajectory_WholeBody_Planner.create()`.
    # The aerodynamic model can be changed by updating the pickle file in `aerodynamics/database_aerodynamic_models`.
    # Parameters of the evolutionary algorithm can be modified here.
    cod = Codesign_DEAP(
        n_pop=100,
        n_gen=100,
        crossover_prob=0.9,
        mutation_prob=1 / 15,
        start_with_feasible_initial_population=False,
        n_processors=multiprocessing.cpu_count(),
    )
    pop = cod.run()
