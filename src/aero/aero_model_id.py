import os, pandas as pd, casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import pickle
import utils_muav
from sklearn.linear_model import Lasso
from sklearn import metrics
import multiprocessing


def load_pickle_aerodynamic_model(name_pickle):
    if os.path.isfile(name_pickle):
        fun = pickle.load(open(name_pickle, "rb"))
    else:
        print(f"aerodynamic model {name_pickle} doesn't exist. Loading an empty model.")
        fun = {}
        fun["CD"] = lambda x, y, z: 0
        fun["CY"] = lambda x, y, z: 0
        fun["CL"] = lambda x, y, z: 0
        fun["Cm"] = lambda x, y, z: 0
        fun["Cn"] = lambda x, y, z: 0
        fun["Cl"] = lambda x, y, z: 0
    return fun


def compute_metrics(fun, data, name):
    y_true = data[name]
    y_est = np.zeros_like(y_true)
    for i, (alpha, beta, reynolds) in enumerate(zip(data["alpha"], data["beta"], data["reynolds"])):
        y_est[i] = fun[name](alpha, beta, reynolds)
    N = len(y_true)
    m = {}
    m["MAE"] = metrics.mean_absolute_error(y_true, y_est)
    m["MSE"] = metrics.mean_squared_error(y_true, y_est)
    m["RMSE"] = m["MSE"] ** 0.5
    m["NRMSE"] = m["RMSE"] / (y_true.max() - y_true.min() + 1e-12)
    m["SMAPE"] = 100 * sum(abs(y_true - y_est)) / sum(abs(y_true) + abs(y_est) + 1e-12) / N
    m["MAPE"] = metrics.mean_absolute_percentage_error(y_true, y_est)
    m["MedAE"] = metrics.median_absolute_error(y_true, y_est)
    m["R2"] = metrics.r2_score(y_true, y_est)
    m["MaxError"] = metrics.max_error(y_true, y_est)

    s = f"\t{name}"
    for key in ["MAE", "SMAPE", "NRMSE", "R2"]:
        s += f"\t{key}: {m[key]:.4f}"
    print(s)
    return m


def get_model_via_regression(fun_row_A: cs.casadi.Function, data: np.ndarray, name: str):
    print(f"\tstart regression for {name}")
    fun_matrix_AT = fun_row_A.map(len(data["alpha"]))
    A = fun_matrix_AT(data["alpha"], data["beta"], data["reynolds"]).T
    b = data[name]

    solver = "lasso"

    if solver == "casadi":
        try:
            casadi_solver = "ipopt"
            if casadi_solver == "osqp":
                opti = cs.Opti("conic")
                p_opts = {"expand": True, "error_on_fail": False}
                s_opts = {"verbose": False}
            elif casadi_solver == "ipopt":
                opti = cs.Opti()
                p_opts = {"expand": True, "error_on_fail": False, "print_time": False}
                s_opts = {
                    "print_level": 0,
                    "sb": "yes",
                    "linear_solver": "ma27",
                    "hessian_approximation": "limited-memory",
                    "max_iter": 10000,
                }
            else:
                exit()
            opti.solver(casadi_solver, p_opts, s_opts)
            x = opti.variable(A.shape[1], 1)
            L = A @ x - b
            opti.minimize(L.T @ L)
            if name == "CD":
                opti.subject_to(A @ x > 0)
            sol = opti.solve()
            coeff = np.array(sol.value(x))
        except:
            coeff = np.linalg.lstsq(A.full(), data[name], rcond=-1)[0]
            print(f"\t\tcasadi solver failed, using lstsq instead")
    elif solver == "lstsq":
        coeff = np.linalg.lstsq(A.full(), data[name], rcond=-1)[0]
    elif solver == "lasso":
        clf = Lasso(alpha=1e-7, max_iter=10000)
        # weights = 1 / (data["beta"] * data["beta"] + 0.1)
        clf.fit(A.full(), data[name])  # , sample_weight=weights)
        coeff = clf.coef_
        if metrics.r2_score(b, A.full() @ coeff) < 0.5:
            print(f"\t\tregression {name}: R2 < 0.5, using lstsq instead")
            coeff = np.linalg.lstsq(A.full(), data[name], rcond=-1)[0]
    else:
        exit()

    print(f"\tend regression for {name}, {(coeff!=0).sum()} non-zero coefficients")

    alpha = cs.SX.sym("alpha")
    beta = cs.SX.sym("beta")
    reynolds = cs.SX.sym("Re")
    return cs.Function(
        name,
        [alpha, beta, reynolds],
        [cs.DM(coeff).T @ fun_row_A(alpha, beta, reynolds)],
        ["alpha", "beta", "reynolds"],
        [name],
    )


if __name__ == "__main__":
    for sim_name in [
        "wing0009_1_0_230426",
        "wing0009_1_5_230426",
        "wing0009_2_0_230426",
        "wing0009_2_5_230426",
        "wing0009_3_0_230426",
        "wing0009_3_5_230426",
        "wing0009_4_0_230426",
        "wing0009_4_5_230426",
        "wing0009_5_0_230426",
        "fuselage0014_tail0009_221202_101041",
    ]:
        print(sim_name)

        repo_tree = utils_muav.get_repository_tree()
        repo_tree["name_pickle"] = os.path.join(repo_tree["database_aerodynamic_models"], sim_name + ".p")

        df = pd.read_csv(os.path.join(repo_tree["output"], sim_name + "/aero_database.csv"))

        df_mask = df[
            (df["CD"] >= 0) & (df["Beta"] >= 0) & (df["Beta"] <= 130) & (df["Alpha"] >= -10) & (df["Alpha"] <= 10)
        ]
        if sim_name == "fuselage0014_tail0009_221202_101041":
            df_mask = df_mask[(df_mask["Beta"] <= 30)]
        df_mask_beta = df_mask.copy()
        df_mask_beta["Beta"] = -df_mask_beta["Beta"]
        df_mask_beta["CY"] = -df_mask_beta["CY"]
        df_mask_beta["Cl"] = -df_mask_beta["Cl"]
        df_mask_beta["Cn"] = -df_mask_beta["Cn"]

        df = pd.concat([df_mask, df_mask_beta])

        try:
            df = df.sample(n=10000, random_state=1)
        except:
            pass

        data = {}
        data["alpha"] = df["Alpha"].values * np.pi / 180
        data["beta"] = df["Beta"].values * np.pi / 180
        data["reynolds"] = df["reynolds"].values
        for coeff in ["CD", "CL", "CY", "Cn", "Cl", "Cm"]:
            data[coeff] = df[coeff].values

        lib = {}
        x = cs.SX.sym("x")
        alpha = cs.SX.sym("alpha")
        beta = cs.SX.sym("beta")
        reynolds = cs.SX.sym("Re")
        library_fun = {}
        vector_alpha = {}
        vector_beta = {}
        vector_Re = {}
        for name in ["CD", "CL", "CY", "Cn", "Cl", "Cm"]:
            vector_alpha[name] = cs.vcat((1, cs.cos(alpha), cs.sin(alpha), cs.sin(2 * alpha)))
            vector_beta[name] = cs.vcat(
                (
                    1,
                    cs.cos(beta),
                    cs.sin(beta),
                    cs.sin(beta) ** 2,
                    cs.sin(2 * beta),
                    cs.sin(2 * beta) ** 2,
                    cs.cos(2 * beta) ** 3,
                )
            )
            vector_Re[name] = cs.vcat((1, (reynolds - 2.5e4) ** (-0.279)))
        if sim_name == "wing0009_4_5_230426" or sim_name == "wing0009_5_0_230426":
            vector_alpha["CD"] = cs.vcat((1, cs.cos(alpha), cs.cos(alpha**2)))
            vector_beta["CD"] = cs.vcat(
                (1, cs.cos(beta), cs.sin(beta) ** 2, cs.sin(2 * beta) ** 2, cs.cos(2 * beta) ** 3)
            )
        elif sim_name == "fuselage0014_tail0009_221202_101041":
            vector_alpha["CD"] = cs.vcat((1, cs.cos(alpha), cs.sin(2 * alpha)))
            vector_beta["CD"] = cs.vcat((1, cs.sin(beta) ** 2, cs.sin(2 * beta) ** 2, cs.cos(2 * beta) ** 3))
            vector_alpha["CL"] = cs.vcat((cs.cos(alpha), cs.sin(2 * alpha)))
            vector_beta["CL"] = cs.vcat((cs.sin(2 * beta) ** 2, cs.cos(2 * beta) ** 3))
            vector_Re["CL"] = cs.vcat((1,))
            vector_alpha["Cm"] = cs.vcat((1, cs.cos(alpha), cs.sin(2 * alpha)))
            vector_beta["Cm"] = cs.vcat((cs.cos(beta), cs.cos(2 * beta) ** 3))
            vector_Re["Cm"] = cs.vcat((1,))
        else:
            vector_alpha["CD"] = cs.vcat((1, cs.cos(alpha)))
            vector_beta["CD"] = cs.vcat((1, cs.cos(beta), cs.sin(beta) ** 2, cs.sin(2 * beta) ** 2))

        for name in ["CD", "CL", "CY", "Cn", "Cl", "Cm"]:
            vector_abr = cs.reshape(
                cs.reshape(vector_alpha[name] @ vector_beta[name].T, -1, 1) @ vector_Re[name].T, -1, 1
            )
            library_fun[name] = cs.Function("library", [alpha, beta, reynolds], [vector_abr])

        fun = {}

        def process_element(c):
            return c, get_model_via_regression(library_fun[c], data, c)

        with multiprocessing.Pool(processes=6) as pool:
            results = pool.map(process_element, ["CD", "CL", "CY", "Cn", "Cl", "Cm"])
            for c, result in results:
                fun[c] = result
                compute_metrics(fun, data, c)

        s = pickle.dump(fun, open(repo_tree["name_pickle"], "wb"))
