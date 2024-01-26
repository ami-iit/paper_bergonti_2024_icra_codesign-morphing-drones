import os, math, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils_muav
from aero.aero_model_id import load_pickle_aerodynamic_model, compute_metrics


def plot_aerodynamics(fun, data, list_coeff, reynolds_list):
    grid_alpha, grid_beta = np.meshgrid(
        np.linspace(min(data["alpha"]), max(data["alpha"]), num=100),
        np.linspace(min(data["beta"]), max(data["beta"]), num=100),
    )
    for c in list_coeff:
        fig = plt.figure(figsize=(16, 9), dpi=80)
        for i, reynolds in enumerate(reynolds_list):
            ax = fig.add_subplot(2, int(len(reynolds_list) / 2), i + 1, projection="3d")
            ax.scatter(
                data["alpha"][data["reynolds"] == reynolds] * 180 / math.pi,
                data["beta"][data["reynolds"] == reynolds] * 180 / math.pi,
                data[c][data["reynolds"] == reynolds],
                s=0.5,
            )
            ax.plot_surface(
                grid_alpha * 180 / math.pi,
                grid_beta * 180 / math.pi,
                np.array(fun[c](grid_alpha, grid_beta, reynolds)),
                rstride=1,
                cstride=1,
                cmap="viridis",
                edgecolor="none",
            )
            ax.set_xlabel("alpha [deg]")
            ax.set_ylabel("beta [deg]")
            ax.set_zlabel(c)
            ax.set_title(f"reynolds={np.round(reynolds)}")
            if c == "CD" and fun[c](grid_alpha, grid_beta, reynolds).full().min() < 0:
                Warning("CD is negative")
            plt.draw()
        fig.suptitle(c)
        # plt.savefig("{}".format(c))
    plt.show()


def plot_aerodynamic_coefficient(
    fun, data, coeff_name, reynolds_value, figsize: tuple = (16, 9), dpi: float = 80, label: str = ""
):
    grid_alpha, grid_beta = np.meshgrid(
        np.linspace(min(data["alpha"]), max(data["alpha"]), num=100),
        np.linspace(min(data["beta"]), max(data["beta"]), num=100),
    )
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes(projection="3d")
    ax.scatter(
        data["alpha"][data["reynolds"] == reynolds_value] * 180 / math.pi,
        data["beta"][data["reynolds"] == reynolds_value] * 180 / math.pi,
        data[coeff_name][data["reynolds"] == reynolds_value],
        s=0.5,
    )
    ax.plot_surface(
        grid_alpha * 180 / math.pi,
        grid_beta * 180 / math.pi,
        np.array(fun[coeff_name](grid_alpha, grid_beta, reynolds_value)),
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    # set xlabel using latex
    xlabel = ax.set_xlabel(r"$\alpha \left[ deg \right]$")
    ylabel = ax.set_ylabel(r"$\beta \left[ deg \right]$")
    # if coeff_name == "CD":
    #     ax.set_zlabel(r"$C_D $")
    #     ax.set_title(r"$C_D $")
    # elif coeff_name == "CL":
    #     ax.set_zlabel(r"$C_L $")
    #     ax.set_title(r"$C_L $")
    # elif coeff_name == "CY":
    #     ax.set_zlabel(r"$C_Y $")
    #     ax.set_title(r"$C_Y $")
    # elif coeff_name == "Cl":
    #     ax.set_zlabel(r"$C_l $")
    #     ax.set_title(r"$C_l $")
    # elif coeff_name == "Cm":
    #     ax.set_zlabel(r"$C_m $")
    #     ax.set_title(r"$C_m $")
    # elif coeff_name == "Cn":
    #     ax.set_zlabel(r"$C_n $")
    #     ax.set_title(r"$C_n $")
    plt.savefig(
        f"aero_coeff_{label}_Re{int(reynolds_value)}_{coeff_name}.png",
        format="png",
        bbox_inches="tight",
        bbox_extra_artists=[xlabel],
    )
    plt.savefig(
        f"aero_coeff_{label}_Re{int(reynolds_value)}_{coeff_name}.pdf",
        format="pdf",
        bbox_inches="tight",
        bbox_extra_artists=[xlabel],
    )


if __name__ == "__main__":
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    df_metr = pd.DataFrame(
        columns=["sim_name", "coeff", "MAE", "MSE", "RMSE", "NRMSE", "SMAPE", "MAPE", "MedAE", "R2", "MaxError"]
    )
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

        fun = load_pickle_aerodynamic_model(repo_tree["name_pickle"])
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

        data = {}
        data["alpha"] = df["Alpha"].values * math.pi / 180
        data["beta"] = df["Beta"].values * math.pi / 180
        data["reynolds"] = df["reynolds"].values
        chord = df["chord"].values[0]
        density = df["density"].values[0]
        viscosity = df["viscosity"].values[0]
        for coeff in ["CD", "CL", "CY", "Cn", "Cl", "Cm"]:
            data[coeff] = df[coeff].values

        for coeff in ["CD", "CL", "CY", "Cn", "Cl", "Cm"]:
            m = compute_metrics(fun, data, coeff)
            m["sim_name"] = sim_name
            m["coeff"] = coeff
            df_metr.loc[len(df_metr)] = m
            plot_aerodynamic_coefficient(
                fun, data, coeff, (10 * chord * density / viscosity).round(), label=sim_name, figsize=(3, 3), dpi=80
            )
            plot_aerodynamic_coefficient(
                fun, data, coeff, (20 * chord * density / viscosity).round(), label=sim_name, figsize=(3, 3), dpi=80
            )
        # plot_aerodynamics(
        #     fun,
        #     data,
        #     ["CD", "CL", "CY", "Cn", "Cl", "Cm"],
        #     (np.array([4, 6, 8, 10, 12, 14, 16, 18]) * chord * density / viscosity).round(),
        # )
