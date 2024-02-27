import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_validation(name_csv_database: str, savefig: bool = False, plots_icra_video: bool = False) -> None:
    df = pd.read_csv(name_csv_database)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    list_drones = ["bix3", "opt1", "opt2", "opt3", "opt4"]
    list_color = ["tab:orange", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:green", "tab:green"]
    list_color = ["#D62728", "#FF7F0E", "#CBBF5F", "#15B7C3", "#2CA02C"]

    for drone in list_drones:
        if drone not in df["name_drone"].values:
            list_drones.remove(drone)

    df["energy_normalized"] = df["energy"] / (2 * df["distance"])
    df["time_normalized"] = df["time"] / (2 * df["distance"])
    df["drone_id"] = df["name_drone"].apply(lambda x: list_drones.index(x))

    time_decrease = (
        100
        - (
            df.groupby(["drone_id"])["time_normalized"].mean()[1:]
            / df.groupby(["drone_id"])["time_normalized"].mean()[0]
        ).values
        * 100
    )
    energy_decrease = (
        100
        - (
            df.groupby(["drone_id"])["energy_normalized"].mean()[1:]
            / df.groupby(["drone_id"])["energy_normalized"].mean()[0]
        ).values
        * 100
    )
    print(f"time decrease: {time_decrease.min():.0f} - {time_decrease.max():.0f}")
    print(f"energy decrease: {energy_decrease.min():.0f} - {energy_decrease.max():.0f}")

    # boxplot energy
    if plots_icra_video:
        plt.figure(figsize=(4.5, 4), dpi=600)
    else:
        plt.figure(figsize=(2.8, 2))
    sns.violinplot(
        x="drone_id", y="energy_normalized", data=df, inner=None, linewidth=0, saturation=0.4, palette=list_color, cut=0
    )
    sns.boxplot(
        x="drone_id",
        y="energy_normalized",
        data=df,
        width=0.3,
        boxprops={"zorder": 2},
        showfliers=False,
        palette=list_color,
    )
    plt.gca().set(xlabel=None)
    plt.ylabel("[J/m]")
    if plots_icra_video:
        plt.title("energy consumption (normalised)")
    plt.xticks(np.arange(len(list_drones)), list_drones, rotation=0)
    plt.ylim(0, 6.5)
    plt.tight_layout()
    plt.grid(axis="y", color="0.9")
    plt.gca().set_axisbelow(True)
    if savefig:
        plt.savefig("boxplot_energy_normalized.png", bbox_inches="tight")
        plt.savefig("boxplot_energy_normalized.pdf", bbox_inches="tight")
    # boxplot time
    # boxplot energy
    if plots_icra_video:
        plt.figure(figsize=(4.5, 4), dpi=600)
    else:
        plt.figure(figsize=(2.8, 2))
    sns.violinplot(
        x="drone_id", y="time_normalized", data=df, inner=None, linewidth=0, saturation=0.4, palette=list_color, cut=0
    )
    sns.boxplot(
        x="drone_id",
        y="time_normalized",
        data=df,
        width=0.3,
        boxprops={"zorder": 2},
        showfliers=False,
        palette=list_color,
    )
    plt.gca().set(xlabel=None)
    plt.ylabel("[s/m]")
    if plots_icra_video:
        plt.title("mission completion time (normalised)")
    plt.xticks(np.arange(len(list_drones)), list_drones, rotation=0)
    plt.ylim(0.06, 0.155)
    plt.tight_layout()
    plt.grid(axis="y", color="0.9")
    plt.gca().set_axisbelow(True)
    if savefig:
        plt.savefig("boxplot_time_normalized.png", bbox_inches="tight")
        plt.savefig("boxplot_time_normalized.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Script for plotting the results of `run_validation.py`
    # If you leave the code unchanged, it will plot the results from the paper (figure 9).
    # If you want to plot your own results, change the path to the CSV file.
    plot_validation("result/mt_2024-02-22_10h31m37s.csv", True, False)
