import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv("result/mt_2023-08-02_09h44m14s.csv")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    encoding = {
        "fixed_wing_drone_back": 0,
        "drone_nsga_46295d0_1": 1,
        "drone_nsga_46295d0_2": 2,
        "drone_nsga_46295d0_3": 3,
        "drone_nsga_46295d0_4": 4,
    }
    list_encoding = ["bix3", "opt1", "opt2", "opt3", "opt4"]
    list_color = ["tab:orange", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:green", "tab:green"]
    list_color = ["#D62728", "#FF7F0E", "#CBBF5F", "#15B7C3", "#2CA02C"]

    for i in range(len(list_encoding) - 1, -1, -1):
        if list(encoding.keys())[i] not in df["name_drone"].values:
            list_encoding.pop(i)

    df["energy_normalized"] = df["energy"] / (2 * df["distance"])
    df["time_normalized"] = df["time"] / (2 * df["distance"])
    df["drone_id"] = df["name_drone"].apply(lambda x: encoding[x])

    df = df[df["drone_id"] != -1]

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
    success_rate = (df.groupby("drone_id").sum()["success"] / df.groupby("drone_id").count()["timestamp"] * 100).values
    success_improve = (success_rate[1:] / success_rate[0]) * 100 - 100
    print(f"time decrease: {time_decrease.min():.0f} - {time_decrease.max():.0f}")
    print(f"energy decrease: {energy_decrease.min():.0f} - {energy_decrease.max():.0f}")
    print(f"success rate improve: {success_improve.min():.0f} - {success_improve.max():.0f}")

    # boxplot energy
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
    plt.xticks(np.arange(len(list_encoding)), list_encoding, rotation=0)
    plt.ylim(0, 6.5)
    plt.tight_layout()
    plt.grid(axis="y", color="0.9")
    plt.gca().set_axisbelow(True)
    plt.savefig("boxplot_energy_normalized.png", bbox_inches="tight")
    plt.savefig("boxplot_energy_normalized.pdf", bbox_inches="tight")
    # boxplot time
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
    plt.xticks(np.arange(len(list_encoding)), list_encoding, rotation=0)
    plt.ylim(0.06, 0.155)
    plt.tight_layout()
    plt.grid(axis="y", color="0.9")
    plt.gca().set_axisbelow(True)
    plt.savefig("boxplot_time_normalized.png", bbox_inches="tight")
    plt.savefig("boxplot_time_normalized.pdf", bbox_inches="tight")
    # boxplot success rate
    plt.figure(figsize=(3, 2))
    sucess_rate = (df.groupby("drone_id").sum()["success"] / df.groupby("drone_id").count()["timestamp"] * 100).values
    print(sucess_rate)
    sns.barplot(x=list_encoding, y=sucess_rate, palette=list_color)
    plt.gca().set(xlabel=None)
    plt.ylabel("[%]")
    plt.xticks(np.arange(len(list_encoding)), list_encoding, rotation=40)
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.grid(axis="y", color="0.9")
    plt.gca().set_axisbelow(True)
    plt.savefig("success_rate.png")
    plt.savefig("success_rate.pdf")
    plt.show()
    # plot time per angle
    df_time_per_angle = df.groupby(["angle", "drone_id"])["time_normalized"].mean().unstack()
    plt.figure(figsize=(6, 3))
    plt.plot(df_time_per_angle, marker=".", label=list_encoding)
    plt.grid(color="0.9")
    plt.gca().set_axisbelow(True)
    xticks = df_time_per_angle.index.values
    plt.xticks(xticks, xticks)
    plt.legend(loc="upper left")
    plt.ylabel("time normalized [s/m]")
    plt.xlabel(r"$\gamma$ [deg]")
    plt.tight_layout()
    plt.savefig("time_per_angle.png")
    plt.savefig("time_per_angle.pdf")
    # plot time per distance
    df_time_per_distance = df.groupby(["distance", "drone_id"])["time_normalized"].mean().unstack()
    plt.figure(figsize=(6, 3))
    plt.plot(df_time_per_distance, marker=".", label=list_encoding)
    plt.grid(color="0.9")
    plt.gca().set_axisbelow(True)
    xticks = df_time_per_distance.index.values
    plt.xticks(xticks, xticks)
    plt.legend(loc="upper left")
    plt.ylabel("time normalized [s/m]")
    plt.xlabel(r"$\gamma$ [deg]")
    plt.tight_layout()
    plt.savefig("time_per_distance.png")
    plt.savefig("time_per_distance.pdf")
    # plot success rate per distance
    df_angle_id = (
        df.groupby(["distance", "drone_id"]).count()["time"]
        / df.groupby(["distance", "drone_id"]).count()["timestamp"]
        * 100
    ).unstack()
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(df_angle_id, marker=".", label=list_encoding)
    xticks = df_angle_id.index.values
    plt.xticks(xticks, xticks)
    plt.grid(color="0.9")
    plt.gca().set_axisbelow(True)
    plt.ylabel("success rate [%]")
    plt.xlabel(r"$d$ [m]")
    plt.tight_layout()
    plt.savefig("success_rate_per_distance.png")
    plt.savefig("success_rate_per_distance.pdf")
    # energy per speed
    df_energy_per_speed = df.groupby(["initial_speed_x", "drone_id"])["energy_normalized"].mean().unstack()
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(df_energy_per_speed, marker=".", label=list_encoding)
    xticks = df_energy_per_speed.index.values
    plt.xticks(xticks, xticks)
    plt.grid(color="0.9")
    plt.gca().set_axisbelow(True)
    plt.ylabel("energy normalized [J/m]")
    plt.xlabel(r"$v_x$ [m/s]")
    plt.tight_layout()
    plt.savefig("energy_per_speed.png")
    plt.savefig("energy_per_speed.pdf")
    # heatmap energy AxD
    df_energy_per_AxD = df.groupby(["angle", "distance"])["energy_normalized"].mean().unstack()
    plt.figure(figsize=(5, 5))
    sns.heatmap(df_energy_per_AxD)
    plt.savefig("energy_per_AxD.png")
    for id in encoding.values():
        plt.figure(figsize=(5, 5))
        df_energy_per_AxD = (
            df[df["drone_id"] == id].groupby(["angle", "distance"])["energy_normalized"].mean().unstack()
        )
        # create heatmap setting the range and gradient of the color
        sns.heatmap(df_energy_per_AxD, vmin=0, vmax=df["energy_normalized"].max(), cmap="YlGnBu")
        plt.title(list_encoding[id])
        plt.savefig(f"energy_per_AxD_{list_encoding[id]}.png")
    # heatmap success rate AxD
    df_success_rate_per_AxD = (
        df.groupby(["angle", "distance"]).count()["time"] / df.groupby(["angle", "distance"]).count()["timestamp"] * 100
    ).unstack()
    plt.figure(figsize=(5, 5))
    sns.heatmap(df_success_rate_per_AxD)
    plt.savefig("success_rate_per_AxD.png")
    for id in encoding.values():
        if id >= 0:
            plt.figure(figsize=(5, 5))
            df_success_rate_per_AxD = (
                df[df["drone_id"] == id].groupby(["angle", "distance"]).count()["time"]
                / df[df["drone_id"] == id].groupby(["angle", "distance"]).count()["timestamp"]
                * 100
            ).unstack()
            sns.heatmap(df_success_rate_per_AxD)
            plt.title(list_encoding[id])
            plt.savefig(f"success_rate_per_AxD_{list_encoding[id]}.png")
