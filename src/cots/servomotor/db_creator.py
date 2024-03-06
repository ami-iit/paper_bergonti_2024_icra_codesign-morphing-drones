import pandas as pd
import matplotlib.pyplot as plt
import utils_muav
from dataclasses import dataclass
import numpy as np
from typing import List
import math
from matplotlib import pyplot as plt


@dataclass
class Servomotor:
    name: str
    mass__g: float  # [g]
    dimension__mm: List  # [mm] [x,y,z]
    stall_torque__Nm: float  # [Nm]
    stall_current__A: float  # [A]
    stall_voltage__V: float  # [V]
    no_load_speed__rpm: float  # [rpm]
    no_load_voltage__V: float  # [V]
    range_of_motion__deg: float  # [deg]
    url: str

    def __compute_values(self):
        self._name = self.name
        self._mass = self.mass__g / 1000
        d = np.array(self.dimension__mm).mean() / 1000
        I = 1 / 6 * self._mass * d**2
        self._inertia_xx = I
        self._inertia_yy = I
        self._inertia_zz = I
        self._resistance = self.stall_voltage__V / self.stall_current__A
        self._motor_velocity_constant = self.no_load_speed__rpm / self.no_load_voltage__V * (2 * math.pi / 60)
        self._motor_torque_constant = self.stall_torque__Nm / self.stall_current__A
        self._torque_limit = self.stall_torque__Nm * 0.5
        self._speed_limit = self.no_load_speed__rpm * (2 * math.pi / 60) * 0.5
        self._lb_position = -np.deg2rad(self.range_of_motion__deg / 2)
        self._ub_position = +np.deg2rad(self.range_of_motion__deg / 2)
        self._viscous_friction__Nms = (0.1 * self._torque_limit) / self._speed_limit
        self._url = self.url

    def write_to_db(self, name_csv: str) -> None:
        self.__compute_values()

        with open(name_csv, "r") as f:
            id = f.read().count("\n") - 1
        string = (
            ""
            + f"{id},"
            + f"{self._name},"
            + f"{self._mass},"
            + f"{self._inertia_xx},{self._inertia_yy},{self._inertia_zz},"
            + f"{self._resistance},"
            + f"{self._motor_velocity_constant},"
            + f"{self._motor_torque_constant},"
            + f"{self._torque_limit},"
            + f"{self._speed_limit},"
            + f"{self._lb_position},{self._ub_position},"
            + f"{self._viscous_friction__Nms},"
            + f"{self._url}"
        )
        with open(name_csv, "a") as f:
            f.write(f"{string}\n")


def create_empty_db(name_csv: str) -> None:
    string = (
        ""
        + "id,"
        + "name,"
        + "mass,"
        + "inertia_xx,inertia_yy,inertia_zz,"
        + "resistance,"
        + "motor_velocity_constant,"
        + "motor_torque_constant,"
        + "torque_limit,"
        + "speed_limit,"
        + "lb_position,ub_position,"
        + "viscous_friction,"
        + "url"
    )
    with open(name_csv, "w") as f:
        f.write(f"{string}\n")


if __name__ == "__main__":
    name_csv = f'{utils_muav.get_repository_tree()["database_servomotor"]}/db_servomotor.csv'
    create_empty_db(name_csv)

    Servomotor(
        name="XL330-M077-T",
        mass__g=18,
        dimension__mm=[20, 34, 26],
        stall_torque__Nm=0.215,
        stall_current__A=1.47,
        stall_voltage__V=5,
        no_load_speed__rpm=383,
        no_load_voltage__V=5,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XL330-M288-T",
        mass__g=18,
        dimension__mm=[20, 34, 26],
        stall_torque__Nm=0.52,
        stall_current__A=1.47,
        stall_voltage__V=5,
        no_load_speed__rpm=103,
        no_load_voltage__V=5,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XC330-M181-T",
        mass__g=23,
        dimension__mm=[20, 34, 26],
        stall_torque__Nm=0.6,
        stall_current__A=1.8,
        stall_voltage__V=5,
        no_load_speed__rpm=129,
        no_load_voltage__V=5,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xc330-m181/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XC330-T288-T",
        mass__g=23,
        dimension__mm=[20, 34, 26],
        stall_torque__Nm=0.92,
        stall_current__A=0.8,
        stall_voltage__V=11.1,
        no_load_speed__rpm=65,
        no_load_voltage__V=11.1,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XC330-M288-T",
        mass__g=23,
        dimension__mm=[20, 34, 26],
        stall_torque__Nm=0.93,
        stall_current__A=1.8,
        stall_voltage__V=5,
        no_load_speed__rpm=81,
        no_load_voltage__V=5,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XL430-W250",
        mass__g=57.2,
        dimension__mm=[28.5, 46.5, 34],
        stall_torque__Nm=1.4,
        stall_current__A=1.4,
        stall_voltage__V=11.1,
        no_load_speed__rpm=57,
        no_load_voltage__V=11.1,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XC430-W240",
        mass__g=65,
        dimension__mm=[28.5, 46.5, 34],
        stall_torque__Nm=1.9,
        stall_current__A=1.4,
        stall_voltage__V=12,
        no_load_speed__rpm=70,
        no_load_voltage__V=12,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xc430-w240/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XW430-T200-R",
        mass__g=96,
        dimension__mm=[28.5, 46.5, 34],
        stall_torque__Nm=2.3,
        stall_current__A=1.3,
        stall_voltage__V=12,
        no_load_speed__rpm=53,
        no_load_voltage__V=12,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xw430-t200/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XM430-W210",
        mass__g=82,
        dimension__mm=[28.5, 46.5, 34],
        stall_torque__Nm=3,
        stall_current__A=2.3,
        stall_voltage__V=12,
        no_load_speed__rpm=77,
        no_load_voltage__V=12,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xm430-w210/",
    ).write_to_db(name_csv)

    Servomotor(
        name="MX-28R/T",
        mass__g=72,
        dimension__mm=[35.6, 50.6, 35.5],
        stall_torque__Nm=2.5,
        stall_current__A=1.4,
        stall_voltage__V=12,
        no_load_speed__rpm=55,
        no_load_voltage__V=12,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/mx/mx-28-2/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XH430-W350",
        mass__g=82,
        dimension__mm=[28.5, 46.5, 34],
        stall_torque__Nm=3.4,
        stall_current__A=1.3,
        stall_voltage__V=12,
        no_load_speed__rpm=30,
        no_load_voltage__V=12,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xh430-w350/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XD430-T350-R",
        mass__g=85,
        dimension__mm=[28.5, 46.5, 34],
        stall_torque__Nm=3.4,
        stall_current__A=1.3,
        stall_voltage__V=12,
        no_load_speed__rpm=30,
        no_load_voltage__V=12,
        range_of_motion__deg=360,
        url="",
    ).write_to_db(name_csv)

    Servomotor(
        name="XH540-W150-T/R",
        mass__g=165,
        dimension__mm=[33.5, 58.5, 44],
        stall_torque__Nm=7.1,
        stall_current__A=4.9,
        stall_voltage__V=12,
        no_load_speed__rpm=70,
        no_load_voltage__V=12,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xh540-w150/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XW540-T260-R",
        mass__g=185,
        dimension__mm=[33.5, 58.5, 45.9],
        stall_torque__Nm=9.5,
        stall_current__A=4.9,
        stall_voltage__V=12,
        no_load_speed__rpm=40,
        no_load_voltage__V=12,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xw540-t260/",
    ).write_to_db(name_csv)

    Servomotor(
        name="XD540-T270-R",
        mass__g=170,
        dimension__mm=[33.5, 58.5, 44],
        stall_torque__Nm=9.9,
        stall_current__A=4.9,
        stall_voltage__V=12,
        no_load_speed__rpm=39,
        no_load_voltage__V=12,
        range_of_motion__deg=360,
        url="https://emanual.robotis.com/docs/en/dxl/x/xd540-t270/",
    ).write_to_db(name_csv)

    df = pd.read_csv(f'{utils_muav.get_repository_tree()["database_servomotor"]}/db_servomotor.csv')

    plt.figure()
    for index, row in df.iterrows():
        T = np.linspace(0, row["torque_limit"], int(1e3))
        w = row["speed_limit"]
        R = row["resistance"]
        kV = row["motor_velocity_constant"]
        kI = row["motor_torque_constant"]
        P = w * T / (kV * kI) + R * kV / kI * T**2
        plt.plot(T, P)
    plt.ylabel("P [W]")
    plt.xlabel("T [Nm]")

    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    for index, row in df.iterrows():
        plt.plot(row["mass"], row["torque_limit"], "o")
    plt.ylabel("T [Nm]")
    plt.xlabel("m [kg]")
    plt.grid()
    plt.subplot(2, 1, 2)
    for index, row in df.iterrows():
        T = row["torque_limit"]
        w = row["speed_limit"]
        R = row["resistance"]
        kV = row["motor_velocity_constant"]
        kI = row["motor_torque_constant"]
        P = w * T / (kV * kI) + R * kV / kI * T**2
        plt.plot(row["mass"], P, "o")
    plt.ylabel("P [W]")
    plt.xlabel("m [kg]")
    plt.grid()
    plt.show()
