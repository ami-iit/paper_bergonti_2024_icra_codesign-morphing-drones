import pandas as pd
import scipy
import matplotlib.pyplot as plt
import utils_muav
from dataclasses import dataclass
import numpy as np


def quadratic_function(x, a, b):
    y = a * x**2 + b * x
    return y


@dataclass
class Propeller:
    name: str
    mass__g: float  # [g]
    url: str

    def __compute_values(self):
        df = pd.read_csv(f'{utils_muav.get_repository_tree()["database_propeller"]}/data/{self.name}.csv')
        thrust = df["Thrust (N)"].values
        power = df["Electrical power (W)"].values
        voltage = df["Voltage (V)"].mean().round(2)
        coeff = scipy.optimize.curve_fit(quadratic_function, xdata=thrust, ydata=power)[0]
        self._name = self.name
        self._mass = self.mass__g / 1000
        self._voltage = voltage
        self._max_thrust = max(thrust)
        self._max_power = max(power)
        self._power_coeff = [0, coeff[1], coeff[0]]
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
            + f"{self._voltage},"
            + f"{self._max_thrust:.3f},"
            + f"{self._max_power:.3f},"
            + f"{self._power_coeff[0]:.3f},{self._power_coeff[1]:.3f},{self._power_coeff[2]:.3f},"
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
        + "voltage,"
        + "max_thrust,"
        + "max_power,"
        + "power_coeff_c0,power_coeff_c1,power_coeff_c2,"
        + "url"
    )
    with open(name_csv, "w") as f:
        f.write(f"{string}\n")


if __name__ == "__main__":
    name_csv = f'{utils_muav.get_repository_tree()["database_propeller"]}/db_propeller.csv'
    create_empty_db(name_csv)

    Propeller(
        name="EMAX Emax 1104 5250KV Babyhawk",
        mass__g=6.8,
        url="https://database.tytorobotics.com/tests/j3qz/emax-emax-1104-5250kv-babyhawk",
    ).write_to_db(name_csv)

    Propeller(
        name="T-motor LF40 2305 T-5143 prop",
        mass__g=29.7,
        url="https://database.rcbenchmark.com/tests/mny/t-motor-lf40-2305-t-5143-prop",
    ).write_to_db(name_csv)

    Propeller(
        name="T-motor LF40 5150 prop",
        mass__g=31.2,
        url="https://database.rcbenchmark.com/tests/eqn/t-motor-lf40-5150-prop",
    ).write_to_db(name_csv)

    Propeller(
        name="T-LF40 2450kv T-5150R propeller",
        mass__g=31,
        url="https://database.rcbenchmark.com/tests/w4x/t-lf40-2450kv-t-5150r-propeller",
    ).write_to_db(name_csv)

    Propeller(
        name="Hypetrain Blaster 2207 2450Kv Gemfan 5040",
        mass__g=39.96,
        url="https://database.rcbenchmark.com/tests/9nr/hypetrain-blaster-2207-2450kv-gemfan-5040",
    ).write_to_db(name_csv)

    Propeller(
        name="Motor EMAX RS2306 with Turnigy Plush 30 A",
        mass__g=38.84,
        url="https://database.rcbenchmark.com/tests/pdg/test-for-motor-emax-rs2306-with-turnigy-plush-30-a",
    ).write_to_db(name_csv)

    df = pd.read_csv(f'{utils_muav.get_repository_tree()["database_propeller"]}/db_propeller.csv')

    plt.figure()
    for index, row in df.iterrows():
        T = np.linspace(0, row["max_thrust"], int(1e3))
        P = row["power_coeff_c2"] * T**2 + row["power_coeff_c1"] * T + row["power_coeff_c0"]
        plt.plot(T, P)
    plt.ylabel("P [W]")
    plt.xlabel("T [N]")

    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    for index, row in df.iterrows():
        plt.plot(row["mass"], row["max_thrust"], "o")
    plt.ylabel("T [N]")
    plt.xlabel("m [kg]")
    plt.grid()
    plt.subplot(2, 1, 2)
    for index, row in df.iterrows():
        plt.plot(row["mass"], row["max_power"], "o")
    plt.ylabel("P [W]")
    plt.xlabel("m [kg]")
    plt.grid()
    plt.show()
