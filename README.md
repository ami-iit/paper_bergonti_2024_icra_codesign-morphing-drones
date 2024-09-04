<h1 align="center">
Co-Design Optimisation of Morphing Topology and Control of Winged Drones
</h1>


<div align="center">


_F. Bergonti, G. Nava, V. Wüest, A. Paolino, G. L'Erario, D. Pucci, D. Floreano "Co-Design Optimisation of Morphing Topology and Control of Winged Drones" in 
2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan, May 2024, pp. 8679-8685, doi: 10.1109/ICRA57147.2024.10611506.

</div>

<p align="center">

https://github.com/ami-iit/paper_bergonti_2024_icra_codesign-morphing-drones/assets/38210073/9303a1b1-40c2-44cc-9146-23eb557068d6

</p>

<div align="center">
  <a href="#installation"><b>Installation</b></a> |
  <a href="https://ieeexplore.ieee.org/abstract/document/10611506"><b>Paper</b></a> | 
  <a href="https://arxiv.org/abs/2309.13948"><b>arXiv</b></a> |
  <a href="https://youtu.be/uWYuQ8gT404"><b>Video</b></a>
</div>


## Abstract

The design and control of winged aircraft and drones is an iterative process aimed at identifying a compromise of mission-specific costs and constraints. When agility is required, shape-shifting (morphing) drones represent an efficient solution. However, morphing drones require the addition of actuated joints that increase the topology and control coupling, making the design process more complex. We propose a co-design optimisation method that assists the engineers by proposing a morphing drone’s conceptual design that includes topology, actuation, morphing strategy, and controller parameters. The method consists of applying multi-objective constraint-based optimisation to a multi-body winged drone with trajectory optimisation to solve the motion intelligence problem under diverse flight mission requirements, such as energy consumption and mission completion time. We show that co-designed morphing drones outperform fixed-winged drones in terms of energy efficiency and mission time, suggesting that the proposed co-design method could be a useful addition to the aircraft engineering toolbox.

## Installation

A quick way to install the dependencies is via [conda package manager](https://docs.conda.io) which provides binary packages for Linux, macOS and Windows of the software contained in the robotology-superbuild. Relying on the community-maintained [`conda-forge`](https://conda-forge.org/) channel and also the `robotology` conda channel.

Please refer to [the documentation in `robotology-superbuild`](https://github.com/robotology/robotology-superbuild/blob/master/doc/conda-forge.md) to install and configure a conda distribution. Then, once your environment is set, you can run the following command to install the required dependencies.

1. Clone the repository:
    ```sh
    git clone https://github.com/ami-iit/paper_bergonti_2024_icra_codesign-morphing-drones.git
    ```
2. Install conda dependencies:
    ```sh
    cd paper_bergonti_2024_icra_codesign-morphing-drones
    mamba env create -n <conda-environment-name> --file environment.yml
    mamba activate  <conda-environment-name>
    ```
3. Specify the number of threads used by the optimiser:
   ```sh
   mamba env config vars set OMP_NUM_THREADS=1
   ```
4. Build ROS packages:
    ```sh
    cd src
    catkin_init_workspace
    cd ..
    catkin_make
    echo "source $(pwd)/devel/setup.sh" > "${CONDA_PREFIX}/etc/conda/activate.d/rosmuav_activate.sh"
    chmod +x "${CONDA_PREFIX}/etc/conda/activate.d/rosmuav_activate.sh"
    ```
5. Install python repository:
    ```sh
    pip install --no-deps -e .
    ```

> [!WARNING]
> When you activate the conda environment, the ROS environment is automatically sourced. If you want to deactivate the ROS environment, you should open a new terminal.

> [!WARNING]
> Note that to replicate the paper results, you need to install the HSL solvers (here we use `ma27`), which can be downloaded but not redistributed. Please check [here](https://licences.stfc.ac.uk/product/coin-hsl). Once you have downloaded and configured the solver, you have to modify [this line](https://github.com/ami-iit/paper_bergonti_2024_icra_codesign-morphing-drones/blob/main/src/traj/trajectory.py#L51) and set `ma27`.

> [!NOTE]
> The installation procedure has been tested on `Ubuntu 22.04` in a `WSL2` environment. Windows is not supported due to the lack of support for the `ros-noetic-jsk-rviz-plugins` package as of October 2023.

## Usage

The results of the paper can be reproduced by running the following scripts:
- [`run_codesign.py`](src/run_codesign.py)
- [`run_validation.py`](src/run_validation.py)

The running time is approximately ~15.7 hours for [`run_codesign.py`](src/run_codesign.py) and ~1.7 hours for [`run_validation.py`](src/run_validation.py) on a machine with two AMD EPYC 7513 CPUs, utilizing 100 cores.

The figures from the paper can be reproduced by running the following scripts:
- [`plot_codesign.py`](src/plot_codesign.py)
- [`plot_validation.py`](src/plot_validation.py)


https://github.com/user-attachments/assets/48d7fd2a-a4b7-4d10-b4ca-d680dae1e999

## Citing this work

If you find the work useful, please consider citing:

```bibtex
@inproceedings{bergonti2024co,
  title={Co-design optimisation of morphing topology and control of winged drones},
  author={Bergonti, Fabio and Nava, Gabriele and W{\"u}est, Valentin and Paolino, Antonello and L’Erario, Giuseppe and Pucci, Daniele and Floreano, Dario},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8679--8685},
  year={2024},
  organization={IEEE}
}
```

### Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="https://github.com/FabioBergonti.png" width="40">](https://github.com/FabioBergonti) | [@FabioBergonti](https://github.com/FabioBergonti) |

<p align="left">
   <a href="https://github.com/ami-iit/paper_bergonti_2022_tro_kinematics-control-morphingcovers/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ami-iit/paper_bergonti_2022_tro_kinematics-control-morphingcovers" alt="Size" class="center"/></a>
</p>
