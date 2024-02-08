<h1 align="center">
Co-Design Optimisation of Morphing Topology and Control of Winged Drones
</h1>


<div align="center">


_F. Bergonti, G. Nava, V. Wüest, A. Paolino, G. L'Erario, D. Pucci, D. Floreano "Co-Design Optimisation of Morphing Topology and Control of Winged Drones" in 
TODO, vol. TODO, no. TODO, pp. TODO-TODO, mm YYYY, doi: TODO_

</div>

<p align="center">

Video TODO

</p>

<div align="center">
  TODO
</div>

<div align="center">
  <a href="#installation"><b>Installation</b></a> |
  <a href="TODO"><b>Paper</b></a> |
  <a href="https://arxiv.org/abs/2309.13948"><b>arXiv</b></a> |
  <a href="TODO"><b>Video</b></a>
</div>

## Abstract

The design and control of winged aircraft and drones is an iterative process aimed at identifying a compromise of mission-specific costs and constraints. When agility is required, shape-shifting (morphing) drones represent an efficient solution. However, morphing drones require the addition of actuated joints that increase the topology and control coupling, making the design process more complex. We propose a co-design optimisation method that assists the engineers by proposing a morphing drone's conceptual design that includes topology, actuation, morphing strategy, and controller parameters. The method consists of applying multi-objective constraint-based optimisation to a multi-body winged drone with trajectory optimisation to solve the motion intelligence problem under diverse flight mission requirements. We show that co-designed morphing drones outperform fixed-winged drones in terms of energy efficiency and agility, suggesting that the proposed co-design method could be a useful addition to the aircraft engineering toolbox.

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
    mamba create --name <conda-environment-name> --file environment.txt
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
> Note that to replicate the paper results, you need to install the HSL solvers (here we use `ma27`), which can be downloaded but not redistributed. Please check [here](https://licences.stfc.ac.uk/product/coin-hsl). Once you have downloaded and configured the solver, you have to modify [this line](TODO) and set `ma27`.

> [!NOTE]
> The installation procedure has been tested on `Ubuntu 22.04` in a `WSL2` environment. Windows is not supported due to the lack of support for the `ros-noetic-jsk-rviz-plugins` package as of October 2023.

## Usage

The results of the paper can be reproduced by running the following scripts:
- [`run_codesign.py`](src/run_codesign.py)
- [`run_validation.py`](src/run_validation.py)

The running time is approximately ~31.9 hours and ~4.2 hours for [`run_codesign.py`](src/run_codesign.py) and [`run_validation.py`](src/run_validation.py), respectively, on a PC with an Intel Xeon Silver 4214 CPU (48 cores).

The figures from the paper can be reproduced by running the following scripts:
- [`plot_codesign.py`](src/plot_codesign.py)
- [`plot_validation.py`](src/plot_validation.py)

## Citing this work

If you find the work useful, please consider citing:

```bibtex
@article{bergonti2023co,
  title={Co-Design Optimisation of Morphing Topology and Control of Winged Drones},
  author={Bergonti, Fabio and Nava, Gabriele and W{\"u}est, Valentin and Paolino, Antonello and L'Erario, Giuseppe and Pucci, Daniele and Floreano, Dario},
  journal={arXiv preprint arXiv:2309.13948},
  year={2023}
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
