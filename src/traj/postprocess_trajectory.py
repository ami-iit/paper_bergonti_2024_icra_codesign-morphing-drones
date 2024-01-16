import pickle
from traj.trajectory import Postprocess, Goal
from core.visualizer import Visualize_robot
from core.robot import Robot
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def get_trajectory_output(traj_name: str = None):
    if traj_name is None:
        traj_name = max(glob.glob("result/*.p"), key=os.path.getctime)
        print(traj_name)

    s = pickle.load(open(traj_name, "rb"))
    return s["out"], traj_name


if __name__ == "__main__":
    out, traj_name = get_trajectory_output(traj_name=None)

    pp = Postprocess(out)
    pp.save_pics(name_directory_images=traj_name[:-2], figsize=(16, 9))
    pp.print_stats()

    robot = Robot(fullpath_model=out["robot"]["fullpath_model"])
    robot.set_joint_limit(
        ub_joint_pos=out["robot"]["limits"]["joint_pos"]["ub"],
        lb_joint_pos=out["robot"]["limits"]["joint_pos"]["lb"],
        ub_joint_vel=out["robot"]["limits"]["joint_vel"]["ub"],
        ub_joint_acc=out["robot"]["limits"]["joint_acc"]["ub"],
        ub_joint_tor=out["robot"]["limits"]["joint_tor"]["ub"],
    )
    robot.set_propeller_limit(
        ub_thrust=out["robot"]["limits"]["prop_thrust"]["ub"],
        ub_dot_thrust=out["robot"]["limits"]["prop_dot_thrust"]["ub"],
    )

    viz = Visualize_robot(robot)
    viz.set_goals(out["constant"]["goals"])
    viz.set_obstacles(out["constant"]["obstacles"])
    viz.run(
        t=out["time"],
        base_state=np.concatenate((out["optivar"]["pos_w_b"], out["optivar"]["quat_w_b"])),
        joint_values=out["optivar"]["s"],
        prop_values=out["optivar"]["thrust"],
        aero_values=out["optivar"]["vcat_wrenchAero_w"],
        base_twist_values=out["optivar"]["twist_w_b"],
    )

    pp.base()
    pp.base_3d()
    pp.thrust()
    pp.joint()
    pp.centroidal()
    pp.com_bodies()
    pp.frame_bodies()
    pp.wind()
    pp.wind_body()
    pp.show()

    viz.close_rviz()
