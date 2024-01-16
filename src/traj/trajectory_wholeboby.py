from core.robot import Robot
import casadi as cs
import numpy as np
from core.visualizer import Visualize_robot
from liecasadi import SE3, SO3, SO3Tangent
import math
from typing import Dict
import utils_muav
import time
from traj.trajectory import (
    Trajectory,
    Goal,
    Obstacle_Sphere,
    Obstacle_InfiniteCylinder,
    Obstacle_Plane,
    Postprocess,
    Task,
    Gains_Trajectory,
)


class Trajectory_WholeBody_Planner(Trajectory):
    def __init__(
        self, robot: Robot, knots: int, time_horizon: float = None, regularize_control_input_variations: bool = False
    ) -> None:
        super().__init__(robot, knots, time_horizon)
        self._regularize_control_input_variations = regularize_control_input_variations

    def _get_optivar(self):
        # robot
        self.optivar["s"] = 1 * self.opti.variable(self.robot.ndofs, self.N + 1)
        self.optivar["dot_s"] = 5 * self.opti.variable(self.robot.ndofs, self.N + 1)
        self.optivar["ddot_s"] = 20 * self.opti.variable(self.robot.ndofs, self.N + 1)
        self.optivar["torque"] = 1 * self.opti.variable(self.robot.ndofs, self.N + 1)
        self.optivar["thrust"] = 2 * self.opti.variable(self.robot.nprop, self.N + 1)
        self.optivar["dot_torque"] = self.opti.variable(
            self.robot.ndofs * self._regularize_control_input_variations, self.N + 1
        )
        self.optivar["dot_thrust"] = self.opti.variable(
            self.robot.nprop * self._regularize_control_input_variations, self.N + 1
        )
        self.optivar["pos_w_b"] = 20 * self.opti.variable(3, self.N + 1)
        self.optivar["quat_w_b"] = 1 * self.opti.variable(4, self.N + 1)
        self.optivar["twist_w_b"] = cs.vertcat(
            20 * self.opti.variable(3, self.N + 1), 2 * self.opti.variable(3, self.N + 1)
        )
        self.optivar["dot_twist_w_b"] = cs.vertcat(
            20 * self.opti.variable(3, self.N + 1), 2 * self.opti.variable(3, self.N + 1)
        )
        # wind
        self.optivar["vcat_wrenchAero_w"] = self.opti.variable(self.robot.naero * 6, self.N + 1)
        self.optivar["vcat_alpha"] = self.opti.variable(self.robot.naero * 1, self.N + 1)
        self.optivar["vcat_beta"] = self.opti.variable(self.robot.naero * 1, self.N + 1)
        self.optivar["vcat_vinf_norm"] = self.opti.variable(self.robot.naero * 1, self.N + 1)

    def create(self):
        super().create()
        self.opti = cs.Opti()
        super()._set_solver_stgs()

        # optivar
        self._get_optivar()
        self._get_optivar_time()
        #
        self._set_initial_guess()

        # CONSTRAINTS
        # feasibility
        self._set_feasibility_constraint_quaternion(tol=1e-3, index_range=1)  # norm2==1
        # box limit
        self._set_box_constraint_joint_pos()
        self._set_box_constraint_joint_vel()
        self._set_box_constraint_joint_tor()
        self._set_box_constraint_propeller_thrust()
        self._set_box_constraint_joint_acc()
        if self._regularize_control_input_variations:
            self._set_box_constraint_joint_dot_tor()
            self._set_box_constraint_propeller_dot_thrust()
        self._set_box_constraint_aerodynamic_angles()
        # self._set_box_constraint_robot_base_rpy()
        self._set_box_constraint_robot_base_vel()
        self._set_box_constraint_robot_base_acc()
        # initial
        self._set_constraint_initial_condition("pos_w_b", tol=1e-3)
        self._set_constraint_initial_condition_quat("quat_w_b", tol=1e-2 * math.pi / 180)
        self._set_constraint_initial_condition("twist_w_b", tol=1e-3)
        if self.robot.ndofs > 0:
            self._set_constraint_initial_condition("s", tol=1e-3)
            self._set_constraint_initial_condition("dot_s", tol=1e-3)
            self._set_constraint_initial_condition("ddot_s", tol=1e-3)
        # goal & obstacles
        self._set_goals_condition()
        self._set_constraint_obstacles_avoidance()
        # integration
        self._integrate_robot()
        # wind
        self._set_constraint_aerodynamic_forces()
        # centroidal
        self._set_constraint_robot_dynamics()
        # time
        self._set_constraint_time_increment()

        # COST FUNCTION
        self._minimize_time_horizon(gain=self.constant["gains"].cost_function_weight_time)
        self._minimize_energy_consumption(gain=self.constant["gains"].cost_function_weight_energy)
        self.opti.minimize(self.cost)

    def set_initial_condition(
        self,
        s: np.ndarray,
        dot_s: np.ndarray,
        ddot_s: np.ndarray,
        pos_w_b: np.ndarray,
        quat_w_b: np.ndarray,
        twist_w_b: np.ndarray,
    ) -> None:
        self.initial_condition = {}
        self.initial_condition["s"] = s
        self.initial_condition["dot_s"] = dot_s
        self.initial_condition["ddot_s"] = ddot_s
        self.initial_condition["pos_w_b"] = pos_w_b
        self.initial_condition["quat_w_b"] = quat_w_b
        self.initial_condition["twist_w_b"] = twist_w_b

    def _set_initial_guess(self):
        pos_w_b = np.zeros((3, self.N + 1))
        quat_w_b = np.zeros((4, self.N + 1))
        twist_w_b = np.zeros((6, self.N + 1))
        dot_twist_w_b = np.zeros((6, self.N + 1))
        s = np.zeros((self.robot.ndofs, self.N + 1))
        dot_s = np.zeros((self.robot.ndofs, self.N + 1))
        ddot_s = np.zeros((self.robot.ndofs, self.N + 1))
        torque = np.zeros((self.robot.ndofs, self.N + 1))
        thrust = np.zeros((self.robot.nprop, self.N + 1))
        vcat_wrenchAero_w = np.zeros((self.robot.naero * 6, self.N + 1))
        vcat_alpha = np.zeros((self.robot.naero * 1, self.N + 1))
        vcat_beta = np.zeros((self.robot.naero * 1, self.N + 1))
        vcat_vinf_norm = np.zeros((self.robot.naero * 1, self.N + 1))

        if self.time_horizon is None:
            dt = (self.constant["dt"]["lb"] + self.constant["dt"]["ub"]) / 2
        else:
            dt = self.time_horizon / self.N

        array_k = np.array(0)
        matrix_pos_w_b = np.matrix(self.initial_condition["pos_w_b"]).T
        for goal in self.constant["goals"]:
            if bool(goal.position):
                array_k = np.append(array_k, goal.k)
                matrix_pos_w_b = np.hstack((matrix_pos_w_b, np.matrix(goal.position["value"]).T))
        indexes = np.argsort(array_k)
        array_k = array_k[indexes]
        matrix_pos_w_b = matrix_pos_w_b[:, indexes]

        for i, aero_frame in enumerate(self.robot.aero_frame_list):
            alpha, beta, Vinf_norm = self.robot.get_aerodynamic_angles_and_speed_fun(aero_frame)(
                SE3(pos=self.initial_condition["pos_w_b"], xyzw=self.initial_condition["quat_w_b"]).as_matrix(),
                self.initial_condition["twist_w_b"],
                self.initial_condition["s"],
                self.initial_condition["dot_s"],
                self.constant["vel_w_wind"],
            )
            vcat_alpha[i, :] = alpha
            vcat_beta[i, :] = beta
            vcat_vinf_norm[i, :] = Vinf_norm
            aero_wrench_w = self.robot.get_aerodynamic_wrench_fun(aero_frame)(
                SE3(pos=self.initial_condition["pos_w_b"], xyzw=self.initial_condition["quat_w_b"]).as_matrix(),
                self.initial_condition["s"],
                self.constant["air_density"],
                self.constant["air_viscosity"],
                alpha,
                beta,
                Vinf_norm,
            )
            vcat_wrenchAero_w[6 * i : 6 * i + 6, :] = aero_wrench_w

        static_torque = self.robot.get_joint_torque_static_fun()(
            cs.vcat((self.initial_condition["pos_w_b"], self.initial_condition["quat_w_b"])),
            self.initial_condition["s"],
            thrust[:, 0],
            vcat_wrenchAero_w[:, 0],
        ).full()

        for k in range(self.N + 1):
            pos_w_b[0, k] = np.interp(x=np.array([k]), xp=array_k, fp=np.asarray(matrix_pos_w_b[0, :]).reshape(-1))
            pos_w_b[1, k] = np.interp(x=np.array([k]), xp=array_k, fp=np.asarray(matrix_pos_w_b[1, :]).reshape(-1))
            pos_w_b[2, k] = np.interp(x=np.array([k]), xp=array_k, fp=np.asarray(matrix_pos_w_b[2, :]).reshape(-1))
            quat_w_b[:, k] = [0, 0, 0, 1]
            twist_w_b[:, k] = [0, 0, 0, 0, 0, 0]
            dot_twist_w_b[:, k] = self.robot.kinDyn.g
            s[:, k] = self.initial_condition["s"]
            dot_s[:, k] = self.initial_condition["dot_s"]
            ddot_s[:, k] = 0
            thrust[:, k] = 0
            torque[:, k] = static_torque.ravel()

        self.opti.set_initial(self.optivar["dt"], dt)
        self.opti.set_initial(self.optivar["pos_w_b"], pos_w_b)
        self.opti.set_initial(self.optivar["quat_w_b"], quat_w_b)
        self.opti.set_initial(self.optivar["twist_w_b"], twist_w_b)
        self.opti.set_initial(self.optivar["dot_twist_w_b"], dot_twist_w_b)
        self.opti.set_initial(self.optivar["vcat_wrenchAero_w"], vcat_wrenchAero_w)
        self.opti.set_initial(self.optivar["vcat_alpha"], vcat_alpha)
        self.opti.set_initial(self.optivar["vcat_beta"], vcat_beta)
        self.opti.set_initial(self.optivar["vcat_vinf_norm"], vcat_vinf_norm)

        if self.robot.ndofs > 0:
            self.opti.set_initial(self.optivar["s"], s)
            self.opti.set_initial(self.optivar["dot_s"], dot_s)
            self.opti.set_initial(self.optivar["ddot_s"], ddot_s)
            self.opti.set_initial(self.optivar["torque"], torque)
            if self._regularize_control_input_variations:
                self.opti.set_initial(self.optivar["dot_torque"], 0)
        if self.robot.nprop > 0:
            self.opti.set_initial(self.optivar["thrust"], thrust)
            if self._regularize_control_input_variations:
                self.opti.set_initial(self.optivar["dot_thrust"], 0)

    def _set_box_constraint_joint_tor(self):
        if self.robot.ndofs > 0:
            for k in range(self.N + 1):
                self._bound(
                    self.robot.limits["joint_tor"]["lb"],
                    self.optivar["torque"][:, k],
                    self.robot.limits["joint_tor"]["ub"],
                )

    def _set_box_constraint_joint_dot_tor(self):
        if self.robot.ndofs > 0:
            for k in range(self.N + 1):
                self._bound(
                    self.robot.limits["joint_dot_tor"]["lb"],
                    self.optivar["dot_torque"][:, k],
                    self.robot.limits["joint_dot_tor"]["ub"],
                )

    def _integrate_robot(self):
        s = self.optivar["s"]
        ds = self.optivar["dot_s"]
        dds = self.optivar["ddot_s"]
        thrust = self.optivar["thrust"]
        torque = self.optivar["torque"]
        dot_torque = self.optivar["dot_torque"]
        dot_thrust = self.optivar["dot_thrust"]
        p = self.optivar["pos_w_b"]
        q = self.optivar["quat_w_b"]
        w = self.optivar["twist_w_b"][3:, :]
        v = self.optivar["twist_w_b"][:3, :]
        V = self.optivar["twist_w_b"]
        dV = self.optivar["dot_twist_w_b"]

        for k in range(self.N):
            dt = self.vector_dt[k]
            # s
            if self.robot.ndofs > 0:
                s_next = s[:, k] + dt * ds[:, k + 1]  # EI
                self.opti.subject_to(s[:, k + 1] == s_next)
                ds_next = ds[:, k] + dt * dds[:, k + 1]  # EI
                self.opti.subject_to(ds[:, k + 1] == ds_next)
                if self._regularize_control_input_variations:
                    torque_next = torque[:, k] + dt * dot_torque[:, k + 1]  # EI
                    self.opti.subject_to(torque[:, k + 1] == torque_next)
            # thrust
            if self.robot.nprop > 0:
                if self._regularize_control_input_variations:
                    thrust_next = thrust[:, k] + dt * dot_thrust[:, k + 1]  # EI
                    self.opti.subject_to(thrust[:, k + 1] == thrust_next)
            # quat
            q_next = (SO3Tangent(w[:, k + 1] * dt) + SO3(q[:, k])).as_quat().coeffs()  # EI
            self.opti.subject_to(q[:, k + 1] == q_next)
            # pos
            p_next = p[:, k] + v[:, k + 1] * dt
            self.opti.subject_to(p[:, k + 1] == p_next)
            # twist
            V_next = V[:, k] + dt * dV[:, k + 1]
            self.opti.subject_to(V[:, k + 1] == V_next)

    def _set_constraint_robot_dynamics(self):
        mass_matrix_fun = self.robot.kinDyn.mass_matrix_fun()
        bias_force_fun = self.robot.kinDyn.bias_force_fun()
        gravity_term_fun = self.robot.kinDyn.gravity_term_fun()
        B = cs.vertcat(cs.MX(6, self.robot.ndofs), cs.MX.eye(self.robot.ndofs))
        vcat_aero_jacobian_fun = [
            self.robot.kinDyn.jacobian_fun(aero_frame) for aero_frame in self.robot.aero_frame_list
        ]
        vcat_prop_jacobian_fun = [
            self.robot.kinDyn.jacobian_fun(prop_frame) for prop_frame in self.robot.propellers_frame_list
        ]
        vcat_prop_forward_kinematics_fun = [
            self.robot.kinDyn.forward_kinematics_fun(prop_frame) for prop_frame in self.robot.propellers_frame_list
        ]
        for k in range(self.N + 1):
            tform_w_b = self._get_tform_w_b_symbolic(k)
            twist_w_b = self.optivar["twist_w_b"][:, k]
            dot_twist_w_b = self.optivar["dot_twist_w_b"][:, k]
            s = self.optivar["s"][:, k]
            ds = self.optivar["dot_s"][:, k]
            dds = self.optivar["ddot_s"][:, k]
            torque = self.optivar["torque"][:, k]
            thrust = self.optivar["thrust"][:, k]
            vcat_wrenchAero_w = self.optivar["vcat_wrenchAero_w"][:, k]
            nu = cs.vertcat(twist_w_b, ds)
            dnu = cs.vertcat(dot_twist_w_b, dds)
            M = mass_matrix_fun(tform_w_b, s)
            h = bias_force_fun(tform_w_b, s, twist_w_b, ds)
            G = gravity_term_fun(tform_w_b, s)

            sum_Jt_wrenchAero_w = 0
            for i, aero_frame in enumerate(self.robot.aero_frame_list):
                jacobianAero_w = vcat_aero_jacobian_fun[i](tform_w_b, s)
                wrenchAero_w = vcat_wrenchAero_w[i * 6 : i * 6 + 6]
                sum_Jt_wrenchAero_w += jacobianAero_w.T @ wrenchAero_w
            sum_Jt_wrenchPro_w = 0
            for i, prop_frame in enumerate(self.robot.propellers_frame_list):
                jacobianProp_w = vcat_prop_jacobian_fun[i](tform_w_b, s)
                rotm_w_prop = vcat_prop_forward_kinematics_fun[i](tform_w_b, s)[:3, :3]
                wrenchProp_prop = cs.vertcat(cs.MX(2, 1), thrust[i], cs.MX(3, 1))
                wrenchProp_w = cs.diagcat(rotm_w_prop, rotm_w_prop) @ wrenchProp_prop
                sum_Jt_wrenchPro_w += jacobianProp_w.T @ wrenchProp_w

            self.opti.subject_to(M @ dnu + h - sum_Jt_wrenchAero_w - sum_Jt_wrenchPro_w - B @ torque == 0)

    def _minimize_energy_consumption(self, gain: int = 1):
        get_global_power_consumption_fun = self.robot.get_global_power_consumption_fun()
        energy = 0
        for k in range(self.N):
            thrust = self.optivar["thrust"][:, k]
            torque = self.optivar["torque"][:, k]
            ds = self.optivar["dot_s"][:, k]
            dt = self.vector_dt[k]
            energy += get_global_power_consumption_fun(thrust, torque, ds) * dt
        self._add_cost(gain * energy)

    def solve(self):
        t0 = time.time()
        sol = self.opti.solve()
        out = self._get_empty_out()
        # nlp
        out["nlp_output"]["cost_function"] = sol.value(self.opti.f)
        out["nlp_output"]["iter_count"] = sol.stats()["iter_count"]
        out["nlp_output"]["success"] = sol.stats()["success"]
        out["nlp_output"]["return_status"] = sol.stats()["return_status"]
        # time
        out["dt"] = sol.value(self.vector_dt)
        out["time"] = np.cumsum(np.concatenate(([0], out["dt"])))
        out["computational_time"] = time.time() - t0
        # robot
        out["optivar"]["s"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["s"]))
        out["optivar"]["dot_s"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["dot_s"]))
        out["optivar"]["ddot_s"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["ddot_s"]))
        out["optivar"]["torque"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["torque"]))
        out["optivar"]["thrust"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["thrust"]))
        out["optivar"]["dot_torque"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["dot_torque"]))
        out["optivar"]["dot_thrust"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["dot_thrust"]))
        out["optivar"]["pos_w_b"] = sol.value(self.optivar["pos_w_b"])
        out["optivar"]["quat_w_b"] = sol.value(self.optivar["quat_w_b"])
        out["optivar"]["twist_w_b"] = sol.value(self.optivar["twist_w_b"])
        out["optivar"]["dot_twist_w_b"] = sol.value(self.optivar["dot_twist_w_b"])
        # wind
        out["optivar"]["vcat_wrenchAero_w"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["vcat_wrenchAero_w"]))
        out["optivar"]["vcat_alpha"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["vcat_alpha"]))
        out["optivar"]["vcat_beta"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["vcat_beta"]))
        out["optivar"]["vcat_vinf_norm"] = utils_muav.get_2d_ndarray(sol.value(self.optivar["vcat_vinf_norm"]))
        return out


if __name__ == "__main__":
    t0 = time.time()
    robot_name = "drone_dst_t1"
    robot = Robot(f"{utils_muav.get_repository_tree(relative_path=True)['urdf']}/{robot_name}")
    robot.set_joint_limit()
    robot.set_propeller_limit()
    task = Task.define_slalom_trajectory(
        distance_start_obstacle=20,
        obstacles_radius=4,
        initial_velocity=[10, 0, 0, 0, 0, 0],
        initial_orientation_rpy_deg=[0, 0, 0],
    )
    traj = Trajectory_WholeBody_Planner(
        robot=robot, knots=task.knots, time_horizon=None, regularize_control_input_variations=True
    )
    traj.set_gains(
        Gains_Trajectory(
            cost_function_weight_time=robot.controller_parameters["weight_time_energy"], cost_function_weight_energy=1
        )
    )
    traj.set_wind_parameters(air_density=1.225, air_viscosity=1.8375e-5, vel_w_wind=np.array([-1, 0, 0]))
    traj.set_initial_condition(
        s=np.zeros(robot.ndofs),
        dot_s=np.zeros(robot.ndofs),
        ddot_s=np.zeros(robot.ndofs),
        pos_w_b=task.ic_position_w_b,
        quat_w_b=task.ic_quat_w_b,
        twist_w_b=task.ic_twist_w_b,
    )
    # traj.add_goal(Goal(index=traj.N).set_position(xyz=[30, 0, 0], isStrict=True, param=0.1))
    # traj.add_obstacle(Obstacle_Sphere(r=3, xyz=[25, 7, 2]))
    # traj.add_obstacle(Obstacle_InfiniteCylinder(r=3.5, xy=[15, 0]))
    [traj.add_goal(goal) for goal in task.list_goals]
    [traj.add_obstacle(obstacle) for obstacle in task.list_obstacles]
    traj.create()

    out = traj.solve()
    traj.save(out)
    print(time.time() - t0)
    viz = Visualize_robot(robot)
    viz.set_goals(traj.constant["goals"])
    viz.set_obstacles(traj.constant["obstacles"])
    viz.run(
        t=out["time"],
        base_state=np.concatenate((out["optivar"]["pos_w_b"], out["optivar"]["quat_w_b"])),
        joint_values=out["optivar"]["s"],
        prop_values=out["optivar"]["thrust"],
        aero_values=out["optivar"]["vcat_wrenchAero_w"],
        base_twist_values=out["optivar"]["twist_w_b"],
    )
    viz.close_rviz()

    Postprocess(out).print_stats()
