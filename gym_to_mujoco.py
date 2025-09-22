import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R



def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"/home/quad/github/Gym to Mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        lin_vel_scale = config["lin_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            joint_pos = d.qpos[7:]
            joint_vel = d.qvel[6:]

            tau = pd_control(target_dof_pos, joint_pos, kps, np.zeros_like(kds), joint_vel, kds)

            tau_new = np.zeros(12)
            tau_new[0] = -tau[3]
            tau_new[1] = -tau[4]
            tau_new[2] = -tau[5]
            tau_new[3] = -tau[9]
            tau_new[4] = -tau[10]
            tau_new[5] = tau[11]
            tau_new[6] = tau[0]
            tau_new[7] = -tau[1]
            tau_new[8] = -tau[2]
            tau_new[9] = tau[6]
            tau_new[10] = -tau[7]
            tau_new[11] = tau[8]

            d.ctrl[:] = tau

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            # mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)
                   

                quat = d.qpos[3:7]
                quat = quat[[1, 2, 3, 0]]

                cur_lin_vel = d.qvel[0:3]* lin_vel_scale
                cur_ang_vel = d.qvel[3:6]* ang_vel_scale
                grav = get_gravity_orientation(quat)
                joint_pos = d.qpos[7:] * dof_pos_scale
                joint_vel = d.qvel[6:] * dof_vel_scale

                grav[0:2] *= -1
                    
                obs[0:3] = cur_lin_vel  
                obs[3:6] = cur_ang_vel 
                obs[6:9] = grav  
                obs[9:12] = cmd 
                obs[12:24] = joint_pos
                obs[24:36] = joint_vel
                obs[36:48] = action  

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                action = policy(obs_tensor).detach().numpy().squeeze()
                
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            mujoco.mj_step(m, d)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
