import os, sys, argparse

import numpy as np
import torch

from ppo.a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from ppo.a2c_ppo_acktr.utils import get_vec_normalize


# Workaround to unpickle olf model files
import ppo.a2c_ppo_acktr
sys.modules['a2c_ppo_acktr'] = ppo.a2c_ppo_acktr
sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='ScratchItchJaco-v0',
                    help='environment to train on (default: ScratchItchJaco-v0)')
parser.add_argument('--load-dir', default='./trained_models/ppo/',
                    help='directory to save agent logs (default: ./trained_models/ppo/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--test-type', type=int, default=0,
                        help='Test with different types of noise (default: 0)')
parser.add_argument('--test-rounds', type=int, default=1,
                        help='Test rounds (default: 0)')
parser.add_argument('--starting-rate', type=float, default=0.0,
                        help='Test noise increasing rate (default: 0.1)')
parser.add_argument('--ir', type=float, default=0.2,
                        help='Test noise increasing rate (default: 0.w)')
args = parser.parse_args()

args.det = not args.non_det

records = []
starting_rate = args.starting_rate
for test_round in range(args.test_rounds):
    noise_scale = args.starting_rate+args.ir*test_round

    env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None,
                        args.add_timestep, device='cpu', allow_early_resets=False)

    # Determine the observation lengths for the robot and human, respectively
    obs = env.reset()
    action = torch.tensor([env.action_space.sample()])
    _, _, _, info = env.step(action)
    obs_robot_len = info[0]['obs_robot_len']
    obs_human_len = info[0]['obs_human_len']
    obs_robot = obs[:, :obs_robot_len]
    obs_human = obs[:, obs_robot_len:]
    if len(obs_robot[0]) != obs_robot_len or len(obs_human[0]) != obs_human_len:
        print('robot obs shape:', obs_robot.shape, 'obs space robot shape:', (obs_robot_len,))
        print('human obs shape:', obs_human.shape, 'obs space human shape:', (obs_human_len,))
        exit()

    env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None,
                        args.add_timestep, device='cpu', allow_early_resets=False)

    # We need to use the same statistics for normalization as used in training
    actor_critic_robot, actor_critic_human, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    recurrent_hidden_states_robot = torch.zeros(1, actor_critic_robot.recurrent_hidden_state_size)
    recurrent_hidden_states_human = torch.zeros(1, actor_critic_human.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    # Reset environment
    obs = env.reset()
    obs_robot = obs[:, :obs_robot_len]
    obs_human = obs[:, obs_robot_len:]

    iteration = 0
    rewards = []
    forces = []
    task_successes = []
    for iteration in range(100):
        done = False
        reward_total = 0.0
        force_total = 0.0
        force_list = []
        task_success = 0.0
        while not done:
            with torch.no_grad():
                value_robot, action_robot, _, recurrent_hidden_states_robot = actor_critic_robot.act(
                    obs_robot, recurrent_hidden_states_robot, masks, deterministic=args.det)
                value_human, action_human, _, recurrent_hidden_states_human = actor_critic_human.act(
                    obs_human, recurrent_hidden_states_human, masks, deterministic=args.det)
            iteration += 1

            # Obser reward and next obs
            action = torch.cat((action_robot, action_human), dim=-1)
            if args.test_type == 2:
                action += float(np.random.normal(scale=noise_scale, size=len(action)))
            obs, reward, done, info = env.step(action)
            if args.test_type == 1:
                obs += float(np.random.normal(scale=noise_scale, size=len(obs)))
            obs_robot = obs[:, :obs_robot_len]
            obs_human = obs[:, obs_robot_len:]
            reward = reward.numpy()[0, 0]
            reward_total += reward
            force_list.append(info[0]['total_force_on_human'])
            task_success = info[0]['task_success']

            masks.fill_(0.0 if done else 1.0)

        rewards.append(reward_total)
        forces.append(np.mean(force_list))
        task_successes.append(task_success)
        print('Reward total:', reward_total, 'Mean force:', np.mean(force_list), 'Task success:', task_success)
        sys.stdout.flush()

    records.append((test_round, noise_scale, rewards, forces, task_successes))


# print('Rewards:', rewards)
# print('Reward Mean:', np.mean(rewards))
# print('Reward Std:', np.std(rewards))

# print('Forces:', forces)
# print('Force Mean:', np.mean(forces))
# print('Force Std:', np.std(forces))

# print('Task Successes:', task_successes)
# print('Task Success Mean:', np.mean(task_successes))
# print('Task Success Std:', np.std(task_successes))

# print(records)
for record in records:
    test_round, noise_scale, rewards, forces, task_successes = record
    print("Test round: ", test_round, "Noise Scale: ", noise_scale)
    print('Rewards: ', np.mean(rewards), np.std(rewards))
    print('Forces: ', np.mean(forces), np.std(forces)) 
    print('Task Successes: ', np.mean(task_successes), np.std(task_successes)) 
sys.stdout.flush()

