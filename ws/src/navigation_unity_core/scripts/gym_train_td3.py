from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.envs import InitTracker, TransformedEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, PrioritizedSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.objectives.utils import SoftUpdate
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import Actor, TanhNormal, ValueOperator, OrnsteinUhlenbeckProcessWrapper, AdditiveGaussianWrapper
from torchrl.objectives import ClipPPOLoss, KLPENPPOLoss, TD3Loss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from gym_env import GymEnvironment
from nn_model import TD3ActorSimple, TD3ValueSimple
import rospy
import os

USE_WANDB = True

if USE_WANDB:
    import wandb
    wandb_run = wandb.init()

PRETRAINED = False
lr_actor = 1e-6
lr_value = 1e-5
frames_per_batch = 1024
# For a complete training, bring the number of frames up to 1M
total_frames = 1_000_000
sub_batch_size = 256  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 1  # optimisation steps per batch of data collected
gamma = 0.995
lmbda = 0.95
entropy_eps = 1e-1
tau = 0.003
annealing = 300
hidden_size = 512


if USE_WANDB:
    lr_actor = wandb.config.learning_rate_actor
    lr_value = wandb.config.learning_rate_critic
    frames_per_batch = 1024
    # For a complete training, bring the number of frames up to 1M
    total_frames = 1_000_000
    sub_batch_size = wandb.config.batch_size  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = wandb.config.epochs  # optimisation steps per batch of data collected
    gamma = wandb.config.gamma
    entropy_eps = 1e-1
    tau = wandb.config.tau
    annealing = wandb.config.annealing
    hidden_size = wandb.config.hidden_size


HOME = os.path.expanduser('~')
SAVE_DIR = HOME + "/.ros/models/"


env = GymEnvironment()

device = env.device

env = TransformedEnv(
    env,
    Compose(
        # normalize observations
        # ObservationNorm(in_keys=["observation"]),
        InitTracker(),
        DoubleToFloat(),
        StepCounter(),
    ),
)
#
# env.transform[0].init_stats(num_iter=3, reduce_dim=0, cat_dim=0)

# print("normalization constant shape:", env.transform[0].loc.shape)

# print("observation_spec:", env.observation_spec)
# print("reward_spec:", env.reward_spec)
# print("input_spec:", env.input_spec)
# print("action_spec (as defined by input_spec):", env.action_spec)

# check_env_specs(env)
#
# rollout = env.rollout(3)
# print("rollout of three steps:", rollout)
# print("Shape of the rollout TensorDict:", rollout.batch_size)

print("------------ ENVIRONMENT CHECK DONE - INITIALIZING NETWORKS -------------")

actor_net = TD3ActorSimple(2, hidden_size=hidden_size).float().to(device)
if PRETRAINED:
    actor_net.load_state_dict(torch.load(SAVE_DIR + "actor_net.pt"))

policy_module = Actor(module=actor_net, in_keys=["observation"], out_keys=["action"]).to(device)

value_net = TD3ValueSimple(2, hidden_size=hidden_size).float().to(device)
if PRETRAINED:
    value_net.load_state_dict(torch.load(SAVE_DIR + "value_net.pt"))


value_module = ValueOperator(
    module=value_net,
    in_keys=["observation", "action"],
    out_keys=["state_action_value"]
).to(device)

#print("Running policy:", policy_module(env.reset()))
#print("Running value:", value_module(env.reset()))

# actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
#     policy_module,
#     annealing_num_steps=300,
#     sigma=0.05
# ).to(device)

actor_model_explore = AdditiveGaussianWrapper(
    policy_module,
    annealing_num_steps=annealing,
    mean=0.0,
    sigma_init=0.1,
    sigma_end=0.02,
    spec=env.action_spec
).to(device)


collector = SyncDataCollector(
    env,
    actor_model_explore,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)


replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(max_size=10_000),
    sampler=PrioritizedSampler(10_000, 0.7, 0.5),
)


# loss_module = ClipPPOLoss(
#     actor_network=policy_module,
#     critic_network=value_module,
#     clip_epsilon=clip_epsilon,
#     entropy_bonus=True,
#     entropy_coef=entropy_eps,
#     # these keys match by default but we set this for completeness
#     critic_coef=1.0,
#     loss_critic_type="smooth_l1",
# )


loss_module = TD3Loss(
    actor_network=policy_module,
    qvalue_network=value_module,
    action_spec=env.action_spec,
)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optim, total_frames // frames_per_batch, 0.0
# )

target_net_updater = SoftUpdate(loss_module, eps=1 - tau)

optimizer_actor = torch.optim.Adam(
    loss_module.actor_network_params.values(True, True), lr=lr_actor, weight_decay=1e-2
)
optimizer_value = torch.optim.Adam(
    loss_module.qvalue_network_params.values(True, True), lr=lr_value, weight_decay=1e-2
)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
# with set_exploration_type(ExplorationType.MEAN):
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_out = loss_module(subdata.to(device))

            # Optimization: backward, grad clipping and optimization step
            loss_actor = loss_out["loss_actor"]
            loss_q = loss_out["loss_qvalue"]

            loss_actor.backward()
            optimizer_actor.step()
            optimizer_actor.zero_grad()

            loss_q.backward()
            optimizer_value.step()
            optimizer_value.zero_grad()

            replay_buffer.update_tensordict_priority(subdata)

        target_net_updater.step()

    actor_model_explore.step(i)

    if i % 1 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        eval_avg_rwrd = 0.0
        eval_cum_rwrd = 0.0
        eval_steps = 0.0
        eval_num = 3
        for eval_it in range(eval_num):
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                env.set_eval(True)
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )

                eval_avg_rwrd += eval_rollout["next", "reward"].mean().item()
                eval_cum_rwrd += eval_rollout["next", "reward"].sum().item()
                eval_steps += eval_rollout["step_count"].float().mean().item()

                env.set_eval(False)
                del eval_rollout

        if USE_WANDB:
            wandb.log({
                "epoch": i,
                "train_avg_reward": tensordict_data["next", "reward"].mean().item(),
                "eval_avg_reward": eval_avg_rwrd/eval_num,
                "train_cum_reward": tensordict_data["next", "reward"].sum().item(),
                "eval_cum_reward": eval_cum_rwrd/eval_num,
                "train_avg_turn": tensordict_data["action"][0].mean(dim=0)[0].item(),
                "train_avg_dist": tensordict_data["action"][0].mean(dim=0)[1].item(),
                "train_std_turn": tensordict_data["action"][0].std(dim=0)[0].item(),
                "train_std_dist": tensordict_data["action"][0].std(dim=0)[1].item(),
                "train_avg_steps": tensordict_data["step_count"].float().mean().item(),
                "eval_avg_steps": eval_steps/eval_num
            })

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    # scheduler.step()
    torch.save(actor_net.state_dict(), SAVE_DIR + "actor_net.pt")
    torch.save(value_net.state_dict(), SAVE_DIR + "value_net.pt")
    # torch.save(scheduler.state_dict(), SAVE_DIR + "scheduler.pt")
