from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, PrioritizedSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    InitTracker,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, KLPENPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from gym_env import GymEnvironment
from nn_model import PPOActorSimple, PPOValueSimple, PPOActorLSTM
import rospy
import os
import tensordict


USE_WANDB = False

if USE_WANDB:
    import wandb
    wandb_run = wandb.init(project="RLVTR",
                           config={
                               "learning_rate_actor":1e-5,
                               "batch_size":32,
                               "epochs":8,
                               "gamma":0.99,
                               "lmbda":0.95,
                               "clip":0.3,
                               "loss":0,
                               "hidden_size":1024
                           })


PRETRAINED = False
lr = 5e-6
max_grad_norm = 1.0
frames_per_batch = 512
# For a complete training, bring the number of frames up to 1M
total_frames = 1_000_000
sub_batch_size = 32  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 8  # optimisation steps per batch of data collected
clip_epsilon = (0.3)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-2
loss_type = 0
hidden_size = 512


if USE_WANDB:
    lr = wandb.config.learning_rate_actor
    frames_per_batch = 1024
    # For a complete training, bring the number of frames up to 1M
    total_frames = 1_000_000
    sub_batch_size = wandb.config.batch_size  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = wandb.config.epochs  # optimisation steps per batch of data collected
    gamma = wandb.config.gamma
    lmbda = wandb.config.lmbda
    entropy_eps = 1e-1
    clip_epsilon = wandb.config.clip
    loss_type = wandb.config.loss
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

actor_net = PPOActorSimple(2, hidden_size=hidden_size).float().to(device)
if PRETRAINED:
    actor_net.load_state_dict(torch.load(SAVE_DIR + "actor_net.pt"))

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.low,
        "max": env.action_spec.space.high,
        # "event_dims": 2,
        # "tanh_loc": True
    },
    return_log_prob=True,
    default_interaction_type=tensordict.nn.InteractionType.MEAN
    # we'll need the log-prob for the numerator of the importance weights
)

value_net = PPOValueSimple(2, hidden_size=hidden_size).float().to(device)
if PRETRAINED:
    value_net.load_state_dict(torch.load(SAVE_DIR + "value_net.pt"))


value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))


collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
    reset_when_done=True,
    reset_at_each_iter=True,
    # exploration_type=ExplorationType.MEAN,
)


replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch * 10),
    sampler=SamplerWithoutReplacement(),
    # sampler=PrioritizedSampler(10_000, 0.7, 0.5)
)


advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=value_module,
    # average_gae=True
)

if loss_type == 0:
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=True,
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
else:
    loss_module = KLPENPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        entropy_bonus=True,
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )


optim = torch.optim.Adam(loss_module.parameters(), lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optim, total_frames // frames_per_batch, 0.0
# )


logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.

    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()
            # replay_buffer.update_tensordict_priority(subdata)

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

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

    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    # scheduler.step()
    torch.save(actor_net.state_dict(), SAVE_DIR + "actor_net.pt")
    torch.save(value_net.state_dict(), SAVE_DIR + "value_net.pt")
    # torch.save(scheduler.state_dict(), SAVE_DIR + "scheduler.pt")
