import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import itertools
import math
import logging
from model.DQNSearchAgent import Agent, ReplayBuffer,DQN
from model.Scorer import Scorer
from model.editor import RobertaEditor
from model.config import get_args
import torch.multiprocessing as mp
import warnings
from model.nwp import set_seed
import datetime
from dateutil import tz
from torch.utils.data import DataLoader, DistributedSampler
from utils.helper import plot, sync_initial_weights, get_free_port
from utils.dataset import TSTDataset


os.environ["TOKENIZERS_PARALLELISM"] = "true"
tzone = tz.gettz('America/Edmonton')
warnings.filterwarnings('ignore')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to set up the distributed environment
def setup(rank, world_size):
    # Set the IP address and port of the master node
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


class FP32Scaler(torch.cuda.amp.GradScaler):
    """
    FP32Scaler is for compatability with AMPScaler.
    But it also automatically checks gradient overflow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        def get_grad_norm(parameters, norm_type=2.0):
            parameters = list(parameters)
            device = parameters[0].grad.device
            return torch.norm(torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters if p.grad is not None]),
                              norm_type)

        parameters = itertools.chain(*[group['params'] for group in optimizer.param_groups])
        grad_norm = get_grad_norm(parameters)
        if grad_norm.isnan() or grad_norm.isinf():
            return
        return optimizer.step()

    def update(self):
        return

    def get_scale(self):
        return 1.0

    def unscale_(self, optimizer):
        return


# Function to clean up the distributed environment
def cleanup():
    dist.destroy_process_group()

def train(rank,world_size,args,train_set):
    # Set up the distributed environment
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # Wrap model with DistributedDataParallel
    editor = DDP(RobertaEditor(args,device).to(device),device_ids=[rank])
    scorer = DDP(Scorer(args, editor,device).to(device),device_ids=[rank])

    agent = DDP(Agent(editor, args, device).to(device),device_ids=[rank])
    local_net = DQN(agent.module.state_dim, args.num_actions).to(device)
    model_net = DDP(local_net,device_ids=[rank])
    target_net = DQN(agent.module.state_dim, args.num_actions).to(device)

    if args.fp32:
        scaler = FP32Scaler()
    else:
        scaler = torch.cuda.amp.GradScaler()

    sync_initial_weights(local_net, rank,world_size)

    # Initialize the network
    replay_buffer = ReplayBuffer(args.buffer_size)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    train_sampler = DistributedSampler(train_set)
    train_data = DataLoader(train_set,
                            batch_size=args.bsz,
                            sampler=train_sampler,
                            num_workers=world_size,
                            pin_memory=True)

    optimizer = optim.Adam(model_net.parameters(), lr=args.lr)

    # epsilon-exploration for choosing an action
    epsilons = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    global_step = 0

    for idx,batch_data in enumerate(train_data):
        batch_data=sorted(batch_data, key=lambda x: len(x.split()), reverse=True)

        ref_olds=batch_data
        batch_state_vec, _ = editor.module.state_vec(batch_data)

        all_rewards = []
        losses = []

        state=ref_olds

        # ------------------- train Q network ------------------- #
        max_episode_reward = [0 for _ in range(len(ref_olds))]

        print(f"Rank {rank}, Device {device}, Data:\n{max_episode_reward}")
        for step in range(args.max_steps):

            torch.cuda.empty_cache()

            # synchronize all processes
            torch.cuda.synchronize()

            epsilon = epsilons(idx)
            actions = agent.module.act(state, epsilon)

            ref_news = []
            for idx in range(args.bsz):
                state_words = state[idx].split()
                action=actions[idx]
                intermediate_results = []
                for positions in range(len(state_words)):
                    edited_state = editor([state[idx]], [action], [positions])
                    intermediate_results+=edited_state

                ref_news.append(intermediate_results)

            if step<args.max_steps-1:
                done=False
            else: done=True # meaning when step=4, done=True

            # get reward
            results = [scorer.module.scoring(ref_news[i], [ref_olds[i]], [batch_state_vec[i]])
                                                                 for i in range(len(ref_news))]

            index, ref_new_score, new_style_labels = zip(*results)

            temp_next_state = [ref_news[i][index[i]] for i in range(len(ref_news))]
            reward=list(ref_new_score)

            # ------------------- update states ------------------- #
            # if ref_new_score>ref_old_score and reward> max_episode_reward:
            for i in range(len(ref_news)):
                if reward[i]> max_episode_reward[i]:
                    max_episode_reward[i] = reward[i]
                    state[i] = temp_next_state[i]

        print(f"Rank {rank}, Device {device}, Data:\n{state}")

        # ------------------- update replay buffer ------------------- #
        for i in range(args.bsz):
            replay_buffer.push(state[i], actions[i], max_episode_reward[i], state[i], done)

        # ------------------- update Q network ------------------- #
        if len(replay_buffer) >= args.buffer_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(args.buffer_size)

            actions = torch.asarray(actions).to(device)
            rewards = torch.asarray(rewards).float().to(device)
            dones = torch.asarray(dones).to(device) + 0  # From T/F to be 0/1

            model_net.train()
            target_net.eval()
            q_values = model_net(agent.module.text2emb(states.tolist()))
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = target_net(agent.module.text2emb(next_states.tolist()))
                next_q_value = next_q_values.max(1)[0]

            # Max DQN
            expected_q_value = torch.max(rewards, next_q_value * (1 - dones))

            loss = (q_value - expected_q_value.data).pow(2).mean() # MSE loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss is {}".format(str(loss.item())))
            logging.info("loss is {}".format(str(loss.item())))
            losses.append(loss.item())

            # ------------------- soft update target network ------------------- #
            global_step+=1
            if global_step % args.update_interval == 0:
                for target_param, local_param in zip(target_net.parameters(),
                                                     local_net.parameters()):
                    target_param.data.copy_(args.tau * local_param.data + (1 - args.tau) * target_param.data)
                torch.save(target_net.state_dict(), os.path.join(of_dir, global_step+'target_net.pt'))


        if done:
            all_rewards.append(max_episode_reward)

        if idx % 1 == 0:
            plot(idx, all_rewards, losses)

    cleanup()

if __name__ == "__main__":

    # Initialize the model
    args = get_args()
    set_seed(args.seed)

    of_dir = 'results/' + args.output_dir
    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')
    dst = args.dst

    train_file = '{}_{}_seed={}_{}_{}_{}.txt'. \
        format(timestamp, dst, str(args.seed), args.style_mode, str(args.style_weight), args.direction)

    train_dataset = TSTDataset(args, 'process_train')
    # Create sampler for distributed training

    # Run the training process
    world_size = torch.cuda.device_count()
    # Spawn one process per GPU and start the training
    mp.spawn(train,
             nprocs=world_size,
             args=(world_size,args,train_dataset),
             join=True)

    # Clean up the distributed environment
    cleanup()