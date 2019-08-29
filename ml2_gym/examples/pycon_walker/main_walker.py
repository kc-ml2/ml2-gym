import os
import json
import argparse
import random
import numpy as np

import torch
import torch.nn as nn

from settings import *
from ml2_gym.utils.common import ArgumentParser, load_model, save_model
from ml2_gym.utils.logger import Logger


def train(args):
    import policy
    import agent
    import runner

    import gym
    import ml2_gym.pycon_walker

    logger = Logger('wt', args=args)
    logger.log("I Won(Tae)-Chu!")

    # testing policy
    env = gym.make('pycon-v0')
    n_obs = len(env.observation_space.high)
    n_ac = 9
    hidden = [256, 512, 256]
    decimal = True
    enc = policy.MLP(n_obs, hidden, n_ac, device=args.device)
    target = policy.MLP(n_obs, hidden, n_ac, device=args.device)
    runner = runner.Runner(enc, env, device=args.device, decimal=decimal)
    agent = agent.Kyhoon(enc, target, runner, (n_obs,),
                         device=args.device, double=True,
                         optimizer=args.optimizer, lr=args.lr,
                         target_update_step=10,
                         decimal=decimal, per=False)

    save_dir = os.path.join('saved_models', args.tag)
    os.makedirs(save_dir, exist_ok=True)

    for update in range(args.n_update):
        loss = agent.learn(num_steps=args.ep_steps, epoch=args.epoch,
                           mb_size=args.mb_size,
                           eps=0.4*(args.n_update-update)/args.n_update)

        if (update+1) % args.log_step == 0:
            info = {
                'update': update,
                'loss': loss,
                'buffer': agent.rb.curr_idx,
                'score': runner.mean.mean
            }
            logger.scalar_summary(info, update)

        if (update+1) % args.save_step == 0:
            save_model(enc, os.path.join(save_dir, 'enc.pth'))
            save_model(target, os.path.join(save_dir, 'target.pth'))


def test(args):
    import policy
    import runner

    import gym
    import ml2_gym.pycon_walker

    # testing policy
    env = gym.make('pycon-v0')
    n_obs = len(env.observation_space.high)
    n_ac = 9
    hidden = [256, 512, 256]
    enc = policy.MLP(n_obs, hidden, n_ac, device=args.device)
    enc.load_state_dict(torch.load('saved_models/enc.pth'))
    runner = runner.Runner(enc, env, device=args.device, decimal=True)
    #target = policy.MLP(n_obs, hidden, n_ac, device=args.device)
    #target.load_state_dict(torch.load('saved_models/target.pth'))
    s = env.reset()

    score = 0

    while True:
        s_t = torch.tensor(s).float().unsqueeze(0)
        q = enc(s_t)
        ac_cat = torch.argmax(q)
        a = runner.a_pool[ac_cat]

        s_, r, d, _ = env.step(a)
        env.render()
        score += r

        s = s_

        if d:
            print(score)
            break

def double(args):
    import numpy as np
    import keyboard

    import gym
    import ml2_gym.pycon_walker

    import policy
    import runner

    env = gym.make('pycon-v2')
    n_obs = len(env.observation_space.high)
    n_ac = 9
    hidden = [256, 512, 256]
    ai_net = load_model('saved_models/AI.pth')
    runner = runner.Runner(ai_net, env, device=args.device, decimal=True)

    s = env.reset(human=True)
    score = 0
    env.render()
    start_flag = True
    while True:
        if keyboard.is_pressed('q'):
            break
        if keyboard.is_pressed('r'):
            s = env.reset(human=True)
            score = 0
            env.render()
            start_flag = True
        if start_flag:
            start_flag = False
            while True:
                if keyboard.is_pressed('space') or keyboard.is_pressed('q'):
                    break
        a = np.zeros(4)
        if keyboard.is_pressed('j'):
            a = a + np.array([0, 0, 1, -1])
        if keyboard.is_pressed('k'):
            a = a + np.array([0, 0, -1, 1])
        if keyboard.is_pressed('d'):
            a = a + np.array([-1, 1, 0, 0])
        if keyboard.is_pressed('f'):
            a = a + np.array([1, -1, 0, 0])
        a_human = a

        s_t = torch.tensor(s).float().unsqueeze(0)
        q = ai_net(s_t)
        ac_cat = torch.argmax(q)
        a_ai = runner.a_pool[ac_cat]

        s, r, d, _ = env.step(a_ai)
        env.step_human(a_human)
        score += r
        env.render()

        #if d:
        #    print(score)
        #    s = env.reset(human=True)
        #    score = 0
        #    env.render()
        #    start_flag = True


def single(args):
    import numpy as np
    import keyboard

    import gym
    import ml2_gym.pycon_walker

    import policy
    import runner

    env = gym.make('pycon-v0')
    if args.video:
        env = gym.wrappers.Monitor(env, './video/',
                                video_callable=lambda episode_id: True,
                                force=True)

    s = env.reset()
    score = 0
    start_flag = True
    env.render()
    while True:
        if keyboard.is_pressed('q'):
            break
        if keyboard.is_pressed('r'):
            s = env.reset()
            score = 0
            start_flag = True
            env.render()
        if start_flag:
            start_flag = False
            while True:
                if keyboard.is_pressed('enter') or keyboard.is_pressed('q'):
                    break
        a = np.zeros(4)
        if keyboard.is_pressed('j'):
            a = a + np.array([0, 0, 1, -1])
        if keyboard.is_pressed('k'):
            a = a + np.array([0, 0, -1, 1])
        if keyboard.is_pressed('d'):
            a = a + np.array([-1, 1, 0, 0])
        if keyboard.is_pressed('f'):
            a = a + np.array([1, -1, 0, 0])
        a_human = a

        s, r, d, _ = env.step(a_human)
        score += r
        env.render()

        if d:
            print(score)
            s = env.reset()
            score = 0
            start_flag = True
            env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I Won(Tae)-Chu!")
    parser.add_argument("--load_config", type=str, default=None)

    parser.add_argument("--tag", type=str, default='test')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--seed", type=int, default=100)

    parser.add_argument_group("logger options")
    parser.add_argument("--verbose", type=int, default=20)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=100)


    parser.add_argument_group("dataloader options")
    # TODO: add options
    parser.add_argument("--video", action='store_true')

    parser.add_argument_group("optimizer options")
    # TODO: add options
    parser.add_argument("--n_update", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default='RMSprop')
    parser.add_argument("--ep_steps", type=int, default=512)
    parser.add_argument("--mb_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=40)

    args = parser.parse_args()
    if args.load_config is not None:
        with open(os.path.join(PROJECT_ROOT, args.load_config)) as config:
            args = ArgumentParser(json.load(config))
    else:
        assert args.tag is not None
        assert args.mode is not None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    globals()[args.mode](args)
