import numpy as np
import torch
import itertools


class RunningMean:
    def __init__(self, window_size):
        self.count = 0
        self.data = []
        self.mean = None

    def insert(self, d):
        if self.count > 1:
            self.data = self.data[1:]
        self.count += 1
        self.data.append(d)
        self.mean = sum(self.data) / self.count


class Runner:
    def __init__(self, net, env, device='cpu', decimal=False):
        self.net = net
        self.env = env
        self.decimal = decimal

        self.device = torch.device(device)

        a_list = [[1, -1], [-1, 1], [0, 0]]
        a_pool = list(itertools.product(a_list, a_list))
        for i in range(len(a_pool)):
            a_pool[i] = np.concatenate(a_pool[i])
        self.a_pool = a_pool

        self.score = 0

        self.state = self.env.reset()

        self.mean = RunningMean(10)

    def run(self, steps, eps=0.0):
        b_s, b_a, b_r, b_s_, b_d = [], [], [], [], []
        for _ in range(steps):
            s = self.state
            s_t = torch.tensor(s).unsqueeze(0).to(self.device)
            q = self.net(s_t.float())
            ac_cat = torch.argmax(q).cpu().numpy()
            # e-greedy
            if np.random.random() < eps:
                ac_cat = np.random.randint(len(self.a_pool))
            a = self.a_pool[ac_cat]
            s_, r, d, _ = self.env.step(a)

            b_s.append(s)
            b_a.append(ac_cat)
            b_r.append([r])
            b_s_.append(s_)
            b_d.append([d])

            self.score += r
            self.state = s_

            if d:
                self.mean.insert(self.score)
                self.score = 0
                self.state = self.env.reset()
                break

        if self.decimal:
            num_type = np.float32
        else:
            num_type = np.uint8
        b_s = np.asarray(b_s, dtype=num_type)
        b_a = np.asarray(b_a, dtype=num_type)
        b_r = np.asarray(b_r, dtype=num_type)
        b_s_ = np.asarray(b_s_, dtype=num_type)
        b_d = np.asarray(b_d, dtype=num_type)

        return b_s, b_a, b_r, b_s_, b_d
