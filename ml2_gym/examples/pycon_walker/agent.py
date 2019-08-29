import torch
import torch.nn
import torch.optim as optim
import numpy as np


class Kyhoon:
    def __init__(self, net, target_net, runner, s_shape,
                 device='cpu', optimizer='RMSprop', lr=1e-3, double=True,
                 target_update_step=10, per=True, decimal=False):
        super().__init__()
        self.net = net
        self.target_net = target_net
        self.runner = runner
        self.double = double
        self.target_update_step = target_update_step
        self.per = per

        self.tau = 0.01

        #s_shape = (60, 80, 4)
        if self.per:
            self.rb = PER(s_shape, max_size=int(1e6), alpha=0.4, beta=0.6,
                          decimal=decimal)
        else:
            self.rb = ReplayBuffer(s_shape, max_size=int(1e6), decimal=decimal)
        self.device = torch.device(device)

        optim_class = getattr(optim, optimizer)
        self.optimizer = optim_class(self.net.parameters(), lr=lr)

        # self.state = self.env.reset()

        self.update_target()

        self.target_counter = 0

    def infer(self, state):
        if type(state) != torch.tensor:
            state = torch.tensor(state)
        state = state.to(self.device)
        q = self.net(state)

        return q.detach().numpy()

    def update_target(self):
        #self.target_net.load_state_dict(self.net.state_dict())
        for target_param, param in zip(self.target_net.parameters(),
                                       self.net.parameters()):
            target_param.data.copy_(self.tau*param.data +
                                    (1-self.tau)*target_param.data)


    def learn(self, num_steps=128, epoch=10, mb_size=128, gamma=0.99, eps=0.3):
        self.target_counter += 1
        b_s, b_a, b_r, b_s_, b_d = self.runner.run(num_steps, eps=eps)
        #b_s, b_a, b_r, b_s_, b_d = [], [], [], [], []
        #for _ in range(num_steps):
        #    s_t = torch.tensor(self.state).unsqueeze(0).to(self.device)
        #    a = self.net(s_t).numpy()
        #    s_, r, d, _ = self.env.step(a[0])

        #    b_s.append(self.state)
        #    b_a.append(a)
        #    b_r.append([r])
        #    b_s_.append(s_)
        #    b_d.append([d])

        #    self.state = s_

        buffer_size = self.rb.insert(b_s, b_a, b_r, b_s_, b_d)

        loss_avg = 0
        if buffer_size >= int(1e4):
            for _ in range(epoch):
                mb_s, mb_a, mb_r, mb_s_, mb_d, sample_idx = self.rb.sample(
                    mb_size)
                mb_s = torch.tensor(mb_s).to(self.device)
                mb_a = torch.tensor(mb_a).to(self.device)
                mb_r = torch.tensor(mb_r).to(self.device)
                mb_s_ = torch.tensor(mb_s_).to(self.device)
                mb_d = torch.tensor(mb_d).to(self.device)

                all_q = self.net(mb_s)
                q = torch.gather(all_q, 1, mb_a.long())
                if self.double:
                    max_a = torch.max(all_q, dim=1)[1]
                    max_a = max_a.unsqueeze(1)
                    q_ = torch.gather(self.target_net(mb_s_), 1, max_a.long())
                else:
                    q_ = torch.max(self.target_net(mb_s_), 1)[0]
                td_error = (mb_r + gamma * (1.0 - mb_d) * q_ - q)
                if self.per:
                    self.rb.update(sample_idx, td_error)
                loss = 0.5 * td_error**2
                loss = loss.mean()
                loss_avg += loss.detach().cpu().numpy()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.target_counter >= self.target_update_step:
                self.update_target()
                self.target_counter = 0

        return loss_avg/epoch


class ReplayBuffer:
    def __init__(self, s_shape, max_size=int(1e5), decimal=True):
        s_shape = tuple([max_size] + list(s_shape))
        data_type = np.float32 if decimal else np.uint8
        self.s = np.zeros(s_shape, dtype=data_type)
        self.a = np.zeros((max_size, 1), dtype=data_type)
        self.r = np.zeros((max_size, 1), dtype=data_type)
        self.s_ = np.zeros(s_shape, dtype=data_type)
        self.d = np.zeros((max_size, 1), dtype=data_type)

        self.curr_idx = 0
        self.max_size = max_size

    def insert(self, b_s, b_a, b_r, b_s_, b_d):
        for i in range(b_s.shape[0]):
            s = b_s[i]
            a = b_a[i]
            r = b_r[i]
            s_ = b_s_[i]
            d = b_d[i]
            if self.curr_idx >= self.max_size:
                ins_idx = np.random.randint(self.max_size)
            else:
                ins_idx = self.curr_idx
            self.s[ins_idx] = s
            self.a[ins_idx] = a
            self.r[ins_idx] = r
            self.s_[ins_idx] = s_
            self.d[ins_idx] = d
            if self.curr_idx < self.max_size:
                self.curr_idx += 1
        return self.curr_idx

    def sample(self, num):
        sample_idx = np.random.randint(self.curr_idx, size=num)
        return (self.s[sample_idx], self.a[sample_idx], self.r[sample_idx],
                self.s_[sample_idx], self.d[sample_idx], sample_idx)


class PER:
    def __init__(self, s_shape, max_size=int(1e5), alpha=0.4, beta=0.6,
                 decimal=True):
        s_shape = tuple([max_size] + list(s_shape))
        data_type = np.float32 if decimal else np.uint8
        self.p = np.zeros((max_size, 1), dtype=np.float32)
        self.s = np.zeros(s_shape, dtype=data_type)
        self.a = np.zeros((max_size, 1), dtype=data_type)
        self.r = np.zeros((max_size, 1), dtype=data_type)
        self.s_ = np.zeros(s_shape, dtype=data_type)
        self.d = np.zeros((max_size, 1), dtype=data_type)

        self.curr_idx = 0
        self.max_size = max_size

        self.alpha = alpha
        self.beta = beta

    def insert(self, b_s, b_a, b_r, b_s_, b_d):
        for i in range(b_s.shape[0]):
            s = b_s[i]
            a = b_a[i]
            r = b_r[i]
            s_ = b_s_[i]
            d = b_d[i]
            if self.curr_idx >= self.max_size:
                ins_idx = np.random.randint(self.max_size)
            else:
                ins_idx = self.curr_idx
            self.s[ins_idx] = s
            self.a[ins_idx] = a
            self.r[ins_idx] = r
            self.s_[ins_idx] = s_
            self.d[ins_idx] = d
            if self.curr_idx < self.max_size:
                self.curr_idx += 1
        return self.curr_idx

    def sample(self, num):
        #sample_idx = np.random.randint(self.curr_idx, size=num)
        probs = self.p[:self.curr_idx]**self.alpha
        probs = probs / probs.sum()
        sample = np.random.multinomial(1, probs, size=num)
        sample_idx = np.argmax(sample, axis=1)
        return (self.s[sample_idx], self.a[sample_idx], self.r[sample_idx],
                self.s_[sample_idx], self.d[sample_idx], sample_idx)

    def update(self, idx, td):
        self.p[idx] = td
