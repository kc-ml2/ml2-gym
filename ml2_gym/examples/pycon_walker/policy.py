import torch
import torch.nn as nn
import numpy as np



def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class Encoder(nn.Module):
    def __init__(self, n_ac, n_hidden=256, device='cpu'):
        super(Encoder, self).__init__()
        self.device = torch.device(device)
        in_shape = (4, 60, 80)
        _dummy_in = torch.zeros(in_shape).unsqueeze(0).to(self.device)
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        ).to(self.device)
        _dummy_out = self.body(_dummy_in)
        out_shape = tuple(_dummy_out.shape)
        num_node = np.array(out_shape)[1:].prod()
        self.flat = nn.Sequential(
            nn.Linear(num_node, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_ac),
        ).to(self.device)

        self.body.apply(init_weights)
        self.flat.apply(init_weights)

    def forward(self, x):
        out = self.body(x)
        out = self.flat(out)
        return out


class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_out, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        if type(hidden) != list:
            hidden = [hidden]
        modules = []
        prev_out = n_in
        for n_h in hidden:
            modules.append(nn.Linear(prev_out, n_h))
            modules.append(nn.ReLU())
            prev_out = n_h
        modules.append(nn.Linear(prev_out, n_out))
        self.body = nn.Sequential(*modules).to(self.device)

        self.body.apply(init_weights)

    def forward(self, x):
        return self.body(x)

