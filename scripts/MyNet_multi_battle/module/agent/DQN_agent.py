import torch.nn as nn
import torch.nn.functional as F
import torch


class RNN_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN_Agent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        
        
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        
        # 定義輸出層
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)

    def forward(self, x):
     
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        q = self.fc3(x)
        return q



