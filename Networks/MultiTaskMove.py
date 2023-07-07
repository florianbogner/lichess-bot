import torch
import torch.nn as nn
import torch.nn.functional as F
from .map_policy import make_map
import numpy as np
from legal_filtering import get_legal_moves
from policy_indices import policy_index
import pickle

# Config
C_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_HIDDEN_CHANNELS = 64
C_OUTPUT_PLANES = 80
C_OUTCOME_LOSS_FN = 'mse'
C_USE_EVAL_PRED = False
C_EVAL_LOSS_FN = 'mse'
C_RES_BLOCKS = 6
C_IN_CHANNELS = 33
C_PRETRAINED = ""

class SqueezeAndExcite(nn.Module): # https://arxiv.org/abs/1709.01507
    def __init__(self, n_features, reduction=16):
        super(SqueezeAndExcite, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            SqueezeAndExcite(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(torch.add(x, self.layers(x)))

class ApplyPolicyMap(nn.Module):
    def __init__(self, planes=80):
        super(ApplyPolicyMap, self).__init__()
        self.planes = planes
        self.fc1 = torch.Tensor(make_map(planes=planes)).to(C_DEVICE) #5120x1858

    def forward(self, x):
        if not hasattr(self, 'planes'):
            self.planes = 80

        h_conv_pol_flat = torch.reshape(x, [-1, self.planes*8*8])
        return torch.matmul(h_conv_pol_flat.to(C_DEVICE), self.fc1)

class MultiTaskMove(nn.Module):
    def __init__(self):
        super(MultiTaskMove, self).__init__()
  
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=C_IN_CHANNELS, out_channels=C_HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(C_HIDDEN_CHANNELS),
            nn.ReLU()
        )
        self.res_layers = []
        for i in range(C_RES_BLOCKS):
            self.res_layers.append(ResidualBlock(C_HIDDEN_CHANNELS, C_HIDDEN_CHANNELS))
        
        self.res_layers = nn.Sequential(*self.res_layers)

        self.conv_policy = nn.Sequential(
            nn.Conv2d(in_channels=C_HIDDEN_CHANNELS, out_channels=C_HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=C_HIDDEN_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(in_channels=C_HIDDEN_CHANNELS, out_channels=C_OUTPUT_PLANES, kernel_size=3, padding=1),
            nn.Flatten(),
            ApplyPolicyMap(planes=C_OUTPUT_PLANES)
        )

        #if c.outcome_output:
        self.conv_outcome = nn.Sequential(
            nn.Conv2d(in_channels=C_HIDDEN_CHANNELS, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=16*8*8, out_features=4*8*8),
            nn.ReLU(),
            nn.Linear(in_features=4*8*8, out_features=1),
            nn.Tanh()
        )

        if C_USE_EVAL_PRED:
            self.conv_eval = nn.Sequential(
                nn.Conv2d(in_channels=C_HIDDEN_CHANNELS, out_channels=16, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=16*8*8, out_features=4*8*8),
                nn.ReLU(),
                nn.Linear(in_features=4*8*8, out_features=1)
            )

    def predict_legal(self, board_tensor, board_string):
        with torch.no_grad():  
            model_output = self.forward(board_tensor)

        legal_full =  get_legal_moves(board_string).to(C_DEVICE) * model_output

        return policy_index[torch.argmax(legal_full)]

    def predict_legal_distribution(self, board_tensor, board_string):
        with torch.no_grad():  
            model_output = self.forward(board_tensor)

        legal_full =  get_legal_moves(board_string).to(C_DEVICE) * model_output

        return legal_full

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_layers(x)
        policy = self.conv_policy(x)

        result = (policy,)

        outcome = self.conv_outcome(x)
        result = result + (outcome,)
        
        if C_USE_EVAL_PRED:
            eval_out = self.conv_eval(x)
            result = result + (eval_out,)
        
        return result
    
    def load_pretraining(self):
        pretrained = pickle.load(open(C_PRETRAINED, 'rb'))
        self.conv1.load_state_dict(pretrained.conv1.state_dict())
        for param in self.conv1.parameters():
            param.requires_grad = False

        for i in range(len(self.res_layers)):
            self.res_layers[i].load_state_dict(pretrained.res_layers[i].state_dict())
            if i < 5:
                for param in self.res_layers[i].parameters():
                    param.requires_grad = False

        print("Successfully loaded pretrained weights")


class MultiTaskMoveLoss(nn.Module):
    def __init__(self):
        super(MultiTaskMoveLoss, self).__init__()
        if C_OUTCOME_LOSS_FN == 'mse':
            self.outcomeLossFn = nn.MSELoss()
        elif C_OUTCOME_LOSS_FN == 'mae':
            self.outcomeLossFn = nn.L1Loss()
        else:
            self.outcomeLossFn = None
        
        if C_EVAL_LOSS_FN == 'mse':
            self.evalLossFn = nn.MSELoss()
        elif C_EVAL_LOSS_FN == 'mae':
            self.evalLossFn = nn.L1Loss()
        else:
            self.evalLossFn = None
    
    def forward(self, output, target):
        # output: (policy, outcome, eval)
        # target: (policy, outcome, eval)

        loss = (F.cross_entropy(output[0], target[0]),)

        print(len(target))

        if len(target) > 1:
            loss += (self.outcomeLossFn(output[1].squeeze(), target[1]),)
        
        if C_USE_EVAL_PRED and len(target) > 2:
            loss += (self.evalLossFn(output[2].squeeze(), target[2]),)

        return loss