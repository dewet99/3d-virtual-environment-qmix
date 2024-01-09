import math
import os
from typing import Dict, List, Tuple

# import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
import torch.optim as optim
from IPython.display import clear_output
import torch.autograd as autograd 
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.constant_(self.sigma_weight, self.sigma_init)
      init.constant_(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)
    