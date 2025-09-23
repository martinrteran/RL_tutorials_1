import torch
import time
import gymnasium as gym
import numpy as np

data = gym.spaces.MultiDiscrete(np.array([[1,0],[0,1],[-1,0],[0,-1]]))
print("WAIT")