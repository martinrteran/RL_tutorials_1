from gymnasium.envs.registration import register
import gymnasium  as gym
import numpy as np
from .StaticObjects import *


# if "emptyEnv" not in gym.registry:
register(id="emptyEnv",
        entry_point="Environments.Environment:BaseEnvironment",
        reward_threshold=100,
        max_episode_steps=500,
        kwargs={
            "wl":np.array([100,100]),
            "robot_dims": np.array([5,5]),
            "obstacles": None,
            "grid_resolution":100,
            "num_rays":360,
            "max_range":10.0
            }
        )

# if "wallsEnv" not in gym.registry:
register(
    id = "wallsEnv",
    entry_point="Environments.Environment:BaseEnvironment",
    max_episode_steps=100,
    reward_threshold=8,
    kwargs={
        "wl":np.array([100,100]),
        "robot_dims": np.array([5,5]),
        "obstacles": [Wall(np.zeros((2,)),np.array([100,5]),Color(0,0,0)),Wall(np.zeros((2,)),np.array([5,100]),Color(0,0,0)),
                        Wall(np.array([95,0]),np.array([5,100]),Color(0,0,0)),Wall(np.array([0,95]),np.array([100,5]),Color(0,0,0))],
        "grid_resolution":100,
        "num_rays":360,
        "max_range":20.0
    }
)