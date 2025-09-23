from gymnasium.core import RenderFrame
from gymnasium.envs.registration import EnvCreator, EnvSpec, register
from Maps import Map
from typing import Any, Optional, Set, SupportsFloat, Tuple, Dict, Union
from StaticObjects import *
from MobileObjects import *
import gymnasium as gym
from enum import Enum


class BaseActions(Enum):
    right=0
    down=1
    left=2
    up=3


class BaseEnvironment(gym.Env):
    _map: Map
    _robot: RobotStepper
    _lidar: Lidar
    _window: Optional[Surface]
    _fps: Union[int, np.integer]
    metadata = {"render_modes":["human", "rgb_array", "pygame"], "render_fps":120}
    
    def __init__(self, wl: np.ndarray, robot_dims: np.ndarray , obstacles: Union[List,Set,None] = None, 
    grid_resolution: Union[np.integer,int]= 1000, num_rays: Union[np.integer,int] = 360 ,
    max_range: Union[np.floating,float] = 5.0, render_mode: Optional[str] = "pygame"):
        """ Create the Base Environment with all the necessary things"""
        assert not render_mode or render_mode in self.metadata["render_modes"], f"The render mode must be None, empty or in the list {self.metadata['render_modes']}"
        assert grid_resolution > 0, f"The resolution of the grid, number of divisions in a meter, must be more than 0"
        assert wl.shape == (2,) and wl[0] > 0 and wl[1] >0, f"The width and height must be greater than 0 and it must come in a (2,) shaped numpy ndarray"
        self._current_pos = np.array([0,0])
        
        self._map = Map(wl,grid_resolution,obstacles)
        self._robot = RobotStepper(self._current_pos, robot_dims, Color(255,0,0))
        self._lidar = Lidar(self._current_pos, num_rays, max_range)

        limits = np.linalg.norm(wl)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-limits,limits,(4,))

        self._action_to_direction = {
            BaseActions.right.value: np.array([1, 0]),
            BaseActions.up.value: np.array([0, 1]),
            BaseActions.left.value: np.array([-1, 0]),
            BaseActions.down.value: np.array([0, -1]),
        }
        
        self._window = None
        self._fps = self.metadata['render_fps']

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        assert action in self._action_to_direction, f"The action {action} is not supported, keep it in {self._action_to_direction.keys()}"
        direction = self._action_to_direction[action]
        return super().step(action)
    def _get_observation(self):
        """ 
        Here is where the scanning is made after moving the robot
        and the data return to the agent is prepared
        """
        pass
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Any, Dict[str, Any]]:
        return super().reset(seed=seed, options=options)
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()
    def close(self):
        return super().close()

register(id="emptyEnv",
         entry_point="Environment:BaseEnvironment",
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

if __name__ == '__main__':
    env = gym.make("emptyEnv")
    print("END")
    
