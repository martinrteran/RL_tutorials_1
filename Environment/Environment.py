from itertools import product
from gymnasium.core import RenderFrame
from gymnasium.envs.registration import register
import pygame.locals
from pygame.time import Clock
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
    _fps: Union[int, np.integer, float,np.floating]
    metadata = {"render_modes":["human", "rgb_array", "pygame"], "render_fps":60}
    
    def __init__(self, wl: np.ndarray, robot_dims: np.ndarray , obstacles: Union[List,Set,None] = None, 
    grid_resolution: Union[np.integer,int]= 1000, num_rays: Union[np.integer,int] = 360 ,
    max_range: Union[np.floating,float] = 5.0, render_mode: Optional[str] = None, window_size: Optional[np.ndarray] = None):
        """
        Create the Base Environment with all the necessary things
        
        :param self: Description
        :param wl: Description
        :type wl: np.ndarray
        :param robot_dims: Description
        :type robot_dims: np.ndarray
        :param obstacles: Description
        :type obstacles: Union[List, Set, None]
        :param grid_resolution: Description
        :type grid_resolution: Union[np.integer, int]
        :param num_rays: Description
        :type num_rays: Union[np.integer, int]
        :param max_range: Description
        :type max_range: Union[np.floating, float]
        :param render_mode: Description
        :type render_mode: Optional[str]
        
        """
        assert not render_mode or render_mode in self.metadata["render_modes"], f"The render mode must be None, empty or in the list {self.metadata['render_modes']}"
        assert grid_resolution > 0, f"The resolution of the grid, number of divisions in a meter, must be more than 0"
        assert wl.shape == (2,) and wl[0] > 0 and wl[1] >0, f"The width and height must be greater than 0 and it must come in a (2,) shaped numpy ndarray"
        if render_mode == 'pygame':  assert window_size is not None and window_size.shape == (2,), f"The window size 2D array must exist if a render mode is specified as pygame";  self.render_mode = render_mode ; pygame.init(); self._window_resolution = window_size/wl
                
        if window_size is None: self._window_resolution = None
        self._window = None
        self._window_size = window_size

        self._current_pos = np.array([0,0])
        self._goal_pos = np.array([0, 0])
        
        self._map = Map(wl,grid_resolution,obstacles)
        self._robot = RobotStepper(self._current_pos, robot_dims, Color(255,0,0))
        self._lidar = Lidar(self._current_pos, num_rays, max_range)

        limits = np.linalg.norm(wl)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(np.array([0.0,-np.pi,-1.0,-np.pi],np.float32),np.array([limits,np.pi,limits,np.pi], np.float32),(4,))

        self._action_to_direction = {
            BaseActions.right.value: np.array([1, 0]),
            BaseActions.up.value: np.array([0, 1]),
            BaseActions.left.value: np.array([-1, 0]),
            BaseActions.down.value: np.array([0, -1]),
        }
        
        self._fps = self.metadata['render_fps']
        self._clock = Clock()

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:# TODO reward function
        """
        Docstring for step
        
        :param self: Description
        :param action: Description
        :type action: int
        :return: Description
        :rtype: Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]
        """
        assert action in self._action_to_direction, f"The action {action} is not supported, keep it in {self._action_to_direction.keys()}"
        direction = self._action_to_direction[action]
        self._robot.move(direction)
        done = False
        truncated = False
        info = {}
        if self._collision(self._robot.get_pos(),self._robot.get_wl()):
            self._robot.go_back()
            done = True
        else:
            self._lidar.move(direction)
            self._current_pos = self._robot.get_pos()
        
        obs = self._get_observation()
        info = self._get_info()

        if self._reached_goal():
            done = True
            gym.logger.warn("GOAL REACHED")

        reward = 0
        return obs, reward, done, truncated, info
   

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Docstring for reset
        
        :param self: Description
        :param seed: Description
        :type seed: int | None
        :param options: Description
        :type options: Dict[str, Any] | None
        :return: Description
        :rtype: Tuple[Any, Dict[str, Any]]
        """
        np.random.seed(seed)
        super().reset(seed=seed, options=options)
        safety_distance = self._robot.get_wl().max()
        start_pos = None
        while start_pos is None:
            sp = np.random.random((2,))*self._map._wl
            if not self._collision(sp,1.5*safety_distance):
                start_pos = sp
        goal_pos = None
        while goal_pos is None:
            gp = np.random.random((2,))*self._map._wl
            if np.linalg.norm(start_pos-gp) >= 2*safety_distance and not self._collision(gp,1.5*safety_distance):
                goal_pos = gp
        
        self._current_pos = start_pos
        self._goal_pos = goal_pos
        self._robot.reset(start_pos)
        self._lidar.reset(start_pos)
        
        obs = self._get_observation()
        info = self._get_info()

        self.render()
        
        return obs, info
    
    def render(self) -> RenderFrame | list[RenderFrame] | None: # TODO
        """
        Docstring for render
        
        :param self: Description
        :return: Description
        :rtype: RenderFrame | list[RenderFrame] | None
        """
        if self.render_mode == 'pygame':
            if not self._window:
                self._window = pygame.display.set_mode((self._window_size[0],self._window_size[1]), pygame.RESIZABLE) # type: ignore
            
            self._resize_pygame()
            self._render_pygame()
            pygame.display.flip()
            self._clock.tick(self._fps) # type: ignore
            # # Convert pygame surface to NumPy array
            # frame = pygame.surfarray.array3d(self._window)
            # frame = np.transpose(frame, (1, 0, 2))  # Transpose to (height, width, channels)
        return None

    def close(self): # TODO
        """
        Docstring for close
        
        :param self: Description
        """
        if self.render_mode == 'pygame':
            pygame.quit()
        return None
     
    def _reached_goal(self):
        """
        Return True or False depending if the goal has been reached within 1 meter
        
        :param self: the instance itself
        :type self: RobotStepper

        :return reached_goal (bool): Return True if goal has been reached
        """
        dist = np.linalg.norm(self._current_pos - self._goal_pos)
        return dist < 5

    def _collision(self, obj_pos, wl):
        """
        return True if a collision was detected, in other words, the robot hit an obstacle or the wall
        
        :param self: Description
        """
        wl2 = wl/2
        p0, p1 = obj_pos-wl2, obj_pos+wl2
        Xs = np.linspace(p0[0], p1[0], int((p1[0]-p0[0])*self._map.get_grid_resolution()))
        Ys = np.linspace(p0[1], p1[1], int((p1[1]-p0[1])*self._map.get_grid_resolution()))
        return any(not self._map.is_free(np.array([x, y])) for x, y in product(Xs,Ys))

    def _get_info(self):
        """
        Docstring for _get_info
        
        :param self: Description
        """
        lidar_pos = self._current_pos
        goal_pos = self._goal_pos
        lidar_hit = self._lidar.get_min_hit()
        dno = lidar_hit[0:2] - lidar_pos if lidar_hit is not None else np.empty((2,))
        top5 = self._lidar.get_top_hit(5)
            
        return {
            "distance_to_goal": (goal_pos - lidar_pos).astype(np.float32),
            "distance_to_nearest_obstacle": dno.astype(np.float32)
        }
    
    def _get_observation(self):
        """ 
        Here is where the scanning is made after moving the robot
        and the data return to the agent is prepared

        Return
        ------
        :return distance_to_goal (float | np.floating): Distance in meters to the goal
        :return angle_to_goal (float | np.floating): Angle in radians to the goal
        :return distance_to_nearest_obstacle (float | np.floating): Distance in meters to the nearest obstacle or negative if not
        :return angle_to_nearest_obstacle (float | np.floating): Angle in radians to the the nearest obstacle or negative if not
        """
        self._lidar.scan(self._map)
        lidar_hit = self._lidar.get_min_hit()
        distance_to_nearest_obstacle, angle_to_nearest_obstacle  = lidar_hit[2: 4] if lidar_hit is not None else (-1, 0)

        vec_goal = self._goal_pos - self._current_pos
        distance_to_goal = np.linalg.norm(vec_goal)
        angle_to_goal = np.atan2(vec_goal[0],vec_goal[1])

        return np.array([distance_to_goal, angle_to_goal, distance_to_nearest_obstacle, angle_to_nearest_obstacle],np.float32)

    def _render_pygame(self):
        if self._window is not None and self._window_resolution is not None:
            self._window.fill((50, 50, 50))
            self._map.render(self._window,self._window_resolution)
            self._robot.render(self._window,self._window_resolution)
            self._lidar.render(self._window, self._window_resolution)
            pygame.draw.circle(self._window,Color(0,255,0),self._goal_pos * self._window_resolution,2.0*self._window_resolution.max()) # type: ignore

    def _resize_pygame(self):
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                self._window = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                window_width, window_height = self._window.get_size()
                MapW, MapH = self._map._wl
                self._window_resolution = np.array([window_width / MapW, window_height / MapH])
            # if event.type == pygame.KEYDOWN:
            #     keys = pygame.key.get_pressed()
            #     if keys[pygame.K_UP]:
            #         gym.logger.warn("UP")
            #     if keys[pygame.K_ESCAPE]:


if "emptyEnv" not in gym.registry:
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

if "wallsEnv" not in gym.registry:
    register(
        id = "wallsEnv",
        entry_point="Environment:BaseEnvironment",
        reward_threshold=100,
        max_episode_steps=500,
        kwargs={
            "wl":np.array([100,100]),
            "robot_dims": np.array([5,5]),
            "obstacles": [Wall(np.zeros((2,)),np.array([100,5]),Color(0,0,0)),Wall(np.zeros((2,)),np.array([5,100]),Color(0,0,0)),
                            Wall(np.array([95,0]),np.array([5,100]),Color(0,0,0)),Wall(np.array([0,95]),np.array([100,5]),Color(0,0,0))],
            "grid_resolution":100,
            "num_rays":360,
            "max_range":10.0
        }
    )

if __name__ == '__main__':
    from datetime import datetime, timezone
    WinW, WinH = 800,800
    env = gym.make("wallsEnv",None,None,render_mode='pygame',window_size=np.array([WinW,WinH]))
    obs, info = env.reset(seed = datetime.now(timezone.utc).microsecond)
    running = 10_000
    done = False
    while running > 0 and not done:
        try:
            if info['distance_to_goal'][0] < -5 :
                act = BaseActions.left.value
            elif info['distance_to_goal'][0] > 5:
                act = BaseActions.right.value
            elif info['distance_to_goal'][1] < -5:
                act = BaseActions.down.value
            elif info['distance_to_goal'][1] > 5:
                act = BaseActions.up.value
            else:
                print("What do i do?")
                break
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         running = False

            #     if event.type == pygame.KEYDOWN:
                    
            #         key = pygame.key.get_pressed()
            #         if key[pygame.K_UP]:
            #             env.step(BaseActions.down.value)
            #         if key[pygame.K_DOWN]:
            #             env.step(BaseActions.up.value)
            #         if key[pygame.K_LEFT]:
            #             env.step(BaseActions.left.value)
            #         if key[pygame.K_RIGHT]:
            #             env.step(BaseActions.right.value)
            #         if key[pygame.K_ESCAPE]:
            #             running = False
            obs, _, done, _, info =env.step(act)
            env.render()
            running -=1
            print(f"Running {running}/10_000",end="\r")
        except:
            print(f"ERROR")
    

    env.close()
    print("END")
    
