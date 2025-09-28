import numpy as np
from Objects import MobileObjects
from typing import  List, Union
import pygame
from pygame.color import Color
from pygame.surface import Surface
from StaticObjects import EllipseObstacle
from Maps import Map

class RobotStepper(MobileObjects):
    """
    Renders the robot as a coloured rectangle on a pygame surface.
    The rectangle is positioned at ``self._pos`` (world coordinates) and
    sized by ``self._wl`` (width/length in metres).  All values are converted
    to pixels using the supplied *resolution* before drawing.
    """
    def __init__(self, spos: np.ndarray, wl: np.ndarray, color: Color, orientation: np.ndarray = np.array([1,0])):
        """
        Initialize a RobotStepper object.

        :param self: The instance being created
        :param spos: Starting position vector [x, y] in world coordinates
        :type spos: np.ndarray with shape (2,)
        :param wl: Width/length dimensions (in factor form) relative to resolution and they must have the same value
        :type wl: np.ndarray with shape (2,)
        :param color: Pygame Color object defining the appearance
        """
        # assert wl.shape == (2,), f"The width and length must be a 2D vector"
        # assert wl[0] == wl[1], f"The width and the length must be equals"
        assert orientation.shape == (2,), f"The direction/orientation must be a 2D vector"
        assert np.sum(np.maximum(orientation > 1,orientation < -1).astype(int))<=1, f"Incorrect input orientation {orientation}"
        super().__init__(spos, wl, color)
        self._direction = orientation
    def render(self, surface: pygame.surface.Surface, resolution: np.ndarray):
        """
        Draw the robot on the given surface.

        Parameters
        ----------
        surface : pygame.surface.Surface
            The Pygame surface onto which the robot is rendered.
        resolution : np.ndarray
            Conversion factor from world metres to screen pixels
            (``[px_per_m_x, px_per_m_y]``).

        Notes
        -----
        * ``self._pos`` – current world position of the robot’s centre.
          Multiplying by *resolution* gives pixel coordinates for the rectangle.
        * ``self._wl`` – width and length of the robot in metres.
          When multiplied by *resolution* it yields the size of the rectangle
          in pixels.
        * The rectangle is drawn with ``pygame.draw.rect`` using the colour
          stored in ``self._color`` (inherited from :class:`MobileObjects`).
        """
        # resolution is the pixel/meter
        self._draw_rectangle(surface, resolution)
    
    def _draw_rectangle(self, surface:Surface, resolution: np.ndarray):
        pos = (self._pos - self._wl/2)*resolution
        wl = self._wl * resolution
        rect = pygame.rect.Rect(pos[0], pos[1], wl[0], wl[1])
        pygame.draw.rect(surface, self._color,rect)

    def _draw_polygon(self, surface: Surface, resolution: np.ndarray):
        corners = self.get_corners()
        corners = corners * resolution
        corners = [tuple(corner) for corner in corners]
        pygame.draw.polygon(surface,self._color,corners)
    
    def get_corners(self):
        pos = self._pos
        half_w, half_l = self._wl/2
        theta = np.pi/4 #np.atan2(self._direction[0],self._direction[1])
        RM = np.array([ [np.cos(theta), -np.sin(theta)],   [np.sin(theta),  np.cos(theta)]      ])
        # Rotate = lambda P,C,theta:  np.array([ [np.cos(theta), -np.sin(theta)],   [np.sin(theta),  np.cos(theta)]      ]) @ (P-C) + C
        local_offsets = np.array([
            [-half_l, -half_w],  # rear-left
            [-half_l,  half_w],  # rear-right
            [ half_l,  half_w],  # front-right
            [ half_l, -half_w],  # front-left
        ])
        return pos + local_offsets @ RM



def cast_ray(args: tuple) -> np.ndarray | None:
    """
    Cast a single LiDAR ray and return the first obstacle hit.

    Parameters
    ----------
    args : tuple
        A 5‑tuple containing:
            theta     (float) – direction of the ray in radians.
            origin    (np.ndarray[int,2]) – grid indices of the sensor.
            grid      (np.ndarray[int])   – occupancy grid (0 = free).
            grid_res  (int)              – number of cells per metre.
            max_range (float)            – maximum sensing distance in metres.

    Returns
    -------
    np.ndarray | None
        If an obstacle is hit, returns a 4‑element array:
            [x, y, distance, theta]
        where ``x`` and ``y`` are world coordinates (metres),
        ``distance`` is the range to the obstacle (metres),
        and ``theta`` is the ray angle.
        Returns ``None`` if no obstacle is found within *max_range*.

    Notes
    -----
    1. The ray advances in integer grid steps (cell‑by‑cell).
       This keeps the function lightweight and thread‑safe.
    2. `grid[y, x] > 0` denotes an occupied cell; all other values are free.
    3. The loop stops as soon as a hit is detected or the ray leaves
       the bounds of *grid*.
    """
    theta, origin, grid, grid_res, max_range = args
    dx, dy = np.cos(theta), np.sin(theta)
    max_r = int(max_range * grid_res)

    for r in range(max_r):
        x = int(origin[0] + r * dx)
        y = int(origin[1] + r * dy)

        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            if grid[y, x] > 0:
                hit_world = np.array([x, y, r], dtype=float) / grid_res
                return np.append(hit_world, [theta])
        else:
            break
    return None


class Lidar(MobileObjects):
    """
    The Lidar class represents a simulated LIDAR sensor used in robotics or reinforcement learning environments. 
    It performs raycasting scans around the robot to detect obstacles within a specified range and returns hit information.
    
    Attributes:
        _num_rays (Union[np.integer,int]): The number of rays that the lidar has. Default is 360.
        _max_range (Union[np.floating,float]): The maximum range that the lidar can detect objects. Default is 5.0.
        _hits (np.ndarray): A numpy array storing information about each hit by the Lidar.
    """
    _num_rays: Union[np.integer,int]
    _max_range: Union[np.floating,float]
    _hits: np.ndarray

    def __init__(self, spos: np.ndarray, num_rays: Union[np.integer,int] = 360, 
    max_range: Union[np.floating,float] = 5.0, color: Color = Color(150, 150, 0)):
        """
        Initialises the Lidar object with its position and other attributes.
        
        Args:
            spos (np.ndarray): The initial position of the lidar in world coordinates.
            num_rays (Union[np.integer,int], optional): The number of rays that the lidar has. Default is 360.
            max_range (Union[np.floating,float], optional): The maximum range that the lidar can detect objects. Default is 5.0.
            color (Color, optional): The color of the Lidar object. Default is Color(150, 150, 0).
        """
        assert num_rays > 0, f"The number of rays must be more than 0"
        super().__init__(spos, np.zeros((2,)),color)
        self._num_rays = num_rays
        self._max_range = max_range
        self._hits = np.empty((0,4))

    def scan_threading(self, map_obj: Map, max_workers = 8):
        """
        Performs a threaded raycasting scan of the environment to detect obstacles.
        
        Args:
            map_obj (Map): The Map object representing the environment.
            max_workers (int, optional): The maximum number of threads to use. Default is 8.
        """
        from concurrent.futures import ThreadPoolExecutor
        origin = (self._pos * map_obj._grid_resolution).astype(int)
        angles = np.linspace(0, 2 * np.pi, self._num_rays, endpoint=False)

        # Prepare arguments for parallel execution
        args = ((theta, origin, map_obj._grid, map_obj._grid_resolution, self._max_range) for theta in angles)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(cast_ray, args))

        # Filter out None results
        hits = [hit for hit in results if hit is not None]

        self._hits = np.array(hits) if hits else np.empty((0,4))
    
    def scan(self, map_obj: Map):
        """
        Performs a sequential raycasting scan of the environment to detect obstacles for comparison purposes.
        
        Args:
            map_obj (Map): The Map object representing the environment.
        """
        hits = []
        if map_obj._obstacles is not None and len(map_obj._obstacles) > 0:
            
            origin = (self._pos * map_obj._grid_resolution).astype(int)

            angles = np.linspace(0, 2*np.pi, self._num_rays, endpoint=False)
            for theta in angles:
                dx = np.cos(theta)
                dy = np.sin(theta)
                for r in range(int(self._max_range * map_obj._grid_resolution)):
                    x = int(origin[0] + r * dx)
                    y = int(origin[1] + r * dy)
                    if 0 <= y < map_obj._grid.shape[0] and 0 <= x < map_obj._grid.shape[1]:
                        if map_obj._grid[y, x] > 0: #if map_obj._grid[x, y] > 0:
                            # Convert grid coordinates back to world coordinates (meters)
                            hit_world = np.array([x, y, r]) / map_obj._grid_resolution
                            hit_world = np.append(hit_world, [theta])
                            hits.append(hit_world)
                            break
                    else:
                        break  # out of bounds
        self._hits = np.array(hits) if hits else np.empty((0,4))

    def get_min_hit(self):
        """
        Returns information about the closest hit by the Lidar, if there are any hits. 
        Otherwise returns None.
        
        Returns:
            Union[np.ndarray,None]: The information of the closest hit or None if no hits exist.
        """
        if self._hits is not None and self._hits.shape[0] != 0:
            dists = np.linalg.norm(self._hits[:, :2], axis = 1)
            clos_ind = np.argmin(dists)
            return self._hits[clos_ind]
        else: return None
    
    def get_top_hit(self, top : int |np.integer):
        """
        Returns information about the top few hits by the Lidar, if there are any hits. 
        Otherwise returns None.
        
        Args:
            top (int | np.integer): The number of top hits to return.
            
        Returns:
            Union[np.ndarray,None]: The information of the top few hits or None if no hits exist.
        """
        if self._hits is not None and self._hits.shape[0] != 0:
            if top >= self._hits.shape[0]:
                top = self._hits.shape[0]
            idx = np.argpartition(self._hits[:, 2], top-1)[:top]
            return self._hits[idx]
        else: return None

    def render(self, surface: Surface, resolution: np.ndarray):
        """
        Draws the Lidar rays on a surface at the specified resolution with optional hit overlay for visualization purposes.
        
        Args:
            surface (Surface): The Pygame surface to draw onto.
            resolution (np.ndarray): The resolution of the surface in pixels per meter.
        """
        # origin_px = self._pos * resolution
        # max_range_px = self._max_range * resolution

        # angles = np.linspace(0, 2 * np.pi, self._num_rays, endpoint=False)
        # for theta in angles:
        #     dx, dy = np.cos(theta), np.sin(theta)
        #     end_px = origin_px + np.array([dx * max_range_px[0], dy * max_range_px[1]])
        #     pygame.draw.line(surface, self._color, origin_px, end_px, 1)

        # Optionally overlay hits (if scan was called)
        for hit in self._hits:
            hit_px = hit[:2] * resolution
            pygame.draw.circle(surface, (255, 0, 0), hit_px.astype(int), 1*resolution.max())


if __name__ == '__main__':
    import time
    pygame.init()
    MapW, MapH = 100, 100
    MapProp = MapH / MapW
    WinW = 800
    WinH = int(WinW * MapProp)
    FPS = 120

    w1 = EllipseObstacle(np.array([0, 0]), np.array([50, 5]), Color(20, 20, 255))
    lidar = Lidar(np.array([20, 10]), 1000, 10)
    robot = RobotStepper(np.array([10,20]),np.array([5,5]),Color(0,255,20),np.array([1,0]))
    obstacles: List = [w1]
    m = Map(np.array([MapW, MapH]), 1000, obstacles)
    lidar.scan(m)
    camera_offset = np.array([0.0, 0.0])
    zoom = 1.0
    scroll_speed = 1.0
    zoom_step = 0.1

    screen = pygame.display.set_mode((WinW, WinH), pygame.RESIZABLE)
    pygame.display.set_caption("Render de mapa en Pygame")
    clock = pygame.time.Clock()
    running = True

    while running:
        window_width, window_height = screen.get_size()
        resolution = np.array([window_width / MapW, window_height / MapH])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        keys = pygame.key.get_pressed()
        # if keys[pygame.K_LEFT]: camera_offset[0] -= scroll_speed
        # if keys[pygame.K_RIGHT]: camera_offset[0] += scroll_speed
        # if keys[pygame.K_UP]: camera_offset[1] -= scroll_speed
        # if keys[pygame.K_DOWN]: camera_offset[1] += scroll_speed
        # if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]: zoom = min(zoom + zoom_step, 5.0)
        # if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]: zoom = max(zoom - zoom_step, 0.1)
        lidar.scan(m)
        screen.fill((50, 50, 50))
        m.render(screen, resolution)
        lidar.render(screen, resolution)
        robot.render(screen, resolution)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
