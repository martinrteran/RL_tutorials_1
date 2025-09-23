import numpy as np
from Objects import MobileObjects
from typing import  List, Union
import pygame
from pygame.color import Color
from pygame.surface import Surface
from StaticObjects import EllipseObstacle
from Maps import Map

class RobotStepper(MobileObjects):
    def render(self, surface: pygame.surface.Surface, resolution: np.ndarray):
        # resolution is the pixel/meter
        pos = self._pos * resolution
        side = self._wl * resolution
        rect = pygame.rect.Rect(pos[0], pos[1], side[0], side[1])
        pygame.draw.rect(surface, self._color,rect)

def cast_ray(args: tuple) -> np.ndarray | None:
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
    _num_rays: Union[np.integer,int]
    _max_range: Union[np.floating,float]
    _hits: np.ndarray

    def __init__(self, spos: np.ndarray, num_rays: Union[np.integer,int] = 360, 
    max_range: Union[np.floating,float] = 5.0, color: Color = Color(150, 150, 0)):
        assert num_rays > 0, f"The number of rays must be more than 0"
        super().__init__(spos, np.zeros((2,)),color)
        self._num_rays = num_rays
        self._max_range = max_range
        self._hits = np.empty((0,4))

    def scan_threading(self, map_obj: Map, max_workers = 8):
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
    
    def scan(self, map_obj: Map):# TODO - FIX
        hits = []
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
        if self._hits is not None and self._hits.shape[0] != 0:
            dists = np.linalg.norm(self._hits[:, 2], axis = 1)
            clos_ind = np.argmin(dists)
            return self._hits[clos_ind]
        else: return None
    
    def get_top_hit(self, top : int |np.integer):
        if self._hits is not None and self._hits.shape[0] != 0:
            if top >= self._hits.shape[0]:
                top = self._hits.shape[0]
            idx = np.argpartition(self._hits[:, 2], top-1)[:top]
            return self._hits[idx]
        else: return None

    def render(self, surface: Surface, resolution: np.ndarray):
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
            pygame.draw.circle(surface, (255, 0, 0), hit_px.astype(int), 1)


if __name__ == '__main__':
    import time
    pygame.init()
    MapW, MapH = 100, 50
    MapProp = MapH / MapW
    WinW = 800
    WinH = int(WinW * MapProp)
    FPS = 120

    w1 = EllipseObstacle(np.array([0, 0]), np.array([50, 5]), Color(20, 20, 255))
    lidar = Lidar(np.array([20, 10]), 1000, 10)
    obstacles: List = [w1]
    m = Map(np.array([MapW, MapH]), 1000, obstacles)
    start = time.perf_counter()
    lidar.scan(m)
    end = time.perf_counter()
    print(f"{end-start:.6f}")
    start = time.perf_counter()
    lidar.scan_threading(m,10)
    end = time.perf_counter()
    print(f"threads {8}, {end-start:.6f}")
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

        screen.fill((50, 50, 50))
        m.render(screen, resolution)
        lidar.render(screen, resolution)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
