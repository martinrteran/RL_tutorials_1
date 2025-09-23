from typing import Set, Optional, Any, List, Union
import numpy as np
from pygame.color import Color
import pygame
from Objects import StaticObject


class Map:
    _wl: np.ndarray
    _resolution: Union[np.integer, int] # number of square per unit(meter)
    _grid: np.ndarray
    _obstacles: Union[Set[StaticObject], List[StaticObject], None] = None
    s_color = Color(255, 255, 255)

    def __init__(self, wl: np.ndarray, grid_resolution:  Union[np.integer, int] = 100 , obstacles: Union[Set[StaticObject], List[StaticObject], None] = None) -> None:
        assert wl.shape == (2,), f"The map must be length 2"
        self._wl = wl
        w, l = wl
        self._grid_resolution = grid_resolution
        self._grid = np.zeros(shape= (l * grid_resolution, w * grid_resolution), dtype=np.uint8)
        self._obstacles = obstacles
        if obstacles:
            for obst in obstacles:
                self.set_obstacle(obst)

    def set_obstacle(self, obstacle: StaticObject):
        x0, y0 = (obstacle.get_pos()*self._grid_resolution).astype(int)
        gr = obstacle.get_grid(self._grid_resolution)
        h,w = gr.shape
        x1 = min(x0 + w, self._grid.shape[1])
        y1 = min(y0 + h, self._grid.shape[0])
        gr = gr[:y1-y0, :x1-x0]    
        self._grid[y0:y1, x0:x1] = np.maximum(self._grid[y0:y1, x0:x1], gr)
    
    def render(self, surface: pygame.surface.Surface, resolution):
        self.render_map(surface, resolution)
        self.render_obstacles(surface, resolution)

    def render_map(self, surface: pygame.surface.Surface, resolution):
        wl = self._wl * resolution
        rect = pygame.rect.Rect(0,0, wl[0], wl[1])
        pygame.draw.rect(surface, self.s_color, rect)

    def render_obstacles(self, surface: pygame.surface.Surface, resolution):
        if self._obstacles:
            for obs in self._obstacles:
                obs.render(surface, resolution)

    def draw_grid(self, surface: pygame.Surface, resolution):
        cell_size = resolution/self._grid_resolution
        for y, x in zip(*np.nonzero(self._grid)):
            rect = pygame.Rect(x * cell_size[0], y * cell_size[1], cell_size[0], cell_size[1])
            pygame.draw.rect(surface, (0, 0, 0), rect)
        # option 1
        # for y, x in np.ndindex(self._grid.shape):
        #     if self._grid[y, x] > 0:
        #         rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
        #         pygame.draw.rect(surface, (0, 0, 0), rect)  # black for obstacles


if __name__ == '__main__':
    from  StaticObjects import Wall
    MapW, MapH = 200, 100
    MapProp = MapH/MapW
    WinW = 800
    WinH = WinW*MapProp

    FPS = 1   
    w1 = Wall(np.array([0,0]),np.array([50,5]),Color(20,20,255))
    obstacles: List[StaticObject] = [w1]
    m = Map(np.array([MapW,MapH]),10,obstacles)
    resolution = np.array([WinW/MapW, WinH/MapH])

    screen = pygame.display.set_mode((WinW, WinH), pygame.RESIZABLE)
    pygame.display.set_caption("Render de mapa en Pygame")

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                window_width, window_height = event.w, event.h
                screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
                #resolution = window_width/PP, window_height/PP

        screen.fill((50, 50, 50))
        m.render(screen, resolution)
        # m.render_map(screen, resolution)
        # m.draw_grid(screen, resolution)
        pygame.display.flip()
        clock.tick(FPS)
