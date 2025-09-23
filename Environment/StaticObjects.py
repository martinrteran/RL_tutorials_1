import numpy as np
import pygame
from pygame.color import Color
from pygame.surface import Surface
from Objects import StaticObject

class Wall(StaticObject):

    def render(self, surface: Surface, resolution):
        pos = self._pos * resolution
        side = self._wl * resolution
        rect = pygame.rect.Rect(pos[0], pos[1], side[0], side[1])
        pygame.draw.rect(surface, self._color,rect)

    def get_grid(self, resolution: int | np.integer):
        wl = self._wl * resolution
        lw = np.array([wl[1],wl[0]])
        grid = np.ones(lw,np.uint8)
        return grid
        # for x, y in np.ndindex(wl[0], wl[1]):
        #     pass

    def __init__(self, pos: np.ndarray, wl: np.ndarray, color: Color):
        super().__init__(pos,wl,color)

class EllipseObstacle(StaticObject):
    """ Define the square that encapsulates the elipse/circle"""
    def __init__(self, pos: np.ndarray, wl: np.ndarray, color: Color):
        super().__init__(pos,wl,color)
    def render(self, surface: Surface, resolution):
        pos = self._pos * resolution
        side = self._wl * resolution
        rect = pygame.rect.Rect(pos[0], pos[1], side[0], side[1])
        pygame.draw.ellipse(surface, self._color,rect)
    def get_grid(self, resolution: int | np.integer) -> np.ndarray:
        wl = (self._wl * resolution).astype(int) # both radius
        ab = wl/2
        # Create a grid of coordinates
        y_indices, x_indices = np.ogrid[:wl[1], :wl[0]]  # shape: (height, width)
        dist_sq = (((x_indices - ab[0])/ab[0])**2 + (((y_indices - ab[1])/ab[1])**2))

        # Create binary mask where distance <= radius
        grid = (dist_sq <= 1).astype(np.uint8)
        return grid

class CircleObstacle(StaticObject):
    """ Define the square that encapsulates the circle"""
    def render(self, surface: Surface, resolution):
        center = self._center * resolution
        radius = (self._radius * resolution)
        pygame.draw.circle(surface, self._color,center,radius[0])
    def get_grid(self, resolution: int | np.integer) -> np.ndarray:
        r = int(self._radius[0].item() * resolution)
        wl = self._wl.astype(int) * resolution

        # Create a grid of coordinates
        y_indices, x_indices = np.ogrid[:wl[1], :wl[0]]  # shape: (height, width)
        dist_sq = (x_indices - r)**2 + (y_indices - r)**2

        # Create binary mask where distance <= radius
        grid = (dist_sq <= r**2).astype(np.uint8)
        return grid

    def get_radius(self):
        return self._radius
    def get_center(self):
        return self._center
    
    def __init__(self, center: np.ndarray, radius: np.ndarray, color: Color):
        assert radius.shape == (1,), f"For the circle, there must be only radius"
        assert radius > 0, f"The radius must be positive and greater than 0"
        super().__init__(center-radius,2*np.array([radius.item(), radius.item()]),color)
        self._radius = radius * np.array([1,1])
        self._center = center

# class LineObstacle(StaticObject):
#     _thickness: Union[np.floating, float]
#     def render(self, surface: pygame.Surface, resolution):
#         pygame.draw.line(surface, self._color, self._pos, self._pos + self._wl,self._thickness) # type: ignore
#     def __init__(self, pos_ini: np.ndarray, wl: np.ndarray, thickness: Union[np.floating, float], color: pygame.color.Color):
#         assert thickness > 0, f"The line must have a thickness"
#         super().__init__(pos_ini,wl,color)
#         self._thickness = thickness

    

if __name__ == '__main__':
    ratio = 6/8 # height/width

    window_width, window_height = 800, 600
    PW = 100.0
    PH = PW*window_height/window_width
    FPS = 120

    x = Wall(np.array([0,0]), np.array([2,90]), pygame.color.Color(255, 255, 255))
    y = Wall(np.array([0,0]), np.array([90,2]), pygame.color.Color(255, 255, 255))
    w = Wall(np.array([10,10]), np.array([20,30]), pygame.color.Color(255, 9, 0))
    e = EllipseObstacle(np.array([20,20]), np.array([20,10]), pygame.color.Color(100, 100, 100))
    c = CircleObstacle(np.array([15,25]), np.array([5]), pygame.color.Color(100, 200, 100))
    c.get_grid(1000)
    e.get_grid(1000)
    resolution = window_width/PW, window_height/PH

    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
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
        x.render(screen, resolution)
        y.render(screen, resolution)
        w.render(screen, resolution)
        e.render(screen, resolution)
        c.render(screen, resolution)
        pygame.display.flip()
        clock.tick(FPS)