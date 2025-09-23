from abc import ABC, abstractmethod
from pygame.color import Color
from pygame.surface import Surface
from typing import Optional, Union, Any
import numpy as np

class Object(ABC): # AKA: Obstacle
    _pos: np.ndarray
    _wl: np.ndarray
    _color: Color
    # _width: np.ndarray
    # _length: np.ndarray

    def __new__(cls, *args, **kwargs):
        if cls is Object:
            raise TypeError(f"The {cls.__name__} class cannot be instantiated directly")
        return super().__new__(cls)

    def __init__(self, pos: np.ndarray, wl: np.ndarray, color: Color):
        assert pos.shape == (2,), f"The position must be a 2D vector"
        assert wl.shape == (2,), f"The width and length must be a 2D vector"
        self._pos = pos
        self._wl = wl
        self._color = color

    def get_pos(self):
        return self._pos

    def get_wl(self):
        return self._wl

    def reset(self, pos: np.ndarray):
        assert pos.shape == (2,), f"The position must be a 2D vector"
        self._pos = pos

    @abstractmethod
    def render(self, surface: Surface, resolution):
        pass

class MobileObjects(Object):
    _prev_pos: Optional[np.ndarray] = None

    def __new__(cls, *args, **kwargs):
        if cls is MobileObjects:
            raise TypeError("The MobileObjects class cannot be instantiated directly")
        return super().__new__(cls)

    def __init__(self, spos: np.ndarray, wl: np.ndarray, color: Color):
        super().__init__(spos, wl, color)

    def move(self, dpos: np.ndarray):
        assert dpos.shape == (2,), f"The mobile change of direction must be 2D"
        self._prev_pos = self._pos.copy()
        self._pos += dpos
    
    def go_back(self):
        if self._prev_pos:
            self._pos = self._prev_pos
            self._prev_pos = None

class StaticObject(Object): # AKA: Obstacle
    def __new__(cls, *args, **kwargs):
        if cls is StaticObject:
            raise TypeError(f"The {cls.__name__} class cannot be instantiated directly")
        return super().__new__(cls)

    def __init__(self, pos: np.ndarray, wl: np.ndarray, color: Color):
        super().__init__(pos, wl, color)

    @abstractmethod
    def get_grid(self, resolution: Union[int, np.integer])->Any:
        """Return a binary grid representing the shape of the object"""
        pass