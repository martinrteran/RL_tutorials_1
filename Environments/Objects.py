from abc import ABC, abstractmethod
from pygame.color import Color
from pygame.surface import Surface
from typing import Optional, Union, Any
import numpy as np

class Object(ABC): # AKA: Obstacle
    """
    Abstract base class representing general objects with position and visual properties.
    
    Attributes:
        _pos (np.ndarray): Current position vector [x, y] in world coordinates.
        _wl (np.ndarray): Width/length dimensions of the object relative to resolution.
        _color (Color): Pygame color definition for rendering.
    """
    _pos: np.ndarray
    _wl: np.ndarray
    _color: Color
    # _width: np.ndarray
    # _length: np.ndarray

    def __new__(cls, *args, **kwargs):
        """Prevents direct instantiation of base Object class."""
        if cls is Object:
            raise TypeError(f"The {cls.__name__} class cannot be instantiated directly")
        return super().__new__(cls)

    def __init__(self, pos: np.ndarray, wl: np.ndarray, color: Color):
        """
        Initialize a general object.
        
        :param self: The instance being created
        :param pos: 2D position vector [x, y]
        :type pos: np.ndarray with shape (2,)
        :param wl: Width/length dimensions relative to resolution [width_factor, length_factor] 
                   where factor = dimension / resolution. For example, for a 10px obstacle at
                   800x600 screen: if width=40, then wl[0]=40/800; similarly for height.
        :type wl: np.ndarray with shape (2,)
        :param color: Pygame Color object defining the appearance
        """
        assert pos.shape == (2,), f"The position must be a 2D vector"
        assert wl.shape == (2,), f"The width and length must be a 2D vector"
        self._pos = pos
        self._wl = wl
        self._color = color

    def get_pos(self):
        """Return the current position of the object in world coordinates.
        
        :return: Position vector [x, y]
        """
        return self._pos

    def get_wl(self):
        """Return width-length dimensions
        
        :return: Width-length vector [width, length]
        """
        return self._wl

    def reset(self, pos: np.ndarray):
        """Reset object position to the specified one.
        :param pos: new position
        :type pos: np.ndarray of shape (2,)"""
        assert pos.shape == (2,), f"The position must be a 2D vector"
        self._pos = pos

    @abstractmethod
    def render(self, surface: Surface, resolution):
        pass

class MobileObjects(Object):
    _prev_pos: Optional[np.ndarray] = None
    _direction: np.ndarray

    def __new__(cls, *args, **kwargs):
        """Prevents direct instantiation of base MobileObject class."""
        if cls is MobileObjects:
            raise TypeError("The MobileObjects class cannot be instantiated directly")
        return super().__new__(cls)

    def __init__(self, spos: np.ndarray, wl: np.ndarray, color: Color):
        """
        Initialize a movable object.
        
        This constructor exists solely for code clarity and requires all necessary parameters 
        to be passed through the base class. Actual instantiation should occur via subclasses.

        :param self: The instance being created
        :param spos: Starting position vector [x, y] in world coordinates
        :type spos: np.ndarray with shape (2,)
        :param wl: Width/length dimensions (in factor form) relative to resolution
        :type wl: np.ndarray with shape (2,)
        :param color: Pygame Color object defining the appearance
        """
        super().__init__(spos, wl, color)
        self._direction = np.array([1,0])

    def move(self, dpos: np.ndarray):
        """Move object by displacement vector and record previous position.
        :param dpos: Step taken to move, like up, down, left or right
        :type pos: np.ndarray of shape (2,)
        """
        assert dpos.shape == (2,), f"The mobile change of direction must be 2D"
        self._prev_pos = self._pos.copy()
        self._pos += dpos
        self._direction = dpos
    
    def go_back(self):
        """Restore previous position if available."""
        if self._prev_pos is not None:
            self._pos = self._prev_pos
            self._prev_pos = None

class StaticObject(Object): # AKA: Obstacle
    def __new__(cls, *args, **kwargs):
        """Prevents direct instantiation of base StaticObject class."""
        if cls is StaticObject:
            raise TypeError(f"The {cls.__name__} class cannot be instantiated directly")
        return super().__new__(cls)

    def __init__(self, pos: np.ndarray, wl: np.ndarray, color: Color):
        """
        Initialize a static object (obstacle).
        
        This constructor exists solely for code clarity and requires all necessary parameters 
        to be passed through the base class. Actual instantiation should occur via subclasses.

        :param self: The instance being created
        :param pos: Position vector [x, y] in world coordinates
        :type pos: np.ndarray with shape (2,)
        :param wl: Width/length dimensions relative to resolution (factor form)
        :type wl: np.ndarray with shape (2,)
        :param color: Pygame Color object defining the appearance
        """
        super().__init__(pos, wl, color)

    @abstractmethod
    def get_grid(self, resolution: Union[int, np.integer])->Any:
        """
        Abstract method to return grid representation of object based on given screen resolution.
        
        This must be implemented by subclasses. Returns a structure that maps positions 
        in the game world (relative to resolution) to binary values indicating obstacle presence.

        :param resolution: Screen width or height
        :type resolution: int or np.integer
        
        :return: Grid representation of object boundaries and properties in numpy array.
        """
        pass