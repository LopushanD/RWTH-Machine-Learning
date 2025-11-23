import matplotlib.artist
import matplotlib.axes
import numpy as np
from numpy.typing import NDArray
from typing import Callable
import matplotlib.pyplot as plt
import matplotlib

def plot_points(points: NDArray, ax: matplotlib.axes.Axes, marker: str,
                color: str) -> matplotlib.artist.Artist:
    """Plots a set of 2D points on the given matplotlib axes.
    
    Args:
        points: (`N`, 2) numpy array of `N` 2D points.
        ax: matplotlib axes object where the points will be plotted
        marker: the marker to use for the points.
        color: the color to use for the points.
    Returns:
        A matplotlib handle for the plotted points (can be used to e.g.
        associate the points with a legend).
    """
    handle = ax.scatter(points[:, 0], points[:, 1], marker=marker, color=color)
    return handle

def plot_function(
        func: Callable[[NDArray], NDArray], 
        a: float, 
        b: float, ax: matplotlib.axes.Axes) -> matplotlib.artist.Artist:
    """Plots a function `func`: R -> R evaluated for x in [`a`, `b`], on
    the matplotlib axes `ax`.
    
    Args:
        func: a function f: R -> R that, given a numpy vector argument,
            returns a vector of the values of f evaluated on each value
            of the input vector.
        a: The starting point of the range (x_min).
        b: The ending point of the range (x_max).
        ax: Matplotlib axes on which to plot the function.
    Returns:
        A matplotlib handle for the plotted function graph (can be used 
        to e.g. associate it with a legend).
    """
    x = np.linspace(a, b, 100)
    y = func(x)
    handle, = ax.plot(x, y)
    return handle