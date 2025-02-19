from typing import Literal, Tuple

import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_frame(
    T: np.ndarray[Tuple[Literal[4], Literal[4]], np.dtype[np.floating]],
    ax: Axes3D,
    label: str = "Frame",
    scale=1.0,
):
    """
    Plots a 3D coordinate frame given a transformation matrix.

    Parameters:
    - T: 4x4 numpy array, homogeneous transformation matrix.
    - ax: matplotlib 3D axis to plot on.
    - label: Name of the frame.
    - scale: Scaling factor for the axis lengths.
    """
    x, y, z = T[:3, 3]
    origin = x, y, z
    x_axis = T[:3, 0] * scale
    y_axis = T[:3, 1] * scale
    z_axis = T[:3, 2] * scale

    ax.quiver(*origin, *x_axis, color="r", label=f"{label} X" if label else None)
    ax.quiver(*origin, *y_axis, color="g", label=f"{label} Y" if label else None)
    ax.quiver(*origin, *z_axis, color="b", label=f"{label} Z" if label else None)

    # Label the frame
    ax.text(*origin, label, color="black")  #
