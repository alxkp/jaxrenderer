"""
This type stub file was generated by pyright.
"""

from collections.abc import MutableMapping

from matplotlib import _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.patches as mpatches

class Spine(mpatches.Patch):
    """
    An axis spine -- the line noting the data area boundaries.

    Spines are the lines connecting the axis tick marks and noting the
    boundaries of the data area. They can be placed at arbitrary
    positions. See `~.Spine.set_position` for more information.

    The default position is ``('outward', 0)``.

    Spines are subclasses of `.Patch`, and inherit much of their behavior.

    Spines draw a line, a circle, or an arc depending on if
    `~.Spine.set_patch_line`, `~.Spine.set_patch_circle`, or
    `~.Spine.set_patch_arc` has been called. Line-like is the default.

    For examples see :ref:`spines_examples`.
    """

    def __str__(self) -> str: ...
    @_docstring.dedent_interpd
    def __init__(self, axes, spine_type, path, **kwargs) -> None:
        """
        Parameters
        ----------
        axes : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance containing the spine.
        spine_type : str
            The spine type.
        path : `~matplotlib.path.Path`
            The `.Path` instance used to draw the spine.

        Other Parameters
        ----------------
        **kwargs
            Valid keyword arguments are:

            %(Patch:kwdoc)s
        """
        ...

    def set_patch_arc(self, center, radius, theta1, theta2):  # -> None:
        """Set the spine to be arc-like."""
        ...

    def set_patch_circle(self, center, radius):  # -> None:
        """Set the spine to be circular."""
        ...

    def set_patch_line(self):  # -> None:
        """Set the spine to be linear."""
        ...

    def get_patch_transform(self):  # -> IdentityTransform | Affine2D:
        ...

    def get_window_extent(self, renderer=...):  # -> Bbox:
        """
        Return the window extent of the spines in display space, including
        padding for ticks (but not their labels)

        See Also
        --------
        matplotlib.axes.Axes.get_tightbbox
        matplotlib.axes.Axes.get_window_extent
        """
        ...

    def get_path(self):  # -> Unknown | Path:
        ...

    def register_axis(self, axis):  # -> None:
        """
        Register an axis.

        An axis should be registered with its corresponding spine from
        the Axes instance. This allows the spine to clear any axis
        properties when needed.
        """
        ...

    def clear(self):  # -> None:
        """Clear the current spine."""
        ...

    @allow_rasterization
    def draw(self, renderer):  # -> None:
        ...

    def set_position(self, position):  # -> None:
        """
        Set the position of the spine.

        Spine position is specified by a 2 tuple of (position type,
        amount). The position types are:

        * 'outward': place the spine out from the data area by the specified
          number of points. (Negative values place the spine inwards.)
        * 'axes': place the spine at the specified Axes coordinate (0 to 1).
        * 'data': place the spine at the specified data coordinate.

        Additionally, shorthand notations define a special positions:

        * 'center' -> ``('axes', 0.5)``
        * 'zero' -> ``('data', 0.0)``

        Examples
        --------
        :doc:`/gallery/spines/spine_placement_demo`
        """
        ...

    def get_position(self):  # -> tuple[Literal['outward'], float] | None:
        """Return the spine position."""
        ...

    def get_spine_transform(
        self,
    ):  # -> BlendedAffine2D | BlendedGenericTransform | None:
        """Return the spine transform."""
        ...

    def set_bounds(self, low=..., high=...):  # -> None:
        """
        Set the spine bounds.

        Parameters
        ----------
        low : float or None, optional
            The lower spine bound. Passing *None* leaves the limit unchanged.

            The bounds may also be passed as the tuple (*low*, *high*) as the
            first positional argument.

            .. ACCEPTS: (low: float, high: float)

        high : float or None, optional
            The higher spine bound. Passing *None* leaves the limit unchanged.
        """
        ...

    def get_bounds(
        self,
    ):  # -> tuple[Any | Unknown | None, Any | Unknown | None] | None:
        """Get the bounds of the spine."""
        ...

    @classmethod
    def linear_spine(cls, axes, spine_type, **kwargs):  # -> Self@Spine:
        """Create and return a linear `Spine`."""
        ...

    @classmethod
    def arc_spine(
        cls, axes, spine_type, center, radius, theta1, theta2, **kwargs
    ):  # -> Self@Spine:
        """Create and return an arc `Spine`."""
        ...

    @classmethod
    def circular_spine(cls, axes, center, radius, **kwargs):  # -> Self@Spine:
        """Create and return a circular `Spine`."""
        ...

    def set_color(self, c):  # -> None:
        """
        Set the edgecolor.

        Parameters
        ----------
        c : color

        Notes
        -----
        This method does not modify the facecolor (which defaults to "none"),
        unlike the `.Patch.set_color` method defined in the parent class.  Use
        `.Patch.set_facecolor` to set the facecolor.
        """
        ...

class SpinesProxy:
    """
    A proxy to broadcast ``set_*`` method calls to all contained `.Spines`.

    The proxy cannot be used for any other operations on its members.

    The supported methods are determined dynamically based on the contained
    spines. If not all spines support a given method, it's executed only on
    the subset of spines that support it.
    """

    def __init__(self, spine_dict) -> None: ...
    def __getattr__(self, name):  # -> partial[None]:
        ...

    def __dir__(self):  # -> list[Unknown]:
        ...

class Spines(MutableMapping):
    r"""
    The container of all `.Spine`\s in an Axes.

    The interface is dict-like mapping names (e.g. 'left') to `.Spine` objects.
    Additionally, it implements some pandas.Series-like features like accessing
    elements by attribute::

        spines['top'].set_visible(False)
        spines.top.set_visible(False)

    Multiple spines can be addressed simultaneously by passing a list::

        spines[['top', 'right']].set_visible(False)

    Use an open slice to address all spines::

        spines[:].set_visible(False)

    The latter two indexing methods will return a `SpinesProxy` that broadcasts
    all ``set_*`` calls to its members, but cannot be used for any other
    operation.
    """

    def __init__(self, **kwargs) -> None: ...
    @classmethod
    def from_dict(cls, d):  # -> Self@Spines:
        ...

    def __getstate__(self):  # -> dict[str, Unknown]:
        ...

    def __setstate__(self, state):  # -> None:
        ...

    def __getattr__(self, name): ...
    def __getitem__(self, key):  # -> SpinesProxy:
        ...

    def __setitem__(self, key, value):  # -> None:
        ...

    def __delitem__(self, key):  # -> None:
        ...

    def __iter__(self):  # -> Iterator[str]:
        ...

    def __len__(self):  # -> int:
        ...
