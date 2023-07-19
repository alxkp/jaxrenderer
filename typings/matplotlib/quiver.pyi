"""
This type stub file was generated by pyright.
"""

from matplotlib import _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections

"""
Support for plotting vector fields.

Presently this contains Quiver and Barb. Quiver plots an arrow in the
direction of the vector, with the size of the arrow related to the
magnitude of the vector.

Barbs are like quiver in that they point along a vector, but
the magnitude of the vector is given schematically by the presence of barbs
or flags on the barb.

This will also become a home for things such as standard
deviation ellipses, which can and will be derived very easily from
the Quiver code.
"""
_quiver_doc = ...

class QuiverKey(martist.Artist):
    """Labelled arrow for use as a quiver plot scale key."""

    halign = ...
    valign = ...
    pivot = ...
    def __init__(
        self,
        Q,
        X,
        Y,
        U,
        label,
        *,
        angle=...,
        coordinates=...,
        color=...,
        labelsep=...,
        labelpos=...,
        labelcolor=...,
        fontproperties=...,
        **kwargs
    ) -> None:
        """
        Add a key to a quiver plot.

        The positioning of the key depends on *X*, *Y*, *coordinates*, and
        *labelpos*.  If *labelpos* is 'N' or 'S', *X*, *Y* give the position of
        the middle of the key arrow.  If *labelpos* is 'E', *X*, *Y* positions
        the head, and if *labelpos* is 'W', *X*, *Y* positions the tail; in
        either of these two cases, *X*, *Y* is somewhere in the middle of the
        arrow+label key object.

        Parameters
        ----------
        Q : `~matplotlib.quiver.Quiver`
            A `.Quiver` object as returned by a call to `~.Axes.quiver()`.
        X, Y : float
            The location of the key.
        U : float
            The length of the key.
        label : str
            The key label (e.g., length and units of the key).
        angle : float, default: 0
            The angle of the key arrow, in degrees anti-clockwise from the
            x-axis.
        coordinates : {'axes', 'figure', 'data', 'inches'}, default: 'axes'
            Coordinate system and units for *X*, *Y*: 'axes' and 'figure' are
            normalized coordinate systems with (0, 0) in the lower left and
            (1, 1) in the upper right; 'data' are the axes data coordinates
            (used for the locations of the vectors in the quiver plot itself);
            'inches' is position in the figure in inches, with (0, 0) at the
            lower left corner.
        color : color
            Overrides face and edge colors from *Q*.
        labelpos : {'N', 'S', 'E', 'W'}
            Position the label above, below, to the right, to the left of the
            arrow, respectively.
        labelsep : float, default: 0.1
            Distance in inches between the arrow and the label.
        labelcolor : color, default: :rc:`text.color`
            Label color.
        fontproperties : dict, optional
            A dictionary with keyword arguments accepted by the
            `~matplotlib.font_manager.FontProperties` initializer:
            *family*, *style*, *variant*, *size*, *weight*.
        **kwargs
            Any additional keyword arguments are used to override vector
            properties taken from *Q*.
        """
        ...
    @property
    def labelsep(self): ...
    @martist.allow_rasterization
    def draw(self, renderer):  # -> None:
        ...
    def set_figure(self, fig):  # -> None:
        ...
    def contains(
        self, mouseevent
    ):  # -> tuple[Literal[False], dict[Unknown, Unknown]] | tuple[Literal[True], dict[Unknown, Unknown]]:
        ...

class Quiver(mcollections.PolyCollection):
    """
    Specialized PolyCollection for arrows.

    The only API method is set_UVC(), which can be used
    to change the size, orientation, and color of the
    arrows; their locations are fixed when the class is
    instantiated.  Possibly this method will be useful
    in animations.

    Much of the work in this class is done in the draw()
    method so that as much information as possible is available
    about the plot.  In subsequent draw() calls, recalculation
    is limited to things that might have changed, so there
    should be no performance penalty from putting the calculations
    in the draw() method.
    """

    _PIVOT_VALS = ...
    @_docstring.Substitution(_quiver_doc)
    def __init__(
        self,
        ax,
        *args,
        scale=...,
        headwidth=...,
        headlength=...,
        headaxislength=...,
        minshaft=...,
        minlength=...,
        units=...,
        scale_units=...,
        angles=...,
        width=...,
        color=...,
        pivot=...,
        **kwargs
    ) -> None:
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pyplot interface documentation:
        %s
        """
        ...
    def get_datalim(self, transData):  # -> Bbox:
        ...
    @martist.allow_rasterization
    def draw(self, renderer):  # -> None:
        ...
    def set_UVC(self, U, V, C=...):  # -> None:
        ...
    quiver_doc = ...

_barbs_doc = ...

class Barbs(mcollections.PolyCollection):
    """
    Specialized PolyCollection for barbs.

    The only API method is :meth:`set_UVC`, which can be used to
    change the size, orientation, and color of the arrows.  Locations
    are changed using the :meth:`set_offsets` collection method.
    Possibly this method will be useful in animations.

    There is one internal function :meth:`_find_tails` which finds
    exactly what should be put on the barb given the vector magnitude.
    From there :meth:`_make_barbs` is used to find the vertices of the
    polygon to represent the barb based on this information.
    """

    @_docstring.interpd
    def __init__(
        self,
        ax,
        *args,
        pivot=...,
        length=...,
        barbcolor=...,
        flagcolor=...,
        sizes=...,
        fill_empty=...,
        barb_increments=...,
        rounding=...,
        flip_barb=...,
        **kwargs
    ) -> None:
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pyplot interface documentation:
        %(barbs_doc)s
        """
        ...
    def set_UVC(self, U, V, C=...):  # -> None:
        ...
    def set_offsets(self, xy):  # -> None:
        """
        Set the offsets for the barb polygons.  This saves the offsets passed
        in and masks them as appropriate for the existing U/V data.

        Parameters
        ----------
        xy : sequence of pairs of floats
        """
        ...
    barbs_doc = ...