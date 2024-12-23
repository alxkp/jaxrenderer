"""
This type stub file was generated by pyright.
"""

from matplotlib import _api, _docstring
import matplotlib.cm as cm
from matplotlib.text import Text

"""
Classes to support contour plotting and labelling for the Axes class.
"""

@_api.deprecated("3.7", alternative="Text.set_transform_rotates_text")
class ClabelText(Text):
    """
    Unlike the ordinary text, the get_rotation returns an updated
    angle in the pixel coordinate assuming that the input rotation is
    an angle in data coordinate (or whatever transform set).
    """

    def get_rotation(self):  # -> Any:
        ...

class ContourLabeler:
    """Mixin to provide labelling capability to `.ContourSet`."""

    def clabel(
        self,
        levels=...,
        *,
        fontsize=...,
        inline=...,
        inline_spacing=...,
        fmt=...,
        colors=...,
        use_clabeltext=...,
        manual=...,
        rightside_up=...,
        zorder=...
    ):  # -> silent_list:
        """
        Label a contour plot.

        Adds labels to line contours in this `.ContourSet` (which inherits from
        this mixin class).

        Parameters
        ----------
        levels : array-like, optional
            A list of level values, that should be labeled. The list must be
            a subset of ``cs.levels``. If not given, all levels are labeled.

        fontsize : str or float, default: :rc:`font.size`
            Size in points or relative size e.g., 'smaller', 'x-large'.
            See `.Text.set_size` for accepted string values.

        colors : color or colors or None, default: None
            The label colors:

            - If *None*, the color of each label matches the color of
              the corresponding contour.

            - If one string color, e.g., *colors* = 'r' or *colors* =
              'red', all labels will be plotted in this color.

            - If a tuple of colors (string, float, rgb, etc), different labels
              will be plotted in different colors in the order specified.

        inline : bool, default: True
            If ``True`` the underlying contour is removed where the label is
            placed.

        inline_spacing : float, default: 5
            Space in pixels to leave on each side of label when placing inline.

            This spacing will be exact for labels at locations where the
            contour is straight, less so for labels on curved contours.

        fmt : `.Formatter` or str or callable or dict, optional
            How the levels are formatted:

            - If a `.Formatter`, it is used to format all levels at once, using
              its `.Formatter.format_ticks` method.
            - If a str, it is interpreted as a %-style format string.
            - If a callable, it is called with one level at a time and should
              return the corresponding label.
            - If a dict, it should directly map levels to labels.

            The default is to use a standard `.ScalarFormatter`.

        manual : bool or iterable, default: False
            If ``True``, contour labels will be placed manually using
            mouse clicks. Click the first button near a contour to
            add a label, click the second button (or potentially both
            mouse buttons at once) to finish adding labels. The third
            button can be used to remove the last label added, but
            only if labels are not inline. Alternatively, the keyboard
            can be used to select label locations (enter to end label
            placement, delete or backspace act like the third mouse button,
            and any other key will select a label location).

            *manual* can also be an iterable object of (x, y) tuples.
            Contour labels will be created as if mouse is clicked at each
            (x, y) position.

        rightside_up : bool, default: True
            If ``True``, label rotations will always be plus
            or minus 90 degrees from level.

        use_clabeltext : bool, default: False
            If ``True``, use `.Text.set_transform_rotates_text` to ensure that
            label rotation is updated whenever the axes aspect changes.

        zorder : float or None, default: ``(2 + contour.get_zorder())``
            zorder of the contour labels.

        Returns
        -------
        labels
            A list of `.Text` instances for the labels.
        """
        ...

    @_api.deprecated("3.7", alternative="cs.labelTexts[0].get_font()")
    @property
    def labelFontProps(self):  # -> FontProperties:
        ...

    @_api.deprecated(
        "3.7",
        alternative="[cs.labelTexts[0].get_font().get_size()] * len(cs.labelLevelList)",
    )
    @property
    def labelFontSizeList(self):  # -> list[float | Unknown]:
        ...

    @_api.deprecated("3.7", alternative="cs.labelTexts")
    @property
    def labelTextsList(self):  # -> silent_list:
        ...

    def print_label(self, linecontour, labelwidth):
        """Return whether a contour is long enough to hold a label."""
        ...

    def too_close(self, x, y, lw):  # -> bool:
        """Return whether a label is already near this location."""
        ...

    @_api.deprecated("3.7", alternative="Artist.set")
    def set_label_props(self, label, text, color):  # -> None:
        """Set the label properties - color, fontsize, text."""
        ...

    def get_text(self, lev, fmt):  # -> str:
        """Get the text of the label."""
        ...

    def locate_label(
        self, linecontour, labelwidth
    ):  # -> tuple[Any | Unbound, Any | Unbound, Any | int]:
        """
        Find good place to draw a label (relatively flat part of the contour).
        """
        ...

    def calc_label_rot_and_inline(
        self, slc, ind, lw, lc=..., spacing=...
    ):  # -> tuple[Any, list[Unknown]]:
        """
        Calculate the appropriate label rotation given the linecontour
        coordinates in screen units, the index of the label location and the
        label width.

        If *lc* is not None or empty, also break contours and compute
        inlining.

        *spacing* is the empty space to leave around the label, in pixels.

        Both tasks are done together to avoid calculating path lengths
        multiple times, which is relatively costly.

        The method used here involves computing the path length along the
        contour in pixel coordinates and then looking approximately (label
        width / 2) away from central point to determine rotation and then to
        break contour if desired.
        """
        ...

    def add_label(self, x, y, rotation, lev, cvalue):  # -> None:
        """Add contour label without `.Text.set_transform_rotates_text`."""
        ...

    def add_label_clabeltext(self, x, y, rotation, lev, cvalue):  # -> None:
        """Add contour label with `.Text.set_transform_rotates_text`."""
        ...

    def add_label_near(
        self, x, y, inline=..., inline_spacing=..., transform=...
    ):  # -> None:
        """
        Add a label near the point ``(x, y)``.

        Parameters
        ----------
        x, y : float
            The approximate location of the label.
        inline : bool, default: True
            If *True* remove the segment of the contour beneath the label.
        inline_spacing : int, default: 5
            Space in pixels to leave on each side of label when placing
            inline. This spacing will be exact for labels at locations where
            the contour is straight, less so for labels on curved contours.
        transform : `.Transform` or `False`, default: ``self.axes.transData``
            A transform applied to ``(x, y)`` before labeling.  The default
            causes ``(x, y)`` to be interpreted as data coordinates.  `False`
            is a synonym for `.IdentityTransform`; i.e. ``(x, y)`` should be
            interpreted as display coordinates.
        """
        ...

    def pop_label(self, index=...):  # -> None:
        """Defaults to removing last label, but any index can be supplied"""
        ...

    def labels(self, inline, inline_spacing):  # -> None:
        ...

    def remove(self):  # -> None:
        ...

@_docstring.dedent_interpd
class ContourSet(cm.ScalarMappable, ContourLabeler):
    """
    Store a set of contour lines or filled regions.

    User-callable method: `~.Axes.clabel`

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`

    levels : [level0, level1, ..., leveln]
        A list of floating point numbers indicating the contour levels.

    allsegs : [level0segs, level1segs, ...]
        List of all the polygon segments for all the *levels*.
        For contour lines ``len(allsegs) == len(levels)``, and for
        filled contour regions ``len(allsegs) = len(levels)-1``. The lists
        should look like ::

            level0segs = [polygon0, polygon1, ...]
            polygon0 = [[x0, y0], [x1, y1], ...]

    allkinds : ``None`` or [level0kinds, level1kinds, ...]
        Optional list of all the polygon vertex kinds (code types), as
        described and used in Path. This is used to allow multiply-
        connected paths such as holes within filled polygons.
        If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
        should look like ::

            level0kinds = [polygon0kinds, ...]
            polygon0kinds = [vertexcode0, vertexcode1, ...]

        If *allkinds* is not ``None``, usually all polygons for a
        particular contour level are grouped together so that
        ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

    **kwargs
        Keyword arguments are as described in the docstring of
        `~.Axes.contour`.

    %(contour_set_attributes)s
    """

    def __init__(
        self,
        ax,
        *args,
        levels=...,
        filled=...,
        linewidths=...,
        linestyles=...,
        hatches=...,
        alpha=...,
        origin=...,
        extent=...,
        cmap=...,
        colors=...,
        norm=...,
        vmin=...,
        vmax=...,
        extend=...,
        antialiased=...,
        nchunk=...,
        locator=...,
        transform=...,
        negative_linestyles=...,
        **kwargs
    ) -> None:
        """
        Draw contour lines or filled regions, depending on
        whether keyword arg *filled* is ``False`` (default) or ``True``.

        Call signature::

            ContourSet(ax, levels, allsegs, [allkinds], **kwargs)

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` object to draw on.

        levels : [level0, level1, ..., leveln]
            A list of floating point numbers indicating the contour
            levels.

        allsegs : [level0segs, level1segs, ...]
            List of all the polygon segments for all the *levels*.
            For contour lines ``len(allsegs) == len(levels)``, and for
            filled contour regions ``len(allsegs) = len(levels)-1``. The lists
            should look like ::

                level0segs = [polygon0, polygon1, ...]
                polygon0 = [[x0, y0], [x1, y1], ...]

        allkinds : [level0kinds, level1kinds, ...], optional
            Optional list of all the polygon vertex kinds (code types), as
            described and used in Path. This is used to allow multiply-
            connected paths such as holes within filled polygons.
            If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
            should look like ::

                level0kinds = [polygon0kinds, ...]
                polygon0kinds = [vertexcode0, vertexcode1, ...]

            If *allkinds* is not ``None``, usually all polygons for a
            particular contour level are grouped together so that
            ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

        **kwargs
            Keyword arguments are as described in the docstring of
            `~.Axes.contour`.
        """
        ...

    def get_transform(self):  # -> Transform:
        """Return the `.Transform` instance used by this ContourSet."""
        ...

    def __getstate__(self):  # -> dict[str, Any]:
        ...

    def legend_elements(
        self, variable_name=..., str_format=...
    ):  # -> tuple[list[Unknown], list[Unknown]]:
        """
        Return a list of artists and labels suitable for passing through
        to `~.Axes.legend` which represent this ContourSet.

        The labels have the form "0 < x <= 1" stating the data ranges which
        the artists represent.

        Parameters
        ----------
        variable_name : str
            The string used inside the inequality used on the labels.
        str_format : function: float -> str
            Function used to format the numbers in the labels.

        Returns
        -------
        artists : list[`.Artist`]
            A list of the artists.
        labels : list[str]
            A list of the labels.
        """
        ...

    def changed(self):  # -> None:
        ...

    def get_alpha(self):  # -> None:
        """Return alpha to be applied to all ContourSet artists."""
        ...

    def set_alpha(self, alpha):  # -> None:
        """
        Set the alpha blending value for all ContourSet artists.
        *alpha* must be between 0 (transparent) and 1 (opaque).
        """
        ...

    def find_nearest_contour(
        self, x, y, indices=..., pixel=...
    ):  # -> tuple[int | Unknown | None, int | None, signedinteger[_NBitIntP] | Literal[0] | None, Unknown | None, Unknown | None, float | Unknown]:
        """
        Find the point in the contour plot that is closest to ``(x, y)``.

        This method does not support filled contours.

        Parameters
        ----------
        x, y : float
            The reference point.
        indices : list of int or None, default: None
            Indices of contour levels to consider.  If None (the default), all
            levels are considered.
        pixel : bool, default: True
            If *True*, measure distance in pixel (screen) space, which is
            useful for manual contour labeling; else, measure distance in axes
            space.

        Returns
        -------
        contour : `.Collection`
            The contour that is closest to ``(x, y)``.
        segment : int
            The index of the `.Path` in *contour* that is closest to
            ``(x, y)``.
        index : int
            The index of the path segment in *segment* that is closest to
            ``(x, y)``.
        xmin, ymin : float
            The point in the contour plot that is closest to ``(x, y)``.
        d2 : float
            The squared distance from ``(xmin, ymin)`` to ``(x, y)``.
        """
        ...

    def remove(self):  # -> None:
        ...

@_docstring.dedent_interpd
class QuadContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions.

    This class is typically not instantiated directly by the user but by
    `~.Axes.contour` and `~.Axes.contourf`.

    %(contour_set_attributes)s
    """

    ...
