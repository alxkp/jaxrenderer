"""
This type stub file was generated by pyright.
"""

from matplotlib import _api, _docstring
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.offsetbox import DraggableOffsetBox

"""
The legend module defines the Legend class, which is responsible for
drawing legends associated with axes and/or figures.

.. important::

    It is unlikely that you would ever create a Legend instance manually.
    Most users would normally create a legend via the `~.Axes.legend`
    function. For more details on legends there is also a :doc:`legend guide
    </tutorials/intermediate/legend_guide>`.

The `Legend` class is a container of legend handles and legend texts.

The legend handler map specifies how to create legend handles from artists
(lines, patches, etc.) in the axes or figures. Default legend handlers are
defined in the :mod:`~matplotlib.legend_handler` module. While not all artist
types are covered by the default legend handlers, custom legend handlers can be
defined to support arbitrary objects.

See the :doc:`legend guide </tutorials/intermediate/legend_guide>` for more
information.
"""

class DraggableLegend(DraggableOffsetBox):
    def __init__(self, legend, use_blit=..., update=...) -> None:
        """
        Wrapper around a `.Legend` to support mouse dragging.

        Parameters
        ----------
        legend : `.Legend`
            The `.Legend` instance to wrap.
        use_blit : bool, optional
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.
        update : {'loc', 'bbox'}, optional
            If "loc", update the *loc* parameter of the legend upon finalizing.
            If "bbox", update the *bbox_to_anchor* parameter.
        """
        ...

    def finalize_offset(self):  # -> None:
        ...

_legend_kw_doc_base = ...
_loc_doc_base = ...
_loc_doc_best = ...
_legend_kw_axes_st = ...
_outside_doc = ...
_legend_kw_figure_st = ...
_legend_kw_both_st = ...

class Legend(Artist):
    """
    Place a legend on the figure/axes.
    """

    codes = ...
    zorder = ...
    def __str__(self) -> str: ...
    @_api.make_keyword_only("3.6", "loc")
    @_docstring.dedent_interpd
    def __init__(
        self,
        parent,
        handles,
        labels,
        loc=...,
        numpoints=...,
        markerscale=...,
        markerfirst=...,
        reverse=...,
        scatterpoints=...,
        scatteryoffsets=...,
        prop=...,
        fontsize=...,
        labelcolor=...,
        borderpad=...,
        labelspacing=...,
        handlelength=...,
        handleheight=...,
        handletextpad=...,
        borderaxespad=...,
        columnspacing=...,
        ncols=...,
        mode=...,
        fancybox=...,
        shadow=...,
        title=...,
        title_fontsize=...,
        framealpha=...,
        edgecolor=...,
        facecolor=...,
        bbox_to_anchor=...,
        bbox_transform=...,
        frameon=...,
        handler_map=...,
        title_fontproperties=...,
        alignment=...,
        *,
        ncol=...,
        draggable=...
    ) -> None:
        """
        Parameters
        ----------
        parent : `~matplotlib.axes.Axes` or `.Figure`
            The artist that contains the legend.

        handles : list of `.Artist`
            A list of Artists (lines, patches) to be added to the legend.

        labels : list of str
            A list of labels to show next to the artists. The length of handles
            and labels should be the same. If they are not, they are truncated
            to the length of the shorter list.

        Other Parameters
        ----------------
        %(_legend_kw_doc)s

        Attributes
        ----------
        legend_handles
            List of `.Artist` objects added as legend entries.

            .. versionadded:: 3.7
        """
        ...
    legendHandles = ...
    def set_ncols(self, ncols):  # -> None:
        """Set the number of columns."""
        ...
    _loc = ...
    @allow_rasterization
    def draw(self, renderer):  # -> None:
        ...
    _default_handler_map = ...
    @classmethod
    def get_default_handler_map(
        cls,
    ):  # -> dict[type[StemContainer] | type[ErrorbarContainer] | type[Line2D] | type[Patch] | type[StepPatch] | type[LineCollection] | type[RegularPolyCollection] | type[CircleCollection] | type[BarContainer] | type[tuple[Unknown, ...]] | type[PathCollection] | type[PolyCollection], HandlerStem | HandlerErrorbar | HandlerLine2D | HandlerPatch | HandlerStepPatch | HandlerLineCollection | HandlerRegularPolyCollection | HandlerCircleCollection | HandlerTuple | HandlerPathCollection | HandlerPolyCollection]:
        """Return the global default handler map, shared by all legends."""
        ...

    @classmethod
    def set_default_handler_map(cls, handler_map):  # -> None:
        """Set the global default handler map, shared by all legends."""
        ...

    @classmethod
    def update_default_handler_map(cls, handler_map):  # -> None:
        """Update the global default handler map, shared by all legends."""
        ...

    def get_legend_handler_map(
        self,
    ):  # -> dict[type[StemContainer] | type[ErrorbarContainer] | type[Line2D] | type[Patch] | type[StepPatch] | type[LineCollection] | type[RegularPolyCollection] | type[CircleCollection] | type[BarContainer] | type[tuple[Unknown, ...]] | type[PathCollection] | type[PolyCollection], HandlerStem | HandlerErrorbar | HandlerLine2D | HandlerPatch | HandlerStepPatch | HandlerLineCollection | HandlerRegularPolyCollection | HandlerCircleCollection | HandlerTuple | HandlerPathCollection | HandlerPolyCollection]:
        """Return this legend instance's handler map."""
        ...

    @staticmethod
    def get_legend_handler(legend_handler_map, orig_handle):  # -> None:
        """
        Return a legend handler from *legend_handler_map* that
        corresponds to *orig_handler*.

        *legend_handler_map* should be a dictionary object (that is
        returned by the get_legend_handler_map method).

        It first checks if the *orig_handle* itself is a key in the
        *legend_handler_map* and return the associated value.
        Otherwise, it checks for each of the classes in its
        method-resolution-order. If no matching key is found, it
        returns ``None``.
        """
        ...

    def get_children(self):  # -> list[VPacker | FancyBboxPatch | None]:
        ...

    def get_frame(self):  # -> FancyBboxPatch:
        """Return the `~.patches.Rectangle` used to frame the legend."""
        ...

    def get_lines(self):  # -> list[Line2D]:
        r"""Return the list of `~.lines.Line2D`\s in the legend."""
        ...

    def get_patches(self):  # -> silent_list:
        r"""Return the list of `~.patches.Patch`\s in the legend."""
        ...

    def get_texts(self):  # -> silent_list:
        r"""Return the list of `~.text.Text`\s in the legend."""
        ...

    def set_alignment(self, alignment):  # -> None:
        """
        Set the alignment of the legend title and the box of entries.

        The entries are aligned as a single block, so that markers always
        lined up.

        Parameters
        ----------
        alignment : {'center', 'left', 'right'}.

        """
        ...

    def get_alignment(self):  # -> str:
        """Get the alignment value of the legend box"""
        ...

    def set_title(self, title, prop=...):  # -> None:
        """
        Set legend title and title style.

        Parameters
        ----------
        title : str
            The legend title.

        prop : `.font_manager.FontProperties` or `str` or `pathlib.Path`
            The font properties of the legend title.
            If a `str`, it is interpreted as a fontconfig pattern parsed by
            `.FontProperties`.  If a `pathlib.Path`, it is interpreted as the
            absolute path to a font file.

        """
        ...

    def get_title(self):  # -> Text:
        """Return the `.Text` instance for the legend title."""
        ...

    def get_window_extent(self, renderer=...): ...
    def get_tightbbox(self, renderer=...): ...
    def get_frame_on(self):  # -> bool:
        """Get whether the legend box patch is drawn."""
        ...

    def set_frame_on(self, b):  # -> None:
        """
        Set whether the legend box patch is drawn.

        Parameters
        ----------
        b : bool
        """
        ...
    draw_frame = ...
    def get_bbox_to_anchor(self):  # -> TransformedBbox | BboxBase | Bbox:
        """Return the bbox that the legend will be anchored to."""
        ...

    def set_bbox_to_anchor(self, bbox, transform=...):  # -> None:
        """
        Set the bbox that the legend will be anchored to.

        Parameters
        ----------
        bbox : `~matplotlib.transforms.BboxBase` or tuple
            The bounding box can be specified in the following ways:

            - A `.BboxBase` instance
            - A tuple of ``(left, bottom, width, height)`` in the given
              transform (normalized axes coordinate if None)
            - A tuple of ``(left, bottom)`` where the width and height will be
              assumed to be zero.
            - *None*, to remove the bbox anchoring, and use the parent bbox.

        transform : `~matplotlib.transforms.Transform`, optional
            A transform to apply to the bounding box. If not specified, this
            will use a transform to the bounding box of the parent.
        """
        ...

    def contains(
        self, event
    ):  # -> tuple[Literal[False], dict[Unknown, Unknown]] | tuple[bool, dict[Unknown, Unknown]]:
        ...

    def set_draggable(
        self, state, use_blit=..., update=...
    ):  # -> DraggableLegend | None:
        """
        Enable or disable mouse dragging support of the legend.

        Parameters
        ----------
        state : bool
            Whether mouse dragging is enabled.
        use_blit : bool, optional
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.
        update : {'loc', 'bbox'}, optional
            The legend parameter to be changed when dragged:

            - 'loc': update the *loc* parameter of the legend
            - 'bbox': update the *bbox_to_anchor* parameter of the legend

        Returns
        -------
        `.DraggableLegend` or *None*
            If *state* is ``True`` this returns the `.DraggableLegend` helper
            instance. Otherwise this returns *None*.
        """
        ...

    def get_draggable(self):  # -> bool:
        """Return ``True`` if the legend is draggable, ``False`` otherwise."""
        ...
