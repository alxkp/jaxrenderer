"""
This type stub file was generated by pyright.
"""

from matplotlib.axes import Axes
from matplotlib.ticker import Formatter
from matplotlib.transforms import Transform

class GeoAxes(Axes):
    """An abstract base class for geographic projections."""

    class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """

        def __init__(self, round_to=...) -> None: ...
        def __call__(self, x, pos=...):  # -> str:
            ...

    RESOLUTION = ...
    def clear(self):  # -> None:
        ...

    def get_xaxis_transform(self, which=...): ...
    def get_xaxis_text1_transform(
        self, pad
    ):  # -> tuple[Unknown, Literal['bottom'], Literal['center']]:
        ...

    def get_xaxis_text2_transform(
        self, pad
    ):  # -> tuple[Unknown, Literal['top'], Literal['center']]:
        ...

    def get_yaxis_transform(self, which=...): ...
    def get_yaxis_text1_transform(
        self, pad
    ):  # -> tuple[Unknown, Literal['center'], Literal['right']]:
        ...

    def get_yaxis_text2_transform(
        self, pad
    ):  # -> tuple[Unknown, Literal['center'], Literal['left']]:
        ...

    def set_yscale(self, *args, **kwargs):  # -> None:
        ...
    set_xscale = ...
    def set_xlim(self, *args, **kwargs):
        """Not supported. Please consider using Cartopy."""
        ...
    set_ylim = ...
    def format_coord(self, lon, lat):  # -> LiteralString:
        """Return a format string formatting the coordinate."""
        ...

    def set_longitude_grid(self, degrees):  # -> None:
        """
        Set the number of degrees between each longitude grid.
        """
        ...

    def set_latitude_grid(self, degrees):  # -> None:
        """
        Set the number of degrees between each latitude grid.
        """
        ...

    def set_longitude_grid_ends(self, degrees):  # -> None:
        """
        Set the latitude(s) at which to stop drawing the longitude grids.
        """
        ...

    def get_data_ratio(self):  # -> float:
        """Return the aspect ratio of the data itself."""
        ...

    def can_zoom(self):  # -> Literal[False]:
        """
        Return whether this Axes supports the zoom box button functionality.

        This Axes object does not support interactive zoom box.
        """
        ...

    def can_pan(self):  # -> Literal[False]:
        """
        Return whether this Axes supports the pan/zoom button functionality.

        This Axes object does not support interactive pan/zoom.
        """
        ...

    def start_pan(self, x, y, button):  # -> None:
        ...

    def end_pan(self):  # -> None:
        ...

    def drag_pan(self, button, key, x, y):  # -> None:
        ...

class _GeoTransform(Transform):
    output_dims = ...
    def __init__(self, resolution) -> None:
        """
        Create a new geographical transform.

        Resolution is the number of steps to interpolate between each input
        line segment to approximate its path in curved space.
        """
        ...

    def __str__(self) -> str: ...
    def transform_path_non_affine(self, path):  # -> Path:
        ...

class AitoffAxes(GeoAxes):
    name = ...

    class AitoffTransform(_GeoTransform):
        """The base Aitoff transform."""

        def transform_non_affine(self, ll):  # -> NDArray[Unknown]:
            ...

        def inverted(self):  # -> InvertedAitoffTransform:
            ...

    class InvertedAitoffTransform(_GeoTransform):
        def transform_non_affine(self, xy): ...
        def inverted(self):  # -> AitoffTransform:
            ...

    def __init__(self, *args, **kwargs) -> None: ...

class HammerAxes(GeoAxes):
    name = ...

    class HammerTransform(_GeoTransform):
        """The base Hammer transform."""

        def transform_non_affine(self, ll):  # -> NDArray[Any]:
            ...

        def inverted(self):  # -> InvertedHammerTransform:
            ...

    class InvertedHammerTransform(_GeoTransform):
        def transform_non_affine(self, xy):  # -> NDArray[Any]:
            ...

        def inverted(self):  # -> HammerTransform:
            ...

    def __init__(self, *args, **kwargs) -> None: ...

class MollweideAxes(GeoAxes):
    name = ...

    class MollweideTransform(_GeoTransform):
        """The base Mollweide transform."""

        def transform_non_affine(self, ll):  # -> NDArray[Any]:
            ...

        def inverted(self):  # -> InvertedMollweideTransform:
            ...

    class InvertedMollweideTransform(_GeoTransform):
        def transform_non_affine(self, xy):  # -> NDArray[Unknown | Any]:
            ...

        def inverted(self):  # -> MollweideTransform:
            ...

    def __init__(self, *args, **kwargs) -> None: ...

class LambertAxes(GeoAxes):
    name = ...

    class LambertTransform(_GeoTransform):
        """The base Lambert transform."""

        def __init__(self, center_longitude, center_latitude, resolution) -> None:
            """
            Create a new Lambert transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Lambert space.
            """
            ...

        def transform_non_affine(self, ll):  # -> NDArray[Any]:
            ...

        def inverted(self):  # -> InvertedLambertTransform:
            ...

    class InvertedLambertTransform(_GeoTransform):
        def __init__(self, center_longitude, center_latitude, resolution) -> None: ...
        def transform_non_affine(self, xy):  # -> NDArray[Any]:
            ...

        def inverted(self):  # -> LambertTransform:
            ...

    def __init__(
        self, *args, center_longitude=..., center_latitude=..., **kwargs
    ) -> None: ...
    def clear(self):  # -> None:
        ...
