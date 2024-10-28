"""
This type stub file was generated by pyright.
"""

import ast

"""
The rcsetup module contains the validation code for customization using
Matplotlib's rc settings.

Each rc setting is assigned a function used to validate any attempted changes
to that setting.  The validation functions are defined in the rcsetup module,
and are used to construct the rcParams global object which stores the settings
and is referenced throughout Matplotlib.

The default values of the rc settings are set in the default matplotlibrc file.
Any additions or deletions to the parameter set listed here should also be
propagated to the :file:`lib/matplotlib/mpl-data/matplotlibrc` in Matplotlib's
root source directory.
"""
interactive_bk = ...
non_interactive_bk = ...
all_backends = ...

class ValidateInStrings:
    def __init__(self, key, valid, ignorecase=..., *, _deprecated_since=...) -> None:
        """*valid* is a list of legal strings."""
        ...

    def __call__(self, s): ...

def validate_any(s): ...

validate_anylist = ...

def validate_bool(b):  # -> bool:
    """Convert b to ``bool`` or raise."""
    ...

def validate_axisbelow(s):  # -> bool | Literal['line']:
    ...

def validate_dpi(s):  # -> float:
    """Confirm s is string 'figure' or convert s to float or raise."""
    ...

validate_string = ...
validate_string_or_None = ...
validate_stringlist = ...
validate_int = ...
validate_int_or_None = ...
validate_float = ...
validate_float_or_None = ...
validate_floatlist = ...

def validate_fonttype(s):  # -> int | None:
    """
    Confirm that this is a Postscript or PDF font type that we know how to
    convert to.
    """
    ...

_validate_standard_backends = ...
_auto_backend_sentinel = ...

def validate_backend(s): ...
def validate_color_or_inherit(s):  # -> str | Any:
    """Return a valid color arg."""
    ...

def validate_color_or_auto(s):  # -> str | Any:
    ...

def validate_color_for_prop_cycle(s):  # -> str | Any:
    ...

def validate_color(s):  # -> str | Any:
    """Return a valid color arg."""
    ...

validate_colorlist = ...

def validate_aspect(s):  # -> float:
    ...

def validate_fontsize_None(s):  # -> str | float | None:
    ...

def validate_fontsize(s):  # -> str | float:
    ...

validate_fontsizelist = ...

def validate_fontweight(s):  # -> int:
    ...

def validate_fontstretch(s):  # -> int:
    ...

def validate_font_properties(s): ...
def validate_whiskers(s):  # -> list[Unknown] | float:
    ...

def validate_ps_distiller(s):  # -> None:
    ...

_validate_named_linestyle = ...
validate_fillstyle = ...
validate_fillstylelist = ...

def validate_markevery(
    s,
):  # -> slice | float | int | tuple[Unknown, ...] | list[Unknown] | None:
    """
    Validate the markevery property of a Line2D object.

    Parameters
    ----------
    s : None, int, (int, int), slice, float, (float, float), or list[int]

    Returns
    -------
    None, int, (int, int), slice, float, (float, float), or list[int]
    """
    ...

validate_markeverylist = ...

def validate_bbox(s):  # -> Literal['tight'] | None:
    ...

def validate_sketch(s):  # -> tuple[Unknown, ...] | None:
    ...

_range_validators = ...

def validate_hatch(s):  # -> str:
    r"""
    Validate a hatch pattern.
    A hatch pattern string can have any sequence of the following
    characters: ``\ / | - + * . x o O``.
    """
    ...

validate_hatchlist = ...
validate_dashlist = ...
_prop_validators = ...
_prop_aliases = ...

def cycler(*args, **kwargs):  # -> Cycler | Any:
    """
    Create a `~cycler.Cycler` object much like :func:`cycler.cycler`,
    but includes input validation.

    Call signatures::

      cycler(cycler)
      cycler(label=values[, label2=values2[, ...]])
      cycler(label, values)

    Form 1 copies a given `~cycler.Cycler` object.

    Form 2 creates a `~cycler.Cycler` which cycles over one or more
    properties simultaneously. If multiple properties are given, their
    value lists must have the same length.

    Form 3 creates a `~cycler.Cycler` for a single property. This form
    exists for compatibility with the original cycler. Its use is
    discouraged in favor of the kwarg form, i.e. ``cycler(label=values)``.

    Parameters
    ----------
    cycler : Cycler
        Copy constructor for Cycler.

    label : str
        The property key. Must be a valid `.Artist` property.
        For example, 'color' or 'linestyle'. Aliases are allowed,
        such as 'c' for 'color' and 'lw' for 'linewidth'.

    values : iterable
        Finite-length iterable of the property values. These values
        are validated and will raise a ValueError if invalid.

    Returns
    -------
    Cycler
        A new :class:`~cycler.Cycler` for the given properties.

    Examples
    --------
    Creating a cycler for a single property:

    >>> c = cycler(color=['red', 'green', 'blue'])

    Creating a cycler for simultaneously cycling over multiple properties
    (e.g. red circle, green plus, blue cross):

    >>> c = cycler(color=['red', 'green', 'blue'],
    ...            marker=['o', '+', 'x'])

    """
    ...

class _DunderChecker(ast.NodeVisitor):
    def visit_Attribute(self, node):  # -> None:
        ...

def validate_cycler(s):  # -> Cycler:
    """Return a Cycler object from a string repr or the object itself."""
    ...

def validate_hist_bins(s):  # -> str | int | list[Unknown]:
    ...

class _ignorecase(list):
    """A marker class indicating that a list-of-str is case-insensitive."""

    ...

_validators = ...
_hardcoded_defaults = ...
_validators = ...
