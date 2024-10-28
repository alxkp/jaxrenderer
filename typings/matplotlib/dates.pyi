"""
This type stub file was generated by pyright.
"""

from matplotlib import _api, ticker, units

"""
Matplotlib provides sophisticated date plotting capabilities, standing on the
shoulders of python :mod:`datetime` and the add-on module dateutil_.

By default, Matplotlib uses the units machinery described in
`~matplotlib.units` to convert `datetime.datetime`, and `numpy.datetime64`
objects when plotted on an x- or y-axis. The user does not
need to do anything for dates to be formatted, but dates often have strict
formatting needs, so this module provides many axis locators and formatters.
A basic example using `numpy.datetime64` is::

    import numpy as np

    times = np.arange(np.datetime64('2001-01-02'),
                      np.datetime64('2002-02-03'), np.timedelta64(75, 'm'))
    y = np.random.randn(len(times))

    fig, ax = plt.subplots()
    ax.plot(times, y)

.. seealso::

    - :doc:`/gallery/text_labels_and_annotations/date`
    - :doc:`/gallery/ticks/date_concise_formatter`
    - :doc:`/gallery/ticks/date_demo_convert`

.. _date-format:

Matplotlib date format
----------------------

Matplotlib represents dates using floating point numbers specifying the number
of days since a default epoch of 1970-01-01 UTC; for example,
1970-01-01, 06:00 is the floating point number 0.25. The formatters and
locators require the use of `datetime.datetime` objects, so only dates between
year 0001 and 9999 can be represented.  Microsecond precision
is achievable for (approximately) 70 years on either side of the epoch, and
20 microseconds for the rest of the allowable range of dates (year 0001 to
9999). The epoch can be changed at import time via `.dates.set_epoch` or
:rc:`dates.epoch` to other dates if necessary; see
:doc:`/gallery/ticks/date_precision_and_epochs` for a discussion.

.. note::

   Before Matplotlib 3.3, the epoch was 0000-12-31 which lost modern
   microsecond precision and also made the default axis limit of 0 an invalid
   datetime.  In 3.3 the epoch was changed as above.  To convert old
   ordinal floats to the new epoch, users can do::

     new_ordinal = old_ordinal + mdates.date2num(np.datetime64('0000-12-31'))


There are a number of helper functions to convert between :mod:`datetime`
objects and Matplotlib dates:

.. currentmodule:: matplotlib.dates

.. autosummary::
   :nosignatures:

   datestr2num
   date2num
   num2date
   num2timedelta
   drange
   set_epoch
   get_epoch

.. note::

   Like Python's `datetime.datetime`, Matplotlib uses the Gregorian calendar
   for all conversions between dates and floating point numbers. This practice
   is not universal, and calendar differences can cause confusing
   differences between what Python and Matplotlib give as the number of days
   since 0001-01-01 and what other software and databases yield.  For
   example, the US Naval Observatory uses a calendar that switches
   from Julian to Gregorian in October, 1582.  Hence, using their
   calculator, the number of days between 0001-01-01 and 2006-04-01 is
   732403, whereas using the Gregorian calendar via the datetime
   module we find::

     In [1]: date(2006, 4, 1).toordinal() - date(1, 1, 1).toordinal()
     Out[1]: 732401

All the Matplotlib date converters, tickers and formatters are timezone aware.
If no explicit timezone is provided, :rc:`timezone` is assumed, provided as a
string.  If you want to use a different timezone, pass the *tz* keyword
argument of `num2date` to any date tickers or locators you create.  This can
be either a `datetime.tzinfo` instance or a string with the timezone name that
can be parsed by `~dateutil.tz.gettz`.

A wide range of specific and general purpose date tick locators and
formatters are provided in this module.  See
:mod:`matplotlib.ticker` for general information on tick locators
and formatters.  These are described below.

The dateutil_ module provides additional code to handle date ticking, making it
easy to place ticks on any kinds of dates.  See examples below.

.. _dateutil: https://dateutil.readthedocs.io

Date tickers
------------

Most of the date tickers can locate single or multiple values.  For example::

    # import constants for the days of the week
    from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

    # tick on Mondays every week
    loc = WeekdayLocator(byweekday=MO, tz=tz)

    # tick on Mondays and Saturdays
    loc = WeekdayLocator(byweekday=(MO, SA))

In addition, most of the constructors take an interval argument::

    # tick on Mondays every second week
    loc = WeekdayLocator(byweekday=MO, interval=2)

The rrule locator allows completely general date ticking::

    # tick every 5th easter
    rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
    loc = RRuleLocator(rule)

The available date tickers are:

* `MicrosecondLocator`: Locate microseconds.

* `SecondLocator`: Locate seconds.

* `MinuteLocator`: Locate minutes.

* `HourLocator`: Locate hours.

* `DayLocator`: Locate specified days of the month.

* `WeekdayLocator`: Locate days of the week, e.g., MO, TU.

* `MonthLocator`: Locate months, e.g., 7 for July.

* `YearLocator`: Locate years that are multiples of base.

* `RRuleLocator`: Locate using a `rrulewrapper`.
  `rrulewrapper` is a simple wrapper around dateutil_'s `dateutil.rrule`
  which allow almost arbitrary date tick specifications.
  See :doc:`rrule example </gallery/ticks/date_demo_rrule>`.

* `AutoDateLocator`: On autoscale, this class picks the best `DateLocator`
  (e.g., `RRuleLocator`) to set the view limits and the tick locations.  If
  called with ``interval_multiples=True`` it will make ticks line up with
  sensible multiples of the tick intervals.  For example, if the interval is
  4 hours, it will pick hours 0, 4, 8, etc. as ticks.  This behaviour is not
  guaranteed by default.

Date formatters
---------------

The available date formatters are:

* `AutoDateFormatter`: attempts to figure out the best format to use.  This is
  most useful when used with the `AutoDateLocator`.

* `ConciseDateFormatter`: also attempts to figure out the best format to use,
  and to make the format as compact as possible while still having complete
  date information.  This is most useful when used with the `AutoDateLocator`.

* `DateFormatter`: use `~datetime.datetime.strftime` format strings.
"""
__all__ = (
    "datestr2num",
    "date2num",
    "num2date",
    "num2timedelta",
    "drange",
    "set_epoch",
    "get_epoch",
    "DateFormatter",
    "ConciseDateFormatter",
    "AutoDateFormatter",
    "DateLocator",
    "RRuleLocator",
    "AutoDateLocator",
    "YearLocator",
    "MonthLocator",
    "WeekdayLocator",
    "DayLocator",
    "HourLocator",
    "MinuteLocator",
    "SecondLocator",
    "MicrosecondLocator",
    "rrule",
    "MO",
    "TU",
    "WE",
    "TH",
    "FR",
    "SA",
    "SU",
    "YEARLY",
    "MONTHLY",
    "WEEKLY",
    "DAILY",
    "HOURLY",
    "MINUTELY",
    "SECONDLY",
    "MICROSECONDLY",
    "relativedelta",
    "DateConverter",
    "ConciseDateConverter",
    "rrulewrapper",
)
_log = ...
UTC = ...

@_api.caching_module_getattr
class __getattr__:
    JULIAN_OFFSET = ...

EPOCH_OFFSET = ...
MICROSECONDLY = ...
HOURS_PER_DAY = ...
MIN_PER_HOUR = ...
SEC_PER_MIN = ...
MONTHS_PER_YEAR = ...
DAYS_PER_WEEK = ...
DAYS_PER_MONTH = ...
DAYS_PER_YEAR = ...
MINUTES_PER_DAY = ...
SEC_PER_HOUR = ...
SEC_PER_DAY = ...
SEC_PER_WEEK = ...
MUSECONDS_PER_DAY = ...
WEEKDAYS = ...
_epoch = ...

def set_epoch(epoch):  # -> None:
    """
    Set the epoch (origin for dates) for datetime calculations.

    The default epoch is :rc:`dates.epoch` (by default 1970-01-01T00:00).

    If microsecond accuracy is desired, the date being plotted needs to be
    within approximately 70 years of the epoch. Matplotlib internally
    represents dates as days since the epoch, so floating point dynamic
    range needs to be within a factor of 2^52.

    `~.dates.set_epoch` must be called before any dates are converted
    (i.e. near the import section) or a RuntimeError will be raised.

    See also :doc:`/gallery/ticks/date_precision_and_epochs`.

    Parameters
    ----------
    epoch : str
        valid UTC date parsable by `numpy.datetime64` (do not include
        timezone).

    """
    ...

def get_epoch():  # -> None:
    """
    Get the epoch used by `.dates`.

    Returns
    -------
    epoch : str
        String for the epoch (parsable by `numpy.datetime64`).
    """
    ...

_from_ordinalf_np_vectorized = ...
_dateutil_parser_parse_np_vectorized = ...

def datestr2num(
    d, default=...
):  # -> NDArray[Any] | NDArray[floating[Any]] | Any | NDArray[Unknown]:
    """
    Convert a date string to a datenum using `dateutil.parser.parse`.

    Parameters
    ----------
    d : str or sequence of str
        The dates to convert.

    default : datetime.datetime, optional
        The default date to use when fields are missing in *d*.
    """
    ...

def date2num(d):  # -> NDArray[Any] | NDArray[floating[Any]] | Any:
    """
    Convert datetime objects to Matplotlib dates.

    Parameters
    ----------
    d : `datetime.datetime` or `numpy.datetime64` or sequences of these

    Returns
    -------
    float or sequence of floats
        Number of days since the epoch.  See `.get_epoch` for the
        epoch, which can be changed by :rc:`date.epoch` or `.set_epoch`.  If
        the epoch is "1970-01-01T00:00:00" (default) then noon Jan 1 1970
        ("1970-01-01T12:00:00") returns 0.5.

    Notes
    -----
    The Gregorian calendar is assumed; this is not universal practice.
    For details see the module docstring.
    """
    ...

@_api.deprecated("3.7")
def julian2num(j):  # -> Any:
    """
    Convert a Julian date (or sequence) to a Matplotlib date (or sequence).

    Parameters
    ----------
    j : float or sequence of floats
        Julian dates (days relative to 4713 BC Jan 1, 12:00:00 Julian
        calendar or 4714 BC Nov 24, 12:00:00, proleptic Gregorian calendar).

    Returns
    -------
    float or sequence of floats
        Matplotlib dates (days relative to `.get_epoch`).
    """
    ...

@_api.deprecated("3.7")
def num2julian(n):  # -> Any:
    """
    Convert a Matplotlib date (or sequence) to a Julian date (or sequence).

    Parameters
    ----------
    n : float or sequence of floats
        Matplotlib dates (days relative to `.get_epoch`).

    Returns
    -------
    float or sequence of floats
        Julian dates (days relative to 4713 BC Jan 1, 12:00:00).
    """
    ...

def num2date(x, tz=...):  # -> Any:
    """
    Convert Matplotlib dates to `~datetime.datetime` objects.

    Parameters
    ----------
    x : float or sequence of floats
        Number of days (fraction part represents hours, minutes, seconds)
        since the epoch.  See `.get_epoch` for the
        epoch, which can be changed by :rc:`date.epoch` or `.set_epoch`.
    tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
        Timezone of *x*. If a string, *tz* is passed to `dateutil.tz`.

    Returns
    -------
    `~datetime.datetime` or sequence of `~datetime.datetime`
        Dates are returned in timezone *tz*.

        If *x* is a sequence, a sequence of `~datetime.datetime` objects will
        be returned.

    Notes
    -----
    The Gregorian calendar is assumed; this is not universal practice.
    For details, see the module docstring.
    """
    ...

_ordinalf_to_timedelta_np_vectorized = ...

def num2timedelta(x):  # -> Any:
    """
    Convert number of days to a `~datetime.timedelta` object.

    If *x* is a sequence, a sequence of `~datetime.timedelta` objects will
    be returned.

    Parameters
    ----------
    x : float, sequence of floats
        Number of days. The fraction part represents hours, minutes, seconds.

    Returns
    -------
    `datetime.timedelta` or list[`datetime.timedelta`]
    """
    ...

def drange(dstart, dend, delta):  # -> NDArray[floating[Any]]:
    """
    Return a sequence of equally spaced Matplotlib dates.

    The dates start at *dstart* and reach up to, but not including *dend*.
    They are spaced by *delta*.

    Parameters
    ----------
    dstart, dend : `~datetime.datetime`
        The date limits.
    delta : `datetime.timedelta`
        Spacing of the dates.

    Returns
    -------
    `numpy.array`
        A list floats representing Matplotlib dates.

    """
    ...

class DateFormatter(ticker.Formatter):
    """
    Format a tick (in days since the epoch) with a
    `~datetime.datetime.strftime` format string.
    """

    def __init__(self, fmt, tz=..., *, usetex=...) -> None:
        """
        Parameters
        ----------
        fmt : str
            `~datetime.datetime.strftime` format string
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            results of the formatter.
        """
        ...

    def __call__(self, x, pos=...):  # -> str | Any:
        ...

    def set_tzinfo(self, tz):  # -> None:
        ...

class ConciseDateFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use for the
    date, and to make it as compact as possible, but still be complete. This is
    most useful when used with the `AutoDateLocator`::

    >>> locator = AutoDateLocator()
    >>> formatter = ConciseDateFormatter(locator)

    Parameters
    ----------
    locator : `.ticker.Locator`
        Locator that this axis is using.

    tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
        Ticks timezone, passed to `.dates.num2date`.

    formats : list of 6 strings, optional
        Format strings for 6 levels of tick labelling: mostly years,
        months, days, hours, minutes, and seconds.  Strings use
        the same format codes as `~datetime.datetime.strftime`.  Default is
        ``['%Y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f']``

    zero_formats : list of 6 strings, optional
        Format strings for tick labels that are "zeros" for a given tick
        level.  For instance, if most ticks are months, ticks around 1 Jan 2005
        will be labeled "Dec", "2005", "Feb".  The default is
        ``['', '%Y', '%b', '%b-%d', '%H:%M', '%H:%M']``

    offset_formats : list of 6 strings, optional
        Format strings for the 6 levels that is applied to the "offset"
        string found on the right side of an x-axis, or top of a y-axis.
        Combined with the tick labels this should completely specify the
        date.  The default is::

            ['', '%Y', '%Y-%b', '%Y-%b-%d', '%Y-%b-%d', '%Y-%b-%d %H:%M']

    show_offset : bool, default: True
        Whether to show the offset or not.

    usetex : bool, default: :rc:`text.usetex`
        To enable/disable the use of TeX's math mode for rendering the results
        of the formatter.

    Examples
    --------
    See :doc:`/gallery/ticks/date_concise_formatter`

    .. plot::

        import datetime
        import matplotlib.dates as mdates

        base = datetime.datetime(2005, 2, 1)
        dates = np.array([base + datetime.timedelta(hours=(2 * i))
                          for i in range(732)])
        N = len(dates)
        np.random.seed(19680801)
        y = np.cumsum(np.random.randn(N))

        fig, ax = plt.subplots(constrained_layout=True)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.plot(dates, y)
        ax.set_title('Concise Date Formatter')

    """

    def __init__(
        self,
        locator,
        tz=...,
        formats=...,
        offset_formats=...,
        zero_formats=...,
        show_offset=...,
        *,
        usetex=...
    ) -> None:
        """
        Autoformat the date labels.  The default format is used to form an
        initial string, and then redundant elements are removed.
        """
        ...

    def __call__(self, x, pos=...):  # -> str | Any:
        ...

    def format_ticks(self, values):  # -> list[str]:
        ...

    def get_offset(self):  # -> str | Any:
        ...

    def format_data_short(self, value):  # -> Any:
        ...

class AutoDateFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use.  This
    is most useful when used with the `AutoDateLocator`.

    `.AutoDateFormatter` has a ``.scale`` dictionary that maps tick scales (the
    interval in days between one major tick) to format strings; this dictionary
    defaults to ::

        self.scaled = {
            DAYS_PER_YEAR: rcParams['date.autoformatter.year'],
            DAYS_PER_MONTH: rcParams['date.autoformatter.month'],
            1: rcParams['date.autoformatter.day'],
            1 / HOURS_PER_DAY: rcParams['date.autoformatter.hour'],
            1 / MINUTES_PER_DAY: rcParams['date.autoformatter.minute'],
            1 / SEC_PER_DAY: rcParams['date.autoformatter.second'],
            1 / MUSECONDS_PER_DAY: rcParams['date.autoformatter.microsecond'],
        }

    The formatter uses the format string corresponding to the lowest key in
    the dictionary that is greater or equal to the current scale.  Dictionary
    entries can be customized::

        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        formatter.scaled[1/(24*60)] = '%M:%S' # only show min and sec

    Custom callables can also be used instead of format strings.  The following
    example shows how to use a custom format function to strip trailing zeros
    from decimal seconds and adds the date to the first ticklabel::

        def my_format_function(x, pos=None):
            x = matplotlib.dates.num2date(x)
            if pos == 0:
                fmt = '%D %H:%M:%S.%f'
            else:
                fmt = '%H:%M:%S.%f'
            label = x.strftime(fmt)
            label = label.rstrip("0")
            label = label.rstrip(".")
            return label

        formatter.scaled[1/(24*60)] = my_format_function
    """

    def __init__(self, locator, tz=..., defaultfmt=..., *, usetex=...) -> None:
        """
        Autoformat the date labels.

        Parameters
        ----------
        locator : `.ticker.Locator`
            Locator that this axis is using.

        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.

        defaultfmt : str
            The default format to use if none of the values in ``self.scaled``
            are greater than the unit returned by ``locator._get_unit()``.

        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            results of the formatter. If any entries in ``self.scaled`` are set
            as functions, then it is up to the customized function to enable or
            disable TeX's math mode itself.
        """
        ...

    def __call__(self, x, pos=...):  # -> str | Any:
        ...

class rrulewrapper:
    """
    A simple wrapper around a `dateutil.rrule` allowing flexible
    date tick specifications.
    """

    def __init__(self, freq, tzinfo=..., **kwargs) -> None:
        """
        Parameters
        ----------
        freq : {YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY}
            Tick frequency. These constants are defined in `dateutil.rrule`,
            but they are accessible from `matplotlib.dates` as well.
        tzinfo : `datetime.tzinfo`, optional
            Time zone information. The default is None.
        **kwargs
            Additional keyword arguments are passed to the `dateutil.rrule`.
        """
        ...

    def set(self, **kwargs):  # -> None:
        """Set parameters for an existing wrapper."""
        ...

    def __getattr__(
        self, name
    ):  # -> Any | _Wrapped[(...), Unknown, (*args: Unknown, **kwargs: Unknown), list[Unknown]]:
        ...

    def __setstate__(self, state):  # -> None:
        ...

class DateLocator(ticker.Locator):
    """
    Determines the tick locations when plotting dates.

    This class is subclassed by other Locators and
    is not meant to be used on its own.
    """

    hms0d = ...
    def __init__(self, tz=...) -> None:
        """
        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

    def set_tzinfo(self, tz):  # -> None:
        """
        Set timezone info.

        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

    def datalim_to_dt(self):  # -> tuple[Any, Any]:
        """Convert axis data interval to datetime objects."""
        ...

    def viewlim_to_dt(self):  # -> tuple[Any, Any]:
        """Convert the view interval to datetime objects."""
        ...

    def nonsingular(
        self, vmin, vmax
    ):  # -> tuple[NDArray[Any] | Unknown | NDArray[floating[Any]] | Any, NDArray[Any] | Unknown | NDArray[floating[Any]] | Any] | tuple[Unknown, Unknown]:
        """
        Given the proposed upper and lower extent, adjust the range
        if it is too close to being singular (i.e. a range of ~0).
        """
        ...

class RRuleLocator(DateLocator):
    def __init__(self, o, tz=...) -> None: ...
    def __call__(
        self,
    ):  # -> list[Unknown] | NDArray[Any] | NDArray[floating[Any]] | Any:
        ...

    def tick_values(
        self, vmin, vmax
    ):  # -> NDArray[Any] | NDArray[floating[Any]] | Any:
        ...

    @staticmethod
    def get_unit_generic(freq):  # -> float | Literal[-1]:
        ...

class AutoDateLocator(DateLocator):
    """
    On autoscale, this class picks the best `DateLocator` to set the view
    limits and the tick locations.

    Attributes
    ----------
    intervald : dict

        Mapping of tick frequencies to multiples allowed for that ticking.
        The default is ::

            self.intervald = {
                YEARLY  : [1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500,
                           1000, 2000, 4000, 5000, 10000],
                MONTHLY : [1, 2, 3, 4, 6],
                DAILY   : [1, 2, 3, 7, 14, 21],
                HOURLY  : [1, 2, 3, 4, 6, 12],
                MINUTELY: [1, 5, 10, 15, 30],
                SECONDLY: [1, 5, 10, 15, 30],
                MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500,
                                1000, 2000, 5000, 10000, 20000, 50000,
                                100000, 200000, 500000, 1000000],
            }

        where the keys are defined in `dateutil.rrule`.

        The interval is used to specify multiples that are appropriate for
        the frequency of ticking. For instance, every 7 days is sensible
        for daily ticks, but for minutes/seconds, 15 or 30 make sense.

        When customizing, you should only modify the values for the existing
        keys. You should not add or delete entries.

        Example for forcing ticks every 3 hours::

            locator = AutoDateLocator()
            locator.intervald[HOURLY] = [3]  # only show every 3 hours
    """

    def __init__(
        self, tz=..., minticks=..., maxticks=..., interval_multiples=...
    ) -> None:
        """
        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        minticks : int
            The minimum number of ticks desired; controls whether ticks occur
            yearly, monthly, etc.
        maxticks : int
            The maximum number of ticks desired; controls the interval between
            ticks (ticking every other, every 3, etc.).  For fine-grained
            control, this can be a dictionary mapping individual rrule
            frequency constants (YEARLY, MONTHLY, etc.) to their own maximum
            number of ticks.  This can be used to keep the number of ticks
            appropriate to the format chosen in `AutoDateFormatter`. Any
            frequency not specified in this dictionary is given a default
            value.
        interval_multiples : bool, default: True
            Whether ticks should be chosen to be multiple of the interval,
            locking them to 'nicer' locations.  For example, this will force
            the ticks to be at hours 0, 6, 12, 18 when hourly ticking is done
            at 6 hour intervals.
        """
        ...

    def __call__(
        self,
    ):  # -> list[Unknown] | NDArray[Any] | NDArray[floating[Any]] | Any:
        ...

    def tick_values(
        self, vmin, vmax
    ):  # -> NDArray[Any] | NDArray[floating[Any]] | Any:
        ...

    def nonsingular(
        self, vmin, vmax
    ):  # -> tuple[NDArray[Any] | Unknown | NDArray[floating[Any]] | Any, NDArray[Any] | Unknown | NDArray[floating[Any]] | Any] | tuple[Unknown, Unknown]:
        ...

    def get_locator(
        self, dmin, dmax
    ):  # -> YearLocator | RRuleLocator | MicrosecondLocator:
        """Pick the best locator based on a distance."""
        ...

class YearLocator(RRuleLocator):
    """
    Make ticks on a given day of each year that is a multiple of base.

    Examples::

      # Tick every year on Jan 1st
      locator = YearLocator()

      # Tick every 5 years on July 4th
      locator = YearLocator(5, month=7, day=4)
    """

    def __init__(self, base=..., month=..., day=..., tz=...) -> None:
        """
        Parameters
        ----------
        base : int, default: 1
            Mark ticks every *base* years.
        month : int, default: 1
            The month on which to place the ticks, starting from 1. Default is
            January.
        day : int, default: 1
            The day on which to place the ticks.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

class MonthLocator(RRuleLocator):
    """
    Make ticks on occurrences of each month, e.g., 1, 3, 12.
    """

    def __init__(self, bymonth=..., bymonthday=..., interval=..., tz=...) -> None:
        """
        Parameters
        ----------
        bymonth : int or list of int, default: all months
            Ticks will be placed on every month in *bymonth*. Default is
            ``range(1, 13)``, i.e. every month.
        bymonthday : int, default: 1
            The day on which to place the ticks.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

class WeekdayLocator(RRuleLocator):
    """
    Make ticks on occurrences of each weekday.
    """

    def __init__(self, byweekday=..., interval=..., tz=...) -> None:
        """
        Parameters
        ----------
        byweekday : int or list of int, default: all days
            Ticks will be placed on every weekday in *byweekday*. Default is
            every day.

            Elements of *byweekday* must be one of MO, TU, WE, TH, FR, SA,
            SU, the constants from :mod:`dateutil.rrule`, which have been
            imported into the :mod:`matplotlib.dates` namespace.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

class DayLocator(RRuleLocator):
    """
    Make ticks on occurrences of each day of the month.  For example,
    1, 15, 30.
    """

    def __init__(self, bymonthday=..., interval=..., tz=...) -> None:
        """
        Parameters
        ----------
        bymonthday : int or list of int, default: all days
            Ticks will be placed on every day in *bymonthday*. Default is
            ``bymonthday=range(1, 32)``, i.e., every day of the month.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

class HourLocator(RRuleLocator):
    """
    Make ticks on occurrences of each hour.
    """

    def __init__(self, byhour=..., interval=..., tz=...) -> None:
        """
        Parameters
        ----------
        byhour : int or list of int, default: all hours
            Ticks will be placed on every hour in *byhour*. Default is
            ``byhour=range(24)``, i.e., every hour.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

class MinuteLocator(RRuleLocator):
    """
    Make ticks on occurrences of each minute.
    """

    def __init__(self, byminute=..., interval=..., tz=...) -> None:
        """
        Parameters
        ----------
        byminute : int or list of int, default: all minutes
            Ticks will be placed on every minute in *byminute*. Default is
            ``byminute=range(60)``, i.e., every minute.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

class SecondLocator(RRuleLocator):
    """
    Make ticks on occurrences of each second.
    """

    def __init__(self, bysecond=..., interval=..., tz=...) -> None:
        """
        Parameters
        ----------
        bysecond : int or list of int, default: all seconds
            Ticks will be placed on every second in *bysecond*. Default is
            ``bysecond = range(60)``, i.e., every second.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

class MicrosecondLocator(DateLocator):
    """
    Make ticks on regular intervals of one or more microsecond(s).

    .. note::

        By default, Matplotlib uses a floating point representation of time in
        days since the epoch, so plotting data with
        microsecond time resolution does not work well for
        dates that are far (about 70 years) from the epoch (check with
        `~.dates.get_epoch`).

        If you want sub-microsecond resolution time plots, it is strongly
        recommended to use floating point seconds, not datetime-like
        time representation.

        If you really must use datetime.datetime() or similar and still
        need microsecond precision, change the time origin via
        `.dates.set_epoch` to something closer to the dates being plotted.
        See :doc:`/gallery/ticks/date_precision_and_epochs`.

    """

    def __init__(self, interval=..., tz=...) -> None:
        """
        Parameters
        ----------
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        ...

    def set_axis(self, axis):  # -> None:
        ...

    def __call__(self):  # -> list[Unknown]:
        ...

    def tick_values(self, vmin, vmax): ...

@_api.deprecated(
    "3.6", alternative="`AutoDateLocator` and `AutoDateFormatter`" " or vendor the code"
)
def date_ticker_factory(
    span, tz=..., numticks=...
):  # -> tuple[YearLocator | MonthLocator | WeekdayLocator | DayLocator | HourLocator | MinuteLocator, DateFormatter]:
    """
    Create a date locator with *numticks* (approx) and a date formatter
    for *span* in days.  Return value is (locator, formatter).
    """
    ...

class DateConverter(units.ConversionInterface):
    """
    Converter for `datetime.date` and `datetime.datetime` data, or for
    date/time data represented as it would be converted by `date2num`.

    The 'unit' tag for such data is None or a `~datetime.tzinfo` instance.
    """

    def __init__(self, *, interval_multiples=...) -> None: ...
    def axisinfo(self, unit, axis):  # -> AxisInfo:
        """
        Return the `~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a `~datetime.tzinfo` instance or None.
        The *axis* argument is required but not used.
        """
        ...

    @staticmethod
    def convert(value, unit, axis):  # -> NDArray[Any] | NDArray[floating[Any]] | Any:
        """
        If *value* is not already a number or sequence of numbers, convert it
        with `date2num`.

        The *unit* and *axis* arguments are not used.
        """
        ...

    @staticmethod
    def default_units(x, axis):  # -> None:
        """
        Return the `~datetime.tzinfo` instance of *x* or of its first element,
        or None
        """
        ...

class ConciseDateConverter(DateConverter):
    def __init__(
        self,
        formats=...,
        zero_formats=...,
        offset_formats=...,
        show_offset=...,
        *,
        interval_multiples=...
    ) -> None: ...
    def axisinfo(self, unit, axis):  # -> AxisInfo:
        ...

class _SwitchableDateConverter:
    """
    Helper converter-like object that generates and dispatches to
    temporary ConciseDateConverter or DateConverter instances based on
    :rc:`date.converter` and :rc:`date.interval_multiples`.
    """

    def axisinfo(self, *args, **kwargs):  # -> AxisInfo:
        ...

    def default_units(self, *args, **kwargs):  # -> None:
        ...

    def convert(
        self, *args, **kwargs
    ):  # -> NDArray[Any] | NDArray[floating[Any]] | Any:
        ...
