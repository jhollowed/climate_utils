import os
import sys
import pdb
import glob
import warnings
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import climate_toolbox as ctb
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnchoredText

_DEGREE_SYMBOL = u'\u00B0'

# ==========================================================================================


def ncar_rgb_to_cmap(rgb, hdrl=2):
    '''
    Constructs matplotlib colormap object from NCAR .rgb file

    Parameters
    ----------
    rgb : string
        Location of the rgb file
    hdrl : int, optional
        Header length of file; first color will be searched for at line hdrl+1 of the file 
        (where the first line is line 1). Defaults to 2

    Returns
    -------
    matplotlib ListedColormap object
    '''
    with open(rgb) as f:
        colors = f.readlines()
    colors = colors[hdrl:]
    for i in range(len(colors)):
        colors[i] = [float(c) for c in colors[i].strip('\n').split('#')[0].split()]
    colors = np.array(colors)
    if(np.max(colors) > 1):
        colors /= 256
    return ListedColormap(colors)


# -------------------------------------------------------------


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    Truncates an input colormap and returns a new colormap

    Parameters
    ----------

    Returns
    -------
    '''
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# -------------------------------------------------------------


def format_ticks(ax):
    '''
    Formats tick labels for plots in my style. Ticks face inward, and are reproduced on
    both sides of the figure

    Parameters
    ----------
    ax : pyplot axis object, or list of pyplot axis object
        The axis (or axes) to format
    '''
    try: 
        _ = iter(ax)
    except TypeError:
        ax = [ax]

    hasTwin = lambda ax, axis: \
              np.sum([np.logical_and(sib != ax, sib.bbox.bounds==ax.bbox.bounds) for sib in \
              getattr(ax, f'get_shared_{axis}_axes'.format(axis))().get_siblings(ax)])

    for axi in ax:
        axi.xaxis.set_tick_params(direction='in', which='both')
        if(not hasTwin(axi, 'x')):
            axi.yaxis.set_ticks_position('both')
        axi.yaxis.set_tick_params(direction='in', which='both')
        if(not hasTwin(axi, 'y')):
            axi.xaxis.set_ticks_position('both')


# -------------------------------------------------------------


def insert_labelled_tick(ax, axis, value, label=None):
    '''
    Inserts a single labeled tick mark to the x or y axis of a pyplot Axis, 
    maintaining the current axes limits.

    Parameters
    ----------
    ax : pyplot axis object
        The Axis in which to insert the tick
    axis : string
        Either 'x' or 'y'
    value : float
        The value at which to insert the tick
    label : string, optional
        The label to insert at this tick. Can be any string, including latex. 
        Defaults to None, in which case the value is used as the label
    '''
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert axis in ['x', 'y'], 'axis arg must be either \'x\' or \'y\''
    if(axis == 'x'):   ticks = ax.get_xticks()
    elif(axis == 'y'): ticks = ax.get_yticks()
   
    ticks = np.append(ticks, value)
    ticklabs = ticks.tolist()
    ticklabs = ['%.0f'%lab for lab in ticklabs]
    if(label is not None): 
        ticklabs[-1] = label
    if(axis == 'x'):
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabs) 
    if(axis == 'y'):
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabs) 

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# -------------------------------------------------------------


def add_annotation_box(ax, text, loc='lower right', alpha=1, fs=10, bbox_to_anchor=None):
    '''
    Inserts a white text box in the corner of an axis

    Parameters
    ----------
    ax : pyplot axis object
        The Axis in which to insert the annotation box
    text : str
        The text to insert in the annotation box
    loc : str, optional
        Location of the box. Must be a valid option for the 'loc' argument 
        to matplotlib's AnchoredText. Defaults to 'lower right'
    alpha : float, optional
        Alpha of the annotation box. Defaults to 1
    fs : float, optional
        Fontsize of the text in the annotation box. Defaults to 10
    bbox_to_anchor : a Bbox instance, a list of [left, bottom, width, height], 
                     or a list of [left, bottom] where the width and height will 
                     be assumed to be zero, optional
        Set the bbox that the annotation box is anchored to. Defaults to None, 
        in which case bbox is automatic from the parent axis. The tansform will
        be ax.transAxes, i.e. the possible bbox positions span 0->1 across each axis
    '''
    if(bbox_to_anchor is None):
        text_box = AnchoredText(text, frameon=True, loc=loc, pad=0.5, prop=dict(fontsize=fs))
    else:
        text_box = AnchoredText(text, frameon=True, loc=loc, pad=0.5, prop=dict(fontsize=fs), 
                                bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)
    text_box.set_zorder(100)
    plt.setp(text_box.patch, facecolor='white', alpha=alpha)
    ax.add_artist(text_box)


# -------------------------------------------------------------


def _lon_west_formatted(longitude, num_format='g'):
    fmt_string = u'{longitude:{num_format}}{degree}W'
    longitude += int(longitude < 0) * 360
    return fmt_string.format(longitude=abs(longitude), num_format=num_format, degree=_DEGREE_SYMBOL)
LON_WEST_FORMATTER = mticker.FuncFormatter(lambda v, pos: _lon_west_formatted(v))

def _lon_east_formatted(longitude, num_format='g'):
    fmt_string = u'{longitude:{num_format}}{degree}E'
    longitude -= int(longitude > 0) * 360
    return fmt_string.format(longitude=abs(longitude), num_format=num_format, degree=_DEGREE_SYMBOL)
LON_EAST_FORMATTER = mticker.FuncFormatter(lambda v, pos: _lon_east_formatted(v))

def _lon_deg_formatted(longitude, num_format='g'):
    # same as the west formatter right of zero, symmetric negative to the left of zero
    fmt_string = u'{longitude:{num_format}}{degree}'
    return fmt_string.format(longitude=longitude, num_format=num_format, degree=_DEGREE_SYMBOL)
LON_DEG_FORMATTER = mticker.FuncFormatter(lambda v, pos: _lon_deg_formatted(v))
