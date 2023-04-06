import os
import sys
import pdb
import glob
import inspect
import warnings
import numpy as np
import xarray as xr
import artist_utils as aut
import cartopy.crs as ccrs
import climate_toolbox as ctb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cartopy.util import add_cyclic_point
from matplotlib.offsetbox import AnchoredText

# ---- global parameters
cmap = plt.cm.rainbow
label_fs = 14
tick_fs = 12


# ==========================================================================================


def vertical_slice(x, y, var_dict, ax, include_contours=True, 
                   plot_zscale=True, inverty=True, logy=True, 
                   center_x=None, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, 
                   grid=False, gridArgs=None, cyclic=True, 
                   annotation=None, annotation_loc='lower left', annotation_alpha=1, 
                   annotation_bbox=None,
                   no_yticklabs=False, no_xticklabs=False, label_minor_yticks=False, 
                   cbar_ticks_match_levels=True, include_contour_labels=False):
    '''
    Plot the 2D vertical slice of a variable

    Parameters
    ----------
    x : 1D array, or string
        The horizontal coordinate. Internally, this will be used with the vertical
        cordainte y to construct grid via np.meshgrid. If array, this should be the 
        coordinate values. If string, this should be the name of a field present in
        var which gives the coordinate (for e.g. a xarray DataArray).
    y : 1D array, or string
        The vertical coordinate. Internally, this will be used with the horizontal
        cordainte y to construct grid via np.meshgrid. If array, this should be the 
        coordinate values. If string, this should be the name of a field present in
        var which gives the coordinate (for e.g. a xarray DataArray).
    var_dict : dict, or list of dicts
        A dictionary for each variable to be plotted, containing the following items:
        -- var : 2D array object
            The 2D data to plot (xarray DataArray, numpy array, etc.) 
        -- plotType : string, optional
            String specifying a plotting type, matching the name of an attribute of the
            matplotlib axis class (e.g. contour, contourf, ...). 
            Defaults to 'contourf'.
        -- plotArgs : dict, optional
            Args sent to the plotType plotting call, as a dict
            Deafults to matplotlib defaults in most cases. Some specific plotTypes inherit
            custom defaults for keyword args that aren't specified by this item; see function
            source for definitions
        -- colorFormatter : string, optional
            String specifying a function to format colors on the plot, matching the name of
            either an attribute of the axis class, or the parent figure class. 
            Defailts to one of the current associations, or else None, in which no color 
            formatting will be performed:
                plt.contour --> plt.clabel
                plt.contourf --> fig.colorbar
        -- colorArgs : dict, optional
            Args sent to colorFormatter, as a dict. 
            Defaults to matplotlib defaults in most cases; see source for other defaults set
            locally. If colorFormatter is None, this is ignored and a warning will be raised
            if included in the dict.
        Any optional items not included in the dict will be inserted and set to their stated
        default values. If var_dict is a list of dicts, each variable will be overlain on the axis.
    ax : pyplot axis object
        The axis on which to render the plot.
    include_contours : bool, optional
        Whether or not to include contours over a base contourf or tricontourf plot. Contour levels
        will match the base plot, contour colors will be black, and linewidths will be 0.6. Label 
        format will match the base plot, if include_contour_labels=True. If wanting to override these 
        defaults, disable this option and instead call this funciton with plotType=contour. 
        Defaults to True.
    include_contour_labels : bool, optional
        whether or not toinclude labels on the contours automatically generated when using 
        include_contours = True. This has no effect on contour plots that are included manually
        in var_dict. Defaults to False.
    plot_zscale : bool, optional
        Whether or not to include a second y axis of values converting the original y-axis 
        values from pressure to height, assuming an isothermal atmosphere. Assumed that the
        original y-axis in pressure. Defaults to true.
    inverty : bool, optional
        Whether or not to invert the y-axis (often used for a vertical pressure coordinate). 
        Defaults to true.
    logy : string, optional
        Whether or not to set the yscale to 'log'. Defaults to True.
    center_x: float, optional
        x-coordiante on which to center the data. It will be assumed that the x-data is
        periodic, and defined in degrees.
    xlim : list of length 2, optonal
        xlimits to impose on figure.
        Defaults to None, in which case the default chosen by the plotting call is used.
    ylim : list of length 2, optional
        ylimits to impose on figure.
        Defaults to None, in which case the default chosen by the plotting call is used.
    xlabel : string, optonal
        The label fot the x-axis. Fontsize will default to 12.
        Defaults to 'x'. Set to an empty string '' to disable. 
    ylabel : string, optional
        The label for the y-axis. Fontsize will default to 12.
        Default to 'y'. Set to an empty string '' to disable.
    title : string, optional
        The title for the axis. Fontsize will default to 14. 
        Default is None, in which case the title is blank.
    grid : bool, optional
        Whether or not to turn on gridlines with a default customized configuration.
        Default is False
    gridArgs : dict, optional 
        Args sent to ax.grid, as a dict.
    cyclic : bool, optional
        Whether or not to add a cyclic point to the data, to avoid gaps in any contour plots.
        Defaults to True.
    annotation : string, optonal
        String to print in a text box in a plot corner. Defaults to None, in which case the 
        text box is disabled.
    annotation_alpha : float, optional
        Alpha for the annotation box. Defaults to 1.
    annotation_loc : str, optional
        Alpha location. Defaults to 'lower left'.
    annotation_bbox : a Bbox instance, a list of [left, bottom, width, height], 
                     or a list of [left, bottom] where the width and height will 
                     be assumed to be zero, optional
        Set the bbox that the annotation box is anchored to. Defaults to None, 
        in which case bbox is automatic from the parent axis. The tansform will
        be ax.transAxes, i.e. the possible bbox positions span 0->1 across each axis
    label_minor_yticks : bool, optonal
        Whether or not to label minor ticks on the y-axis. Defaults to False.
    cbar_ticks_match_levels : bool, optional
        Whether or not to set the colorbar ticks to match the levels of the corresponding plot. 
        Only applies if plotType is contourf, tricontourf, or pcolormesh. Defaults to True.
    '''
    
    # -------- prepare --------
    fig = ax.get_figure()

    # -------- define default plot args --------
    default_args = {
                    'contour'     :  {'levels':12, 'colors':'k', 'linewidths':0.4, 
                                      'extend':'both','zorder':1},
                    'tricontour'  :  {'levels':12, 'colors':'k', 'linewidths':0.4, 
                                      'extend':'both','zorder':1},
                    'contourf'    :  {'levels':12, 'cmap':'rainbow','extend':'both','zorder':0},
                    'tricontourf' :  {'levels':12, 'cmap':'rainbow','extend':'both','zorder':0},
                    'pcolormesh'  :  {'cmap':'rainbow','shading':'nearest','zorder':0},
                    'clabel'      :  {'inline':True, 'fmt':'%.2f', 'fontsize':tick_fs},
                    'colorbar'    :  {'ax':ax, 'location':'right', 'orientation':'vertical',
                                      'extend':'both', 'extendrect':False, 'format':'%.2f'},
                    'grid'        :  {'color':'k', 'lw':0.3, 'alpha':0.75}
                   }
    color_formatters = {
                        'contour'     : 'clabel',
                        'tricontour'  : 'clabel',
                        'contourf'    : 'colorbar',
                        'tricontourf' : 'colorbar'
                       }
    if(xlabel is None): xlabel = 'x'
    if(ylabel is None): ylabel = 'y'
   

    # -------- check inputs, add missing dict fields --------
    valid_keys = ['var', 'plotType', 'plotArgs', 'colorArgs', 'colorFormatter']
    if isinstance(var_dict, dict): var_dict = [var_dict]
    
    for i in range(len(var_dict)):
        d = var_dict[i]

        # ignore unrecognized keys
        for key in d.keys():
            if(key not in valid_keys):
                warnings.warn('var_dict key {} not recognized; ignoring'.format(key))

        # check types, create missing items
        assert 'var' in d.keys(), '2D field variable must be given in var_dict with key \'var\''
        if 'plotType' not in d.keys():
            d['plotType'] = 'contourf'
        else:
            assert isinstance(d['plotType'], str), \
                   'plotType must be given as a string, not {}'.format(type(d['plotType']))
        pl = d['plotType']
        
        if 'plotArgs' not in d.keys():
            d['plotArgs'] = {}
        else:
            assert isinstance(d['plotArgs'], dict), \
                   'plotArgs must be given as a dict, not {}'.format(type(d['plotArgs']))
        
        if 'colorArgs' not in d.keys():
            d['colorArgs'] = {}
        else:
            assert isinstance(d['colorArgs'], dict), \
                   'colorArgs must be given as a dict, not {}'.format(type(d['colorArgs']))
        
        if 'colorFormatter' not in d.keys():
            if(pl in color_formatters.keys()):
                d['colorFormatter'] = color_formatters[pl]
            else:
                d['colorFormatter'] = None
        else:
            assert isinstance(d['colorFormatter'], str) or d['colorFormatter'] is None, \
                   'colorArgs must be given as a dict, not {}'.format(type(d['colorArgs']))
        
        # merge specified kwargs with defaults
        if(pl in default_args.keys()):
            d['plotArgs'] = {**default_args[pl], **d['plotArgs']}
        if(d['colorFormatter'] in default_args.keys()):
            d['colorArgs'] = {**default_args[d['colorFormatter']], **d['colorArgs']}
        if gridArgs is not None:
            gridArgs = {**default_args['grid'], **gridArgs}
        else:
            gridArgs = default_args['grid'] 


    # -------- make recursive call is overplotting contours --------
    valid_plots_to_incl_contours = ['contourf', 'tricontourf']
    added_plots = 0
    if(include_contours):
        for i in range(len(var_dict)):
            d = var_dict[i]
            if d['plotType'] not in valid_plots_to_incl_contours:
                warnings.warn('include_contours = True not valid for plotType {}; '\
                              'skipping contours'.format(d['plotType']))
                continue
            
            if d['plotType'] == 'contourf': add_plotType = 'contour'
            elif d['plotType'] == 'tricontourf': add_plotType = 'tricontour'
            
            contour_dict = {'var':d['var'], 'plotType':add_plotType, 'plotArgs':{}, 'colorArgs':{}}
            if 'levels' in d['plotArgs'].keys():
                contour_dict['plotArgs']['levels'] = d['plotArgs']['levels']
            if 'fmt' in d['colorArgs'].keys():
                contour_dict['colorArgs']['fmt'] = d['colorArgs']['fmt']
            if 'format' in d['colorArgs'].keys():
                contour_dict['colorArgs']['fmt'] = d['colorArgs']['format']
            if not include_contour_labels:
                contour_dict['colorFormatter'] = None
            
            var_dict.append(contour_dict)
            added_plots += 1
        
        if(added_plots > 0):
            # get all kwargs, recusively call function with added contour plots
            frame = inspect.currentframe()
            argkeys, _, _, argvalues = inspect.getargvalues(frame)
            kwargs = {}
            for key in argkeys:
                if key != 'self':
                    kwargs[key] = argvalues[key]
            kwargs['include_contours'] = False # prevent recursion loop
            return vertical_slice(**kwargs)
      

    # -------- plot variables --------
    for i in range(len(var_dict)):
        d = var_dict[i]
        if(cyclic):
            d['var'], xcyc = add_cyclic_point(d['var'], coord=x, axis=1)
    if(cyclic):
        x = xcyc
    
    if(center_x is not None):
        assert cyclic, 'cannot shift x-axis with arg center_x if cyclic=False'

        # recenter on center_x in degrees, assuming periodicity in x
        xcen = x - center_x  # shifted coords only used to find rolling index
        shift_right, shift_left = False, False
        if(np.max(xcen) >= 180): shift_right = True
        if(np.min(xcen) <= -180): shift_left = True
        assert not(shift_left & shift_right), 'FAILED on centering at center_x;\
                                               data nonunique? x not in degrees? bug here?'
        if(shift_right): xroll = np.searchsorted(xcen, 180, side='right')
        if(shift_left): xroll = np.searchsorted(xcen, -180, side='left')
        x = np.roll(x, -xroll)                       # center x on x_center via a matrix "roll"
        x[x > (center_x + 180)] -= 360
        xlim = [center_x - 180, center_x + 180]

    X, Y = np.meshgrid(x, y)
    
    plots = np.empty(len(var_dict), dtype=object)
    for i in range(len(var_dict)):
        d = var_dict[i]
        if(center_x is not None):
            d['var'] = np.roll(d['var'], -xroll, axis=1) # roll the data for center_x
        plotter = getattr(ax, d['plotType'])
        plots[i] = plotter(X, Y, d['var'], **d['plotArgs'])
        # bold zero contour if exists
        if d['plotType'] in ['contour','tricontourf']:
            try:
                if(not isinstance(plots[i].levels, list)):
                    zero = plots[i].levels.tolist().index(0)
                else:
                    zero = plots[i].levels.index(0)
                bold = plots[i].collections[zero].get_linewidth() * 1.5
                plots[i].collections[zero].set_linewidth(bold)
            except ValueError:
                pass

        
    # -------- format colors --------
    cf = np.empty(len(var_dict), dtype=object)
    for i in range(len(var_dict)):
        d = var_dict[i]
        if d['colorFormatter'] is not None:
            try:
                colorFormatter = getattr(ax, d['colorFormatter'])
            except AttributeError:
                try:
                    colorFormatter = getattr(fig, d['colorFormatter'])
                except AttributeError:
                    raise AttributeError('Neither object {} or {} has attribute {}'.format(
                                          type(ax), type(fig), d['colorFormatter']))
            cf[i] = colorFormatter(plots[i], **d['colorArgs'])
    
        if(d['colorFormatter'] == 'colorbar'):
            cf[i].ax.tick_params(labelsize=tick_fs)
            cf[i].ax.xaxis.get_label().set_fontsize(label_fs)
            cf[i].ax.yaxis.get_label().set_fontsize(label_fs)
            if(cbar_ticks_match_levels):
                cf[i].set_ticks(plots[i].levels.tolist())
  

    # -------- format figure -------- 
    if(inverty): ax.invert_yaxis()
    if(logy): ax.set_yscale('log')
    if(xlim is not None): ax.set_xlim(xlim)
    if(ylim is not None): ax.set_ylim(ylim)
    if(xlabel != ''): ax.set_xlabel(xlabel, fontsize=label_fs)
    if(ylabel != ''): ax.set_ylabel(ylabel, fontsize=label_fs)
    if(no_yticklabs): ax.yaxis.set_ticklabels([])
    if(no_xticklabs): ax.xaxis.set_ticklabels([])
    ax.set_title(title, fontsize=label_fs)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    aut.format_ticks(ax)
    
    # x, y tick labels formats assuming pressure vs. degrees
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: \
                                 ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    if(label_minor_yticks):
        ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(
                                                   int(np.maximum(-np.log10(y),0)))).format(y)))
    ax.xaxis.set_major_formatter(aut.LON_DEG_FORMATTER)
    
    if(grid):
        ax.grid(**gridArgs)
    if(plot_zscale):
        ylimz = ctb.ptoz(ax.get_ylim()).m/1000
        axz = ax.twinx()
        plotterz = getattr(axz, d['plotType'])
        plotterz(X, ctb.ptoz(Y).m, d['var'], **d['plotArgs'], alpha=0)  # pressure must be in hPa
        axz.set_ylim(ylimz)
        axz.set_ylabel(r'Z [km]', fontsize=label_fs)
    if(annotation is not None):
        aut.add_annotation_box(ax, annotation, loc=annotation_loc, fs=tick_fs, alpha=annotation_alpha,
                               bbox_to_anchor=annotation_bbox)
    return cf
    
    
# -------------------------------------------------------------


def horizontal_slice(x, y, var_dict, ax, projection=ccrs.PlateCarree(), include_contours=True,
                     xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, cyclic=True,
                     grid=True, gridArgs=None, coastlines=True, coastlinesArgs=None, 
                     annotation=None, annotation_loc='lower left', annotation_alpha=1, 
                     annotation_bbox=None, include_contour_labels = False,
                     no_yticklabs=False, no_xticklabs=False, cbar_ticks_match_levels=True):
    '''
    Plot the 2D horizontal slice of a variable

    Parameters
    ----------
    x : 1D array, or string
        The horizontal coordinate. Internally, this will be used with the vertical
        cordainte y to construct grid via np.meshgrid. If array, this should be the 
        coordinate values. If string, this should be the name of a field present in
        var which gives the coordinate (for e.g. a xarray DataArray).
    y : 1D array, or string
        The vertical coordinate. Internally, this will be used with the horizontal
        cordainte y to construct grid via np.meshgrid. If array, this should be the 
        coordinate values. If string, this should be the name of a field present in
        var which gives the coordinate (for e.g. a xarray DataArray).
    var_dict : dict, or list of dicts
        A dictionary for each variable to be plotted, containing the following items:
        -- var : 2D array object
            The 2D data to plot (xarray DataArray, numpy array, etc.) 
        -- plotType : string, optional
            String specifying a plotting type, matching the name of an attribute of the
            matplotlib axis class (e.g. contour, contourf, ...). 
            Defaults to 'contourf'.
        -- plotArgs : dict, optional
            Args sent to the plotType plotting call, as a dict
            Defaults to matplotlib defaults in most cases. Some specific plotTypes inherit
            custom defaults for keyword args that aren't specified by this item; see function
            source for definitions
        -- colorFormatter : string, optional
            String specifying a function to format colors on the plot, matching the name of
            either an attribute of the axis class, or the parent figure class. 
            Defailts to one of the current associations, or else None, in which no color 
            formatting will be performed:
                plt.contour --> plt.clabel
                plt.contourf --> fig.colorbar
        -- colorArgs : dict, optional
            Args sent to colorFormatter, as a dict. 
            Defaults to matplotlib defaults in most cases; see source for other defaults set
            locally. If colorFormatter is None, this is ignored and a warning will be raised
            if included in the dict.
        Any optional items not included in the dict will be inserted and set to their stated
        default values. If var_dict is a list of dicts, each variable will be overlain on the axis.
    ax : pyplot axis object
        The axis on which to render the plot
    projection : cartopy ccrs object, optional
        Projection to apply to the slice, as a catropy.ccrs object instance.
        Default is ccrs.PlateCaree. If projection=None, assume the axis is a standard pyplot axis, 
        rather than a Cartopy GeoAxis, and apply no projection transformations to data.
    include_contours : bool, optional
        Whether or not to include contours over a base contourf or tricontourf plot. Contour levels
        will match the base plot, contour colors will be black, and linewidths will be 0.6. Label 
        format will match the base plot, if include_contour_labels=True. If wanting to override these 
        defaults, disable this option and instead call this funciton with plotType=contour. 
        Defaults to True.
    include_contour_labels : bool, optional
        whether or not toinclude labels on the contours automatically generated when using 
        include_contours = True. This has no effect on contour plots that are included manually
        in var_dict. Defaults to False.
    ylim : list of length 2, optional
        ylimits to impose on figure.
        Defaults to None, in which case the default chosen by the plotting call is used.
    xlim : list of length 2, optional
        xlimits to impose on figure.
        Defaults to None, in which case the default chosen by the plotting call is used.
    xlabel : string
        The label fot the x-axis. Fontsize will default to 12.
        Defaults to 'x'. Set to an empty string '' to disable. 
    ylabel : string
        The label for the y-axis. Fontsize will default to 12.
        Default to 'y'. Set to an empty string '' to disable.
    title : string, optional
        The title for the axis. Fontsize will default to 14. 
        Default is None, in which case the title is blank.
    grid : bool, optional
        Whether or not to turn on gridlines with a default customized configuration.
        Default is True
    gridArgs : dict, optional
        Args sent to cartopy gridlines, as a dict, or, if ax is not a GeoAxes, 
        args sent to ax.grid, as a dict.
    coastlines : bool, optional 
        Whether or not to plot coastlines. Defaults to True
    coastlinesArgs : dict, optional 
        Args sent to cartopy coastlines, as a dict.
    cyclic : bool
        Whether or not to add a cyclic point to the data, to avoid gaps in any contour plots.
        Defaults to True.
    annotation : string, optonal
        String to print in a text box in a plot corner. Defaults to None, in which case the 
        text box is disabled.
    annotation_alpha : float, optional
        Alpha for the annotation box. Defaults to 1.
    annotation_loc : str, optional
        Alpha location. Defaults to 'lower left'.
    annotation_bbox : a Bbox instance, a list of [left, bottom, width, height], 
                     or a list of [left, bottom] where the width and height will 
                     be assumed to be zero, optional
        Set the bbox that the annotation box is anchored to. Defaults to None, 
        in which case bbox is automatic from the parent axis. The tansform will
        be ax.transAxes, i.e. the possible bbox positions span 0->1 across each axis
    cbar_ticks_match_levels : bool
        Whether or not to set the colorbar ticks to match the levels of the corresponding plot. 
        Only applies if plotType is contourf, tricontourf, or pcolormesh. Defaults to True.
    '''
    
    # -------- prepare --------
    fig = ax.get_figure()
    
    # -------- define default plot args --------
    default_args = {
            'contour'    :  {'levels':12, 'colors':'k', 'extend':'both', 'transform':projection, 
                             'linewidths':0.6},
            'tricontour' :  {'levels':12, 'colors':'k', 'extend':'both', 'transform':projection, 
                             'linewidths':0.6},
            'contourf'   :  {'levels':12, 'cmap':'rainbow', 'extend':'both', 'transform':projection},
            'tricontourf':  {'levels':12, 'cmap':'rainbow', 'extend':'both', 'transform':projection},
            'clabel'     :  {'inline':True, 'fmt':'%.0f', 'fontsize':tick_fs},
            'colorbar'   :  {'ax':ax, 'orientation':'vertical', 'extend':'both', 
                            'extendrect':False, 'format':'%.2f'},
            'gridlines'  :  {'draw_labels':True, 'dms':True, 'x_inline':False, 'y_inline':False, 
                            'color':'k', 'lw':0.5, 'alpha':0.5, 'linestyle':':', 'crs':projection,
                            'xformatter':aut.LON_DEG_FORMATTER},
            'grid'       :  {'color':'k', 'lw':0.3, 'alpha':0.75},
            'coastlines' :  {'resolution':'110m', 'color':'k', 'linestyle':'-', 'alpha':0.75}
                   }
    color_formatters = {
                        'contour'     : 'clabel',
                        'tricontour'  : 'clabel',
                        'contourf'    : 'colorbar',
                        'tricontourf' : 'colorbar'
                       }
    if(xlabel is None): xlabel = 'x'
    if(ylabel is None): ylabel = 'y'
    
    # Check if axis is not a cartopy GeoAxes
    # if not, remove args to the available plotting functions thatare only defined for GeoAxes 
    if('Geo' not in type(ax).__name__):
        warnings.warn('input axis is not GeoAxes, is {}; ignoring \'transform\' '\
                      'properties of plots'.format(type(ax)))
        for p in default_args.keys(): 
            if('transform' in default_args[p].keys()): default_args[p].pop('transform')
        coastlines = False
        gridKeyStr = 'grid'
        is_geoaxis = False
    else:
        gridKeyStr = 'gridlines'
        is_geoaxis = True

    # triangulating unstructured data seems to be broken in Cartopy at the moment... 
    if(is_geoaxis):
        for i in range(len(var_dict)):
            assert 'tri' not in var_dict[i]['plotType'], \
                   'unstructed contouring via tricontour or tricontourf not supported on CartPy '\
                   'GeoAxis objects; remove projection attribute of axis and try again' 
    
    # -------- check inputs, add missing dict fields --------
    valid_keys = ['var', 'plotType', 'plotArgs', 'colorArgs', 'colorFormatter']
    if isinstance(var_dict, dict): var_dict = [var_dict]
    
    for i in range(len(var_dict)):
        d = var_dict[i]

        # ignore unrecognized keys
        for key in d.keys():
            if(key not in valid_keys):
                warnings.warn('var_dict key {} not recognized; ignoring'.format(key))

        # check types, create missing items
        assert 'var' in d.keys(), '2D field variable must be given in var_dict with key \'var\''
        if 'plotType' not in d.keys():
            d['plotType'] = 'contourf'
        else:
            assert isinstance(d['plotType'], str), \
                   'plotType must be given as a string, not {}'.format(type(d['plotType']))
        pl = d['plotType']
        
        if 'plotArgs' not in d.keys():
            d['plotArgs'] = {}
        else:
            assert isinstance(d['plotArgs'], dict), \
                   'plotArgs must be given as a dict, not {}'.format(type(d['plotArgs']))
        
        if 'colorArgs' not in d.keys():
            d['colorArgs'] = {}
        else:
            assert isinstance(d['colorArgs'], dict), \
                   'colorArgs must be given as a dict, not {}'.format(type(d['colorArgs']))
        
        if 'colorFormatter' not in d.keys():
            if(pl in color_formatters.keys()):
                d['colorFormatter'] = color_formatters[pl]
            else:
                d['colorFormatter'] = None
        else:
            assert isinstance(d['colorFormatter'], str) or d['colorFormatter'] is None, \
                   'colorArgs must be given as a dict, not {}'.format(type(d['colorArgs']))
        
        # merge specified kwargs with defaults
        if(pl in default_args.keys()):
            d['plotArgs'] = {**default_args[pl], **d['plotArgs']}
        if(d['colorFormatter'] in default_args.keys()):
            d['colorArgs'] = {**default_args[d['colorFormatter']], **d['colorArgs']}
        if gridArgs is not None:
            gridArgs = {**default_args[gridKeyStr], **gridArgs}
        else:
            gridArgs = default_args[gridKeyStr]
        if coastlinesArgs is not None:
            coastlinesArgs = {**default_args['coastlines'], **coastlinesArgs}
        else:
            coastlinesArgs = default_args['coastlines']
    

    # -------- make recursive call is overplotting contours --------
    valid_plots_to_incl_contours = ['contourf', 'tricontourf']
    added_plots = 0
    if(include_contours):
        for i in range(len(var_dict)):
            d = var_dict[i]
            if d['plotType'] not in valid_plots_to_incl_contours:
                warnings.warn('include_contours = True not valid for plotType {}; '\
                              'skipping contours'.format(d['plotType']))
                continue
            
            if d['plotType'] == 'contourf': add_plotType = 'contour'
            elif d['plotType'] == 'tricontourf': add_plotType = 'tricontour'
            
            contour_dict = {'var':d['var'], 'plotType':add_plotType, 'plotArgs':{}, 'colorArgs':{}}
            if 'levels' in d['plotArgs'].keys():
                contour_dict['plotArgs']['levels'] = d['plotArgs']['levels']
            if 'fmt' in d['colorArgs'].keys():
                contour_dict['colorArgs']['fmt'] = d['colorArgs']['fmt']
            if 'format' in d['colorArgs'].keys():
                contour_dict['colorArgs']['fmt'] = d['colorArgs']['format']
            if not include_contour_labels:
                contour_dict['colorFormatter'] = None
            
            var_dict.append(contour_dict)
            added_plots += 1
        
        if(added_plots > 0):
            # get all kwargs, recusively call function with added contour plots
            frame = inspect.currentframe()
            argkeys, _, _, argvalues = inspect.getargvalues(frame)
            kwargs = {}
            for key in argkeys:
                if key != 'self':
                    kwargs[key] = argvalues[key]
            kwargs['include_contours'] = False # prevent recursion loop
            return horizontal_slice(**kwargs)
    

    # -------- flag vars on native grid ----------
    native_grid = np.zeros(len(var_dict), dtype=bool)
    for i in range(len(var_dict)):
        d = var_dict[i]
        if(len(d['var'].shape)) == 1:
            native_grid[i] = True
        elif(len(d['var'].shape)) > 1:
            native_grid[i] = False


    # -------- plot variables --------
    plots = np.empty(len(var_dict), dtype=object)
    
    for i in range(len(var_dict)):
        d = var_dict[i]
        if(cyclic and not native_grid[i]):
            d['var'], xcyc = add_cyclic_point(d['var'], coord=x, axis=1)
        else:
            xcyc = x
        
        if(native_grid[i]):
            X, Y = x, y
        else:
            X, Y = np.meshgrid(xcyc, y)

        plotter = getattr(ax, d['plotType'])
        plots[i] = plotter(X, Y, d['var'], **d['plotArgs'])
        
        # bold zero contour if exists 
        if d['plotType'] in ['contour', 'tricontour']:
            try: 
                if(not isinstance(plots[i].levels, list)):
                    zero = plots[i].levels.tolist().index(0)
                else:
                    zero = plots[i].levels.index(0)
                bold = plots[i].collections[zero].get_linewidth() * 1.66
                plots[i].collections[zero].set_linewidth(bold)
            except ValueError:
                pass
        
    # -------- format colors --------
    cf = np.empty(len(var_dict), dtype=object)
    for i in range(len(var_dict)):
        d = var_dict[i]
        if d['colorFormatter'] is not None:
            try:
                colorFormatter = getattr(ax, d['colorFormatter'])
            except AttributeError:
                try:
                    colorFormatter = getattr(fig, d['colorFormatter'])
                except AttributeError:
                    raise AttributeError('Neither object {} or {} has attribute {}'.format(
                                          type(ax), type(fig), d['colorFormatter'])) 
            cf[i] = colorFormatter(plots[i], **d['colorArgs'])
    
        if(d['colorFormatter'] == 'colorbar'):
            cf[i].ax.tick_params(labelsize=tick_fs)
            cf[i].ax.xaxis.get_label().set_fontsize(label_fs)
            cf[i].ax.yaxis.get_label().set_fontsize(label_fs)
            if(cbar_ticks_match_levels):
                cf[i].set_ticks(plots[i].levels.tolist())
    
    # -------- format figure --------
    if(xlim is not None): ax.set_xlim(xlim)
    if(ylim is not None): ax.set_ylim(ylim) 
    if(xlabel != ''): ax.set_xlabel(xlabel, fontsize=label_fs)
    if(ylabel != ''): ax.set_ylabel(ylabel, fontsize=label_fs)
    if(no_yticklabs): ax.yaxis.set_ticklabels([])
    if(no_xticklabs): ax.xaxis.set_ticklabels([])
    if(grid):
        if(is_geoaxis):
            gl = ax.gridlines(**gridArgs)
            gl.xlabels_top = False
            gl.ylabels_right = False
        else:
            ax.grid(**gridArgs)
    if(coastlines):
        ax.coastlines(**coastlinesArgs)
    if(annotation is not None): 
        aut.add_annotation_box(ax, annotation, loc=annotation_loc, fs=tick_fs, alpha=annotation_alpha, 
                               bbox_to_anchor=annotation_bbox)
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.set_title(title, fontsize=label_fs)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)

    return cf
