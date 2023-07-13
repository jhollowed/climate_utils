`'''
This module provides various functions for computing various climate metrics,
indices, and performing other common tasks on climate or weather data
'''

import os
import sys
import pdb
import glob
import cftime
import warnings
import numpy as np
import xarray as xr
from metpy import calc as mc
from metpy.units import units
import metpy.constants as const
from cftime import DatetimeNoLeap

sys.path.append('/glade/u/home/jhollowed/repos/ncl_exports')
from wrappers import dpres_hybrid_ccm

# ===============================================================

def check_data_inputs(data):
    acceptable_types = [xr.core.dataset.Dataset, xr.core.dataarray.DataArray, str, np.str_]
    if(type(data) not in acceptable_types):
        raise RuntimeError('data must be provided as an xarray Dataset,'\
                           'DataArray, or a string to the file')
    if(type(data) == str or type(data) == np.str_):
        data = xr.open_dataset(data)
    return data


# -------------------------------------------------------------


def time2day(time):
    '''
    Converts cftime times to number of days since first timestamp

    Parameters
    ----------
    time : array of type from cftime
        array of times, e.g. an xarray DataSet, DataArray, numpy.array...

    Returns
    -------
    Number of days since time 0, for all times in the input
    '''
    if(type(time) == xr.core.dataarray.DataArray):
        time = np.array(time)
    start = '{}-{}-{}'.format(time[0].year, time[0].month, time[0].day)
    return cftime.date2num(time, 'days since {}'.format(start))


# -------------------------------------------------------------


def ptoz(p):
    '''
    Converts pressure to geopotential height assuming an isothermal atmosphere

    Parameters
    ----------
    p : float or float array
        pressure in hPa

    Returns
    -------
    z : float or flaot array
        the geopotential height in m
    '''
    try: _ = p.m
    except: p = p*units.hPa
    P0 = 1000*units.hPa
    T0 = 250*units.K 
    #T0 = 185.5*units.K # temporary to force 25km to match 10hPa for plume
    H = const.Rd*T0/const.g 
    return H * np.log(P0/p)


# -------------------------------------------------------------


def ztop(z):
    '''
    Converts geopotential height to pressure assuming an isothermal atmosphere

    Parameters
    ----------
    z : float or float array
        the geopotential height

    Returns
    -------
    p : float or float array
        pressure in hPa
    '''
    try: _ = z.m
    except: z = z*units.m
    P0 = 1000*units.hPa
    T0 = 250*units.K
    H = const.Rd*T0/const.g 
    return P0 * np.exp(-z/H)


# -------------------------------------------------------------


def latlon_weighted_stat(X, stat, ps=None, hyai=None, hybi=None, p0=100000):
    '''
    Computes a spatially-weighted statistic of X in lat-lon coordinates. 
    Data values are weighted in the vertical direction by pressure 
    thickness, and in the meridional direction by cos(lat). No 
    weighting is applied in the zonal direction. This function accepts
    input data in a variety of shapes, and supports the following use-cases:
    A: If the spatial dimensions include ('lat'), a horizontal area-weighted 
       statistic is computed and returned. 
    B: If the spatial dimensions include ('lev'), a pressure-weighted 
       statistic is computed and returned.
    C: If the spatial dimensions include ('lat', 'lev'), a volume-weighted 
       statistic is computed and returned.
    - If the input data X suggests use case B or C, then the optional arguments
      (ps, hyai, hybi) become required.
    - Any extra dimensions (e.g. 'time') may also be present, though they will be
      ignored (the statistic will not be computed over them). The dimension 
      ordering does not matter.
    
    Parameters
    ----------
    X : xarray DataArray
        data on which to take the weighted statistic
    stat : string, optional, optional
        weigthted statistic to take. Options are 'sum' or 'mean'
    ps : xarray DataArray, optional
        surface pressure with spatial dimensions matching the horizontal 
        dimensions of X. Required if X includes the dimension 'lev'
    hyai : xarray DataArray, optional
        1-D hybrid A coefficients on the interface levels
        Required if X includes the dimension 'lev'
    hybi : xarray DataArray, optional
        1-D hybrid B coefficients on the interface levels
        Required if X includes the dimension 'lev'
    p0 : float, optional
        reference pressure in Pa. Defaults to 100000 Pa.
        
    Returns
    -------
    X_stat : xarray DataArray
        Array with dimensions matching any dimensions in the input X leftover
        after taking the target statistic over spatial dimensions (e.g. time)
    '''
    
    # check that target function is valid
    available_stats=['sum', 'mean']
    if(stat not in available_stats):
        raise RuntimeError('argument \'stat\' invalid. Options are: {}'\
                           .format(available_stats))
    # lambda function to return function handle of statistic estimator
    stat_func = lambda X, d: X.sum(dim=d) if stat=='sum' else X.mean(dim=d)
    
    # check which horizontal and vertical dimensions exist in dataset
    horz_mask = [dim in ['lat', 'lon'] for dim in X.dims]
    horz_dims = np.array(X.dims)[horz_mask]
    vert_mask = [dim in ['lev'] for dim in X.dims]
    vert_dims = np.array(X.dims)[vert_mask]
   
    # enforce use cases
    if(len(horz_dims) == 0 and len(vert_dims) == 0): 
        raise RuntimeError('input data does not contain dimensions '\
                           '\'lat\', \'lon\', or \'lev\'')
    if(len(vert_dims) != 0 and (ps is None or hyai is None or hybi is None)):
        raise RuntimeError('arguments \'ps\', \'hayi\', and \'hybi\' must be '\
                           'included when input data contains dimension \'lev\'')
    
    # make this assignment in case vertical weighting is skipped
    # if vertical weighting is not skipped, the statistic MUST
    # be applied in the vertical before the horizontal so that
    # dimensions of X_stat and PS agree
    X_stat = X
    
    # ---- if data has vertical dimension, weight in lev, sum in lev
    if(len(vert_dims) > 0):
        # data has lev; weight
        lev_weights = compute_hybrid_pressure_thickness(ps, hyai, hybi, 
                                                        dims_like=X_stat)
        lev_weights.name = "weights"
        X_weighted = X_stat.weighted(lev_weights)
        X_stat = stat_func(X_weighted, vert_dims)
    
    # ---- if data has horizontal dimensions, weight in latitude, sum in lat,lon
    if(len(horz_dims) > 0):
        if('lat' in horz_dims):
            # data has lat; weight
            lat_weights = np.cos(np.deg2rad(X.lat))
            lat_weights.name = "weights"
            X_weighted = X_stat.weighted(lat_weights)
        else:
            # data has only lon; no weighting
            X_weighted = X_stat
        # sum in horizontal
        X_stat = stat_func(X_weighted, horz_dims)
    
    return X_stat


def native_weighted_stat(X, stat, horz_weights=None, 
                         ps=None, hyai=None, hybi=None, p0=100000):
    '''
    Computes a spatially-weighted statistic of X in unstructured (e.g. cubedsphere)
    coordinates. Data values are weighted in the vertical direction by pressure 
    thickness, and horizonrtal columns are weighted by the input horz_weights. This 
    function accepts input data in a variety of shapes, and supports the following 
    use-cases:
    A: If the spatial dimensions include ('ncol'), a horizontal area-weighted 
       statistic is computed and returned. 
    B: If the spatial dimensions include ('lev'), a pressure-weighted 
       statistic is computed and returned.
    C: If the spatial dimensions include ('ncol', 'lev'), a volume-weighted 
       statistic is computed and returned.
    - If the input data X suggests use case A or C, then the optional argument
      (horz_weights) becomes required.
    - If the input data X suggests use case B or C, then the optional arguments
      (ps, hyai, hybi) become required.
    - Any extra dimensions (e.g. 'time') may also be present, though they will be
      ignored (the statistic will not be computed over them). The dimension 
      ordering does not matter.
    
    Parameters
    ----------
    X : xarray DataArray
        data on which to take the weighted statistic
    stat : string
        weigthted statistic to take. Options are 'sum' or 'mean'
    horz_weights : xarray DataArray
        1-D horizontal weights for the unstructred columns with dimension 'ncol'
        Required if X includes the dimension 'ncol'
    ps : xarray DataArray
        surface pressure with at least one dimension 'ncol'
        Required if X includes the dimension 'lev'
    hyai : xarray DataArray
        1-D hybrid A coefficients on the interface levels
        Required if X includes the dimension 'lev'
    hybi : xarray DataArray
        1-D hybrid B coefficients on the interface levels
        Required if X includes the dimension 'lev'
    p0 : float, optional
        reference pressure in Pa. Defaults to 100000 Pa.
        
    Returns
    -------
    X_stat : xarray DataArray
        Array with dimensions matching any dimensions in the input X leftover
        after taking the target statistic over spatial dimensions (e.g. time)
    '''
    
    # check that target function is valid
    available_stats=['sum', 'mean']
    if(stat not in available_stats):
        raise RuntimeError('argument \'stat\' invalid. Options are: {}'\
                           .format(available_stats))
    # lambda function to return function handle of statistic estimator
    stat_func = lambda X, d: X.sum(dim=d) if stat=='sum' else X.mean(dim=d)
    
    # check which horizontal and vertical dimensions exist in dataset
    horz_mask = [dim in ['ncol'] for dim in X.dims]
    horz_dims = np.array(X.dims)[horz_mask]
    vert_mask = [dim in ['lev'] for dim in X.dims]
    vert_dims = np.array(X.dims)[vert_mask]
    
    # enforce use cases
    if(len(horz_dims) == 0 and len(vert_dims) == 0): 
        raise RuntimeError('input data does not contain dimensions \'ncol\' or \'lev\'')
    if(len(horz_dims) != 0 and (horz_weights is None)):
        raise RuntimeError('argument \'horz_weights\' must be '\
                           'included when input data contains dimension \'ncol\'')
    if(len(vert_dims) != 0 and (ps is None or hyai is None or hybi is None)):
        raise RuntimeError('arguments \'ps\', \'hayi\', and \'hybi\' must be '\
                           'included when input data contains dimension \'lev\'')
    
    # make this assignment in case vertical weighting is skipped
    # if vertical weighting is not skipped, the statistic MUST
    # be applied in the vertical before the horizontal so that
    # dimensions of X_stat and PS agree
    X_stat = X
    
    # ---- if data has vertical dimension, weight in lev, sum in lev
    if(len(vert_dims) > 0):
        lev_weights = compute_hybrid_pressure_thickness(ps, hyai, hybi, 
                                                        dims_like=X_stat)
        lev_weights.name = "weights"
        X_weighted = X_stat.weighted(lev_weights)
        X_stat = stat_func(X_weighted, 'lev')
    
    # ---- if data has horizontal dimension, weight by column, sum in ncol
    if(len(horz_dims) > 0):
        col_weights = horz_weights
        col_weights.name = "weights"
        X_weighted = X_stat.weighted(col_weights)
        X_stat = stat_func(X_weighted, 'ncol')
    
    return X_stat


def latlon_weighted_sum(X, ps=None, hyai=None, hybi=None, p0=100000):
    '''
    Computes a volume-weighted sum of X described on a lat-lon grid. 
    See docstrings of latlon_weighted_stat().
    '''
    return latlon_weighted_stat(X, 'sum', ps, hyai, hybi, p0=p0)

def latlon_weighted_mean(X, ps=None, hyai=None, hybi=None, p0=100000):
    '''
    Computes a volume-weighted mean of X described on a lat-lon grid. 
    See docstrings of latlon_weighted_stat().
    '''
    return latlon_weighted_stat(X, 'mean', ps, hyai, hybi, p0)

def native_weighted_sum(X, horz_weights=None, 
                        ps=None, hyai=None, hybi=None, p0=100000):
    '''
    Computes a volume-weighted sum of X described on an unstructured grid. 
    See docstrings of native_weighted_stat().
    '''
    return native_weighted_stat(X, 'sum', horz_weights, ps, hyai, hybi, p0)

def native_weighted_mean(X, horz_weights=None, 
                         ps=None, hyai=None, hybi=None, p0=100000):
    '''
    Computes a volume-weighted mean of X described on an unstructured grid. 
    See docstrings of native_weighted_stat().
    '''
    return native_weighted_stat(X, 'mean', horz_weights, ps, hyai, hybi, p0)
    
    
# -------------------------------------------------------------


def compute_hybrid_pressure(ps, hyam, hybm, dims_like=None, p0=100000):
    '''
    Computes the gridpoint pressure on a hybrid pressure coordinate.
    
    Parameters
    ----------
    ps : xarray DataArray
        2- or 3-D (including time) surface pressure
    hyam : xarray DataArray
        1-D hybrid A coefficients on the model levels
    hybm : xarray DataArray
        1-D hybrid B coefficients on the model levels
    dims_like : xarray DataArray, optional
        pressure data will be transposed such that it's 
        order of dimensions (and thus shape) matches this
        template DataArray. Defaults to None, in which case
        no transpose is performed.
    p0 : float, optional
        reference pressure in Pa. Defaults to 100000 Pa.
        
    Returns
    -------
    p : xarray DataArray
        3- or 4-D pressure field, with variable name 'P'
    '''
    p = hyam * p0 + hybm * ps
    if(dims_like is not None):
        p = p.transpose(*dims_like.dims)
    p.name = 'P'
    return p


# -------------------------------------------------------------


def compute_hybrid_pressure_thickness(ps, hyai, hybi, 
                                      dims_like=None, lev=None, p0=100000):
    '''
    Computes the gridpoint pressure thickness on a hybrid pressure coordinate.
    It is expected that the interface level and model level coordinate names 
    are 'ilev' and 'lev'. Any other dimensions may be present in any order.
    
    Parameters
    ----------
    ps : xarray DataArray
        2- or 3-D (including time) surface pressure
    hyai : xarray DataArray
        1-D hybrid A coefficients on the interface levels
    hybi : xarray DataArray
        1-D hybrid B coefficients on the interface levels 
    dims_like : xarray DataArray, optional
        pressure data will be transposed such that it's 
        order of dimensions (and thus shape) matches this
        template DataArray. Defaults to None, in which case
        no transpose is performed.
    lev : xarray DataArray, optional
        1-D model pressur levels. The pressure widths found between
        each interface level pair will be assigned to these model 
        levels. This is optionally inherited from dims_like, if 
        dims_like was provided. If dims_like was not provided, this
        argument is required.
    p0 : float, optional
        reference pressure in Pa. Defaults to 100000 Pa.
        
    Returns
    -------
    dp : xarray DataArray
        3- or 4-D pressure-thickness field, with variable name 'pdel'
    '''
    
    if(dims_like is None and lev is None):
        raise RuntimeError('At least one of the following arguments must be '\
                           'specified: [dims_like, lev]')
    if(dims_like is not None and lev is not None):
        raise RuntimeError('Only one of the following arguments may be '\
                           'specified: [dims_like, lev]')
    if(dims_like is not None): 
        lev = dims_like.lev
        
    pi = (hyai * p0 + hybi * ps) # get pressure at interface levels
    pdel = pi.diff('ilev')       # take pressure difference
    pdel = pdel / 100            # scale to hPa
    
    # replace ilev coordiante with lev
    pdel['ilev'] = lev.values
    pdel = pdel.rename({'ilev':'lev'})
    
    if(dims_like is not None):
        pdel = pdel.transpose(*dims_like.dims)
    pdel.name = 'pdel'
    return pdel


# -------------------------------------------------------------
  
    
def compute_theta(T, ps, hyai, hybi, pref=100000):
    '''
    Computes the potential temperature on a hybrid pressure coordiante.
    
    Parameters
    ----------
    T : xarray DataArray
        3- or 4-D (incouding time) temperature, in K
    ps : xarray DataArray
        2- or 3-D (including time) surface pressure, in 
    hyai : xarray DataArray
        1-D hybrid A interface coefficients
    hybi : xarray DataArray
        1-D hybrid B interface coefficients
    pref : float, optional
        reference pressure for potential temperature, in Pa. 
        Defaults to 100000 Pa.
    '''
    
    p = compute_hybrid_pressure(ps, hyai, hybi, dims_like = T)
    theta = T * (pref/p)**(const.kappa)
    theta.name = 'THETA'
    return theta
    
# -------------------------------------------------------------


def compute_climatology(data, ncdf_out=None):
    '''
    Compute the monthly climatology on the data

    Parameters
    ----------
    data : xarray Dataset, xarray DataArray, or string
        An xarray object containing the data, or path to the file as a string. 
        Must include times with at least monthly resolution.
    ncdf_out : string, optional
        The file to write the resulting climatology data out to.
        Default is None, in which case nothing is written out. 
        If the file exists, read it in and return, rather than computing 
        the climatology, unless overwrite is True.
    overwrite : bool
        Whether or not to overwrite the contents of ncdf_out.
        Defaults to False, in which case the contents of the
        file area read in and returned instead of computing anything.

    Returns
    -------
        climatology : xarray data object
            The monthly climatology from the input data
    '''

    # --- check inputs
    data = check_data_inputs(data)
        
    if(ncdf_out is not None):
        if(overwrite): 
            try: os.remove(ncdf_out)
            except OSError: pass
        try: 
            climatology = xr.open_dataset(ncdf_out)
            print('Read climatology data from file {}'.format(ncdf_out.split('/')[-1]))
            return climatology
        except FileNotFoundError:
            pass

    # --- compute climatology 
    climatology = data.groupby('time.month').mean('time')
    if(ncdf_out is not None):
        climatology.to_netcdf(ncdf_out, format='NETCDF4')
    return climatology


# -------------------------------------------------------------


def compute_vorticity(data, ncdf_out=None, overwrite=False):
    '''
    Compute the vorticity from horizontal winds in the data, with derivatives
    approximated with a second-order accurate central difference scheme.

    Parameters
    ----------
    data : xarray Dataset, xarray DataArray, or string
        An xarray object containing the data, or path to the file as a string. 
        Must include times with at least monthly resolution.
    ncdf_out : string, optional
        The file to write the resulting vorticity data out to. 
        Default is None, in which case nothing is written out.

    Returns
    -------
        vorticity : xarray data object
            The vorticity from the input data
    '''
    
    # --- check inputs
    data = check_data_inputs(data)
    
    if(ncdf_out is not None):
        if(overwrite): 
            try: os.remove(ncdf_out)
            except OSError: pass
        try: 
            vort = xr.open_dataset(ncdf_out)
            print('Read vorticity data from file {}'\
                  .format(ncdf_out.split('/')[-1]))
            return vort
        except FileNotFoundError:
            pass
   
    #--- compute vorticity via MetPy; assume input is dimensionless
    u = data['U'] * units.m/units.s
    v = data['V'] * units.m/units.s
    # REMOVE MEAN HERE LATER
    vort = mc.vorticity(u, v).mean('lon')
    vort = vort.to_dataset(name='VORT')
   
    if(ncdf_out is not None):
        vort.to_netcdf(ncdf_out)
    return vort


# -------------------------------------------------------------


def QBO_index(data, p=30, u_var_name='ua', p_var_name='plev'):
    '''
    Computes the QBO index as a time series from the input data. Note, lat/lon
    dimensions are assumed to be named exactly 'lat', 'lon'
    
    Parameters
    ----------
    data : xarray Dataset, xarray DataArray, or string
        An xarray object containing the data, or path to the file as a string. 
    p : float, optional
        Pressure level at which to take the zonal-mean zonal wind, in hPa. 
        The standard choices for this index at 30 hPa (the default), and 
        50 hPa. Defaults to 30hPa. If this pressure level not present in the 
        data, it will be linearly interpolated
    u_var_name : string
        Varibale name to expect for the zonal wind. Defaults to the CMIP6
        default, which is 'ua'
    p_var_name : string
        Varibale name to expect for the pressure level. Defaults to the CMIP6
        default, which is 'plev'

    Returns
    -------
    u_qbo : DataArray
        A 1D time series of the QBO index
    '''

    data = check_data_inputs(data)

    # define lat range for meridional averaging (QBO isolation) as +- 5 deg
    latslice = slice(-5, 5)
    
    # open data, select on meridional range and average, take zonal mean
    u = data[u_var_name].sel({'lat':latslice})
    u = u.mean('lat')
    u = u.mean('lon')

    # take zonal wind at indexing pressure level (in Pa)
    p = p*100
    if(p in u['plev']):
        u_qbo = u.sel({p_var_name:3000})
    else:
        u_qbo = u.interp({p_var_name:3000})
    return u_qbo
    
    
# -------------------------------------------------------------


def Nino34_index(data, sst_var_name='tos', time_samples_mon=1):
    '''
    Computes the Nin 3.4 index as a time series from the input data. Note, 
    lat/lon dimensions are assumed to be named exactly 'lat', 'lon'
    
    Parameters
    ----------
    data : xarray Dataset, xarray DataArray, or string
        An xarray object containing the data, or path to the file as a string. 
    sst_var_name : string
        Varibale name to expect for the sea surface temperature. Defaults to 
        the CMIP6 default, which is 'tos'
    time_samples_mon : int
        Number of time sample represented in the data per month (to facilitate
        5 month rolling average). Defaults to 1, in which case the data is
        assumed to be monthly. If e.g. the data is daily, this arg should be 30

    Returns
    -------
    sst_nino : DataArray
        A 1D time series of the El Nino 3.4 index
    '''

    data = check_data_inputs(data)

    # define index averaging region as [5N-5S, 170W-120W]
    latslice = slice(-5, 5)
    lonslice = slice(120, 170)

    # take unweighted average SST in the region (unweighted since region is 
    # meridionally small)
    data_region = data.sel({'lat':latslice, 'lon':lonslice})
    sst = data_region[sst_var_name]
    sst = sst.mean('lat')
    sst = sst.mean('lon')
    
    # take 5 month running average
    sst_nino = sst.rolling(time=time_samples_mon).mean()
    return sst_nino
    

# -------------------------------------------------------------


def ssw_events(dat, lat=60, lev=10):
    '''
    Returns SSW events in a dataset, defined as sign flips of the zonal-mean 
    zonal wind at lat,lev

    Parameters
    ----------
    data : xarray Dataset, xarray DataArray, or string
        An xarray object containing the data, or path to the file as a string. 
        Must contain at least the variable 'U'
    lat : float, optional
        Latitude at which to take the zonal-mean wind measurement in degrees. 
        Defaults to 60 degrees, which is the latitude used for SSW 
        classification by the WMO
    lev : float, optional
        Pressure at which to take the zonal-mean wind measurement in hPa.
        Defaults to 10hPa, which is the pressure used for SSW classification 
        by the WMO

    Returns
    -------
    ssw_data : xarray Dataset or DataArray
        The zonal-mean data, selected on the provided lat,lev positions, and 
        filtered for times when the zonal-mean zonal wind U<0. If the length 
        of any of the field varibales in this returned dataset is zero, there 
        were no SSWs present in the input as defined for this lat,lev
    '''
    
    # check inputs
    data = check_data_inputs(data)

    ssw_data = data.sel({'lat':lat, 'lev':lev}, method='nearest')
    ssw_data = ssw_data.mean('lon')
    ssw_data = data.where(data['U'] < 0, drop=True)
    return ssw_data
    

# -------------------------------------------------------------
  
    
def lp_norm(q, qT, p, weighted=True, horz_weights=None, 
            ps=None, hyai=None, hybi=None, p0=100000):
    '''
    Computes the discrete l-p norm of the error between an estaimte q and 
    exact (or reference) solution qT. Matches the form presented in
    Whitehead et al. 2015 (https://doi.org/10.1002/qj.2389) (hereafter W15).
    The data can be input on either a lat-lon grid (in which case the expected
    horizontal dimensions are ('lat', 'lon'), or an unstructured grid, in 
    which case the expected horizontal dimension is ('ncol'). 
    It is optional whether or not the data includes a vertical coordinate. If
    taking the lp norm on e.g. an isentropic surface, the data will be 2D in 
    space, while if taking the lp norm over a volume, the data will be 3D in
    space. If the latter, then the expected vertical dimension is ('lev'). 
    - If a vertical dimension ('lev') is present, then the optonal arguments 
      (ps, hyai, hybi) become required. 
    - If an unstructured horizontal dimension ('ncol') is present, then the 
      optional argument (horz_weights) becomes required.
    - Any extra dimensions in addition to those stated above (e.g. time) are 
      allowed. The order of the dimensions does not matter.
    
    Parameters
    ----------
    q : xarray DataArray
        the solution estimate
    qT : xarray DataArray
        the exact or reference solution. Dimensionality must match q
    p : int
        degree of the error norm. p=2 is least squares regression, and
        p=inf is the maximum error norm. p=4 is an interpolant of 
        p=2 and p=inf (see W15 section 2.2.2)
    weighted : bool
        whether or not to spatially weight the data sum
    horz_weights : xarray DataArray, optional
        1-D horizontal weights for the unstructred columns with dimension 'ncol'
        Required if X includes the dimension 'ncol' and weighted=True
    ps : xarray DataArray, optional
        surface pressure. Dimensionality must match horizontal dimensions of q 
    hyai : xarray DataArray
        1-D hybrid A interface coefficients
    hybi : xarray DataArray
        1-D hybrid B interface coefficients
    pref : float, optional
        reference pressure for potential temperature, in Pa. 
        Defaults to 100000 Pa.
        
    Returns
    -------
    The total l-p error of q. If q, qT are only spatial datasets, a single
    number is returned. If a 'time' dimension is present, a time series is
    returned. In either case, the return type is an xarray DataArray.    
    '''
    
    # get variable diff
    qdiff = q - qT
    
    # check which horizontal and vertical dimensions exist in dataset
    horz_mask = [dim in ['lat', 'lon', 'ncol'] for dim in q.dims]
    horz_dims = np.array(q.dims)[horz_mask]
    vert_mask = [dim in ['lev'] for dim in q.dims]
    vert_dims = np.array(q.dims)[vert_mask]
    
    # enforce use cases
    if((len(horz_dims) == 0 or len(horz_dims) >= 3) and len(vert_dims) != 1):
        raise RuntimeError('expected horizontal dimensions (\'lat\', \'lon\') '\
                           'or (\'ncol\') and/or vertical dimension (\'lev\')')
    if('ncol' in horz_dims and (horz_weights is None)):
        raise RuntimeError('argument \'horz_weights\' must be '\
                           'included when input data contains dimension \'ncol\'')
    if(len(vert_dims) > 0 and (ps is None or hyai is None or hybi is None)):
        raise RuntimeError('arguments \'ps\', \'hayi\', and \'hybi\' must be '\
                           'included when input data contains dimension \'lev\'')
        
    # use either unstructured or lat-lon weighting routines
    if(weighted):
        sum_args = []
        if('ncol' in horz_dims):
            sum_func = native_weighted_sum
            sum_args.append(horz_weights)
        elif('lat' in horz_dims):
            sum_func = latlon_weighted_sum
        if('lev' in vert_dims):
            sum_args.extend([ps, hyai, hybi, p0])
            
        # the "weighted sum" is a sum that weights gridpoints in the
        # vetical (by pressure thickness) and meridional (by cos(lat)) directions
        num = sum_func(qdiff, *sum_args)
        den = sum_func(qT, *sum_args)
    else:
        num = (qdiff).sum()
        den = (qT).sum() 
    
    if(p == float('inf')):
        lpq = np.max(np.abs(num)) / np.max(np.abs(den))
    else:
        lpq = (num**p/den**p)**(1/p)
    
    lpq.name = 'l{}_norm'.format(p)
    return lpq


# -------------------------------------------------------------


def ensemble_mean(ensemble, std=False, incl_vars=None, outFile=None, overwrite=False):
    '''
    Returns the mean of an ensemble for all variables included in the input 
    ensemble members. Each member is expected as an xarray DataSet, and must
    have the same dimensions and coordinates.

    Parameters
    ----------
    ens : list of datasets
        The ensemble members. Each entry can be a xarray Dataset, Datarray, 
        or string giving a file location
    std : bool, optional
        Whether or not to compute and also return the standard deviation of 
        the ensemble members,in addition to the mean.Defaults to False, in 
        which case only the mean will be computed and
        returned.
    incl_vars : list of strings, optional
        A list of variables to include in the ensemble mean, in the case that
        it is not desired to include all varibales present, or if all members
        variables match only for this list. Defaults to None, in which case all
        ensemble members will be expected to have identical variables, and all
        uppercase variables will be included in the mean computation.
    outFile : str, optional
        File location to write out the ensemble mean and std files in netcdf
        format.Default is None, in which case nothing is written out, and the
        ensemble mean and std is returned to the caller. If this file exists
        already, it will be read in, and the data returned, rather than
        computing anything. If std==True and outFile is provided, a second file
        will also be written with an extension '.std' to contain this
        information.
    overwrite : bool
        whether or not to force a recalculation of the ensemble mean, deleting
        any netcdf files previously written by this function at outFile. 
        Defaults to False, in which case the result is read from file if
        already previously computed and written out.

    Returns
    -------
    ensembleMean : xarray Dataset or list of xarray Datasets
        If std==False, the ensemble mean of the input ensemble members
        If std==True, a list of two elements. Item at position 0 is the
            ensemble mean of the inpit ensemble membersm, item at position 1 
            is the standard deviation of the ensemble members
    '''
    
    # ---- read mean,std from file if exists and return
    if(outFile is not None):
        outFile_std = '{}.std'.format(outFile)
        if(overwrite): 
            try: os.remove(outFile)
            except OSError: pass
            try: os.remove(outFile_std)
            except OSError: pass
        try: 
            ensMean = xr.open_dataset(outFile)
            print('---Read data from file {}'.format(outFile.split('/')[-1]))
            if(std):
                ensStd = xr.open_dataset(outFile_std)
                print('--- Read data from file {}'\
                      .format(outFile_std.split('/')[-1]))
                return [ensMean, ensStd]
            else:
                return ensMean
        except FileNotFoundError:
            pass
    
    # ---- check input
    pdb.set_trace()
    N = len(ensemble)
    ens = np.empty(N, dtype=object)
    ens[0] = check_data_inputs(ensemble[0])
    if(incl_vars is None):
        all_vars = sorted(list(ens[0].keys()))
        all_vars = [l.isupper() for l in all_vars]
    else:
        all_vars = incl_vars
    
    for i in range(N-1):
        print('opening ens {}...'.format(i), end='\r', flush=True)
        ens[i+1] = check_data_inputs(ensemble[i+1])
        assert np.sum([vv in sorted(list(ens[i+1].keys())) \
                       for vv in all_vars]) == len(all_vars),\
                       'not all ensemble members contain these'\
                       'variables:\n{}'.format(all_vars)
    pdb.set_trace()

    # ---- sum members
    ensMean = ens[0]
    if(std): rms = ens[0]**2
    for i in range(len(ens)-1):
        print('--- summing ensemble {}...'.format(i))
        ensMean = xr.concat([ensMean, ens[i+1]], dim='ens', data_vars=all_vars,
                             compat='override', coords='minimal')
        ensMean = ensMean.sum('ens')
        if(std):
            pdb.set_trace()
            rms = rms + ens[i+1]**2

    print('---taking ensemble mean...')
    ensMean = ensMean / len(ens)
    if(std): ensStd = np.sqrt( rms/len(ens) - ensMean**2)

    # ---- write out, return
    if(outFile is not None):
        ensMean.to_netcdf(outFile)
        print('--- Wrote ensemble data to file {}'\
              .format(outFile.split('/')[-1]))
        if(std):
            ensStd.to_netcdf(outFile_std)
            print('--- Wrote ensemble std data to '\
                  'file {}'.format(outFile.split('/')[-1]))
    if(std):    
        return ensMean, ensStd
    else:
        return ensMean


# -------------------------------------------------------------


def concat_run_outputs(run, sel={}, mean=[], outFile=None, overwrite=False, 
                       histnum=0, regridded=True, component='cam'):
    '''
    Concatenates netCDF data along the time dimension. Intended to be used to
    combine outputs of a single run.

    Parameters
    ----------
    run : str list
        CIME run directory contining output to be concatenated. It is assumed
        that the contents of this directory is from a run of a model through
        CIME that generated multiple history files of the same number (e.g.
        there are >1 h0 files). Select varaiables are read from the history
        files and concatenated across time
    sel : dict
        indexers to pass to data.sel(), where data is each history file opened
        via xarray
    mean : list
        list of dimensions along which to take the mean
    outFile : str
        File location to write out the concatenated file in netcdf format.
        Default is None, in which case nothing is written out, and the concatenated data 
        is returned to the caller. If this file exists already, it will be 
        read in, and the data returned, rather than any concatenation actually
        being performed.
    overwrite : bool
        whether or not to force a recalculation of the reduced data, deleting
        any netcdf files previously written by this function at outFile. 
        Defaults to False, in which case the result is read from file if
        already previously computed and written out.
    histnum : int
        hsitory file group to concatenate. Defaults to 0, in which case h0
        files willbe targeted.
    regridded : bool
        Whether or not to look for regridded versions of each history file
        (i.e. ones for which the substring "regrid" appears after the header
        number "h0"). Defaults to True.
    component : string
        Concatenate data for this component (will look for files with
        [component].h[histnum]). Defaults to 'cam'
    '''
   
    run_name = run.rstrip('/').split('/')[-2]
 
    print('\n\n========== Concatenating data at {} =========='.format(run_name))
  
    if(regridded):
        hist = sorted(glob.glob('{}/*{}.h{}*regrid*.nc'\
                                .format(run, component, histnum)))
    else:
        hist = sorted(glob.glob('{}/*{}.h{}*0.nc'\
                                .format(run, component, histnum)))
    
    # remove outFile from glob results, if present
    try: hist.remove(outFile)
    except ValueError: pass
    hist = np.array(hist)
    assert len(hist) > 0, 'no history files found at {} for regridded={}'.format(run, regridded)
    
    print('Found {} files for history group h{}'.format(len(hist), histnum))
    
    # --- read concatenation from file if exists
    if(outFile is not None):
        if(overwrite): 
            try: os.remove(outFile)
            except OSError: pass
        try: 
            allDat = xr.open_dataset(outFile)
            print('Read data from file {}'.format(outFile.split('/')[-1]))
            return allDat
        except FileNotFoundError:
            pass
        
    # ---- concatenate
    allDat_set = False
    for j in range(len(hist)):
        print('------ working on file {} ------'\
              .format(hist[j].split('/')[-1]))
        skip=False

        # open dataset, select data
        # (temporary catch because of annoying xarray warning)
        with warnings.catch_warnings():  
            warnings.simplefilter('ignore', FutureWarning)
            dat = xr.open_dataset(hist[j])
            
            # ------ sel
            for dim in sel.keys():
                try:
                    dat = dat.sel({dim:sel[dim]}, method='nearest')
                except NotImplementedError:
                    dat = dat.sel({dim:sel[dim]})
                # if all data was removed after the slice, continue
                # !!! generalize this at some point, for data that has no U
                if(len(dat['U']) == 0):
                    skip=True
                    break
            if(skip==True): continue

        if(not allDat_set): 
            allDat = dat
            allDat_set = True
        else: 
            allDat = xr.concat([allDat, dat], 'time')
  
    # ------ take means
    for dim in mean:
        allDat = allDat.mean(dim)
   
    # ------ done; write out and return
    if(outFile is not None):
        allDat.to_netcdf(outFile)
        print('Wrote data to file {}'.format(outFile.split('/')[-1]))

    return allDat
        

# -----------------------------------------------------------------
