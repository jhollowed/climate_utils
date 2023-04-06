'''
This module provides various functions for computing various climate metrics, indices, and
performing other common tasks on climate or weather data
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

# ==========================================================================================

def check_data_inputs(data):
    acceptable_types = [xr.core.dataset.Dataset, xr.core.dataarray.DataArray, str, np.str_]
    if(type(data) not in acceptable_types):
        raise RuntimeError('data must be provided as an xarray Dataset, DataArray, '\
                           'or a string to the file')
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
    Converts pressure to geootential height assuming an isothermal atmosphere

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
    Converts geootential height to pressure assuming an isothermal atmosphere

    Parameters
    ----------
    z : float or flaot array
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
            print('Read vorticity data from file {}'.format(ncdf_out.split('/')[-1]))
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
        Pressure level at which to take the zonal-mean zonal wind, in hPa. The standard
        choices for this index at 30 hPa (the default), and 50 hPa. Defaults to 30hPa. If
        this pressure level not present in the data, it will be linearly interpolated
    u_var_name : string
        Varibale name to expect for the zonal wind. Defaults to the CMIP6 default, which is 'ua'
    p_var_name : string
        Varibale name to expect for the pressure level. Defaults to the CMIP6 default, 
        which is 'plev'

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
    Computes the Nin 3.4 index as a time series from the input data. Note, lat/lon 
    dimensions are assumed to be named exactly 'lat', 'lon'
    
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

    # take unweighted average SST in the region (unweighted since region is meridionally small)
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
    Returns SSW events in a dataset, defined as sign flips of the zonal-mean zonal wind
    at lat,lev

    Parameters
    ----------
    data : xarray Dataset, xarray DataArray, or string
        An xarray object containing the data, or path to the file as a string. 
        Must contain at least the variable 'U'
    lat : float, optional
        Latitude at which to take the zonal-mean wind measurement in degrees. 
        Defaults to 60 degrees, which is the latitude used for SSW classification by the WMO
    lev : float, optional
        Pressure at which to take the zonal-mean wind measurement in hPa.
        Defaults to 10hPa, which is the pressure used for SSW classification by the WMO

    Returns
    -------
    ssw_data : xarray Dataset or DataArray
        The zonal-mean data, selected on the provided lat,lev positions, and filtered for times 
        when the zonal-mean zonal wind U<0. If the length of any of the field varibales in this
        returned dataset is zero, there were no SSWs present in the input as defined for this 
        lat,lev
    '''
    
    # check inputs
    data = check_data_inputs(data)

    ssw_data = data.sel({'lat':lat, 'lev':lev}, method='nearest')
    ssw_data = ssw_data.mean('lon')
    ssw_data = data.where(data['U'] < 0, drop=True)
    return ssw_data
    

# -------------------------------------------------------------
    
    
def l2_norm(dat, varname='U', norm_type=15, compdat=None, compMean=False):
    '''
    Computes l2 norms from Jablonowski+Williamson 2006

    Parameters
    ----------
    dat : xarray Dataset
        the input data
    varname : string
        Variable to take the norm of
    norm_type: int
        Either 14 or 15, giving the equation number from JW06
        - Default is 15, in which case the norm is taken between the 
        zonal mean of the field, and the initial zonal mean of the 
        field (this measures the departure from stability)
        - If 14, the norm is taken between the zonal mean of the field, 
        and the 3D field (this measures the departure from symmetry)
    compdat : xarray Dataset
        Another set of input data to compare the input against.
        Default is None, in which case the norm is taken as described
        in Jablonowski+06, respecting norm_type. 
        If provided, norm is taken between the entire 3D fields of each
        file as a time series (i.e. norm_type will be ignored)
        It will be assumed that the comparison data set is on the same 
        horizontal and vertical grid as the input (horizontal only if looking
        at 2D quantities), and output at the same timesteps
    compMean : bool
        If True, and compdat is given, copare the zonally averaged fields
        from each dataset, rather than the 3D field
    '''

    var = dat[varname]
    u = dat['U']
    ps = dat['PS']
    hyai = dat['hyai']
    hybi = dat['hybi']
    P0 = 100000.0         #Pa, to match ps units
    dp = dpres_hybrid_ccm(ps, P0, hyai, hybi).values
    dp = xr.DataArray(dp, dims=u.dims, coords=u.coords, name='dp')
    
    # approx horizontal weighting
    rad = np.pi/180
    lat = dat['lat']
    wy  = np.cos(lat*rad) #proxy [ sin(j+1/2)-sin(j-1/2) ]

    # get time samples and zonal-mean var
    ntime = len(dat['time'])
    varm = var.mean('lon')

    # read comparison dataset if given
    if(compdat is not None):
        norm_type = 0
        compvar = compdat[varname]
        comp_varm = var.mean('lon')
        
    # compute the norm...
    
    # =============== EQ 14 ===============
    if(norm_type == 14):
        
        norm = np.zeros(ntime)
        for i in range(ntime):
            # the broadcasting here works by matching dimensions by their names, 
            # which importantly comes from the input netcdf file
            diff2 = (var[i] - varm[i])**2
            num = np.sum(diff2 * wy * dp[i])
            den = np.sum(wy * dp[i])
            norm[i] = np.sqrt(num/den)    
        
    # =============== EQ 14 ===============
    elif(norm_type == 15):
        
        norm = np.zeros(ntime)
        for i in range(ntime):
            # the broadcasting here works by matching dimensions by their names, 
            # which importantly comes from the input netcdf file
            diff2 = (varm[i] - varm[0])**2
            num = np.sum(diff2 * wy * dp[i])
            den = np.sum(wy * dp[i])
            norm[i] = np.sqrt(num/den)
    
    # =============== COMPARE ===============
    elif(norm_type == 0):
        
        norm = np.zeros(ntime)
        for i in range(ntime):
            # the broadcasting here works by matching dimensions by their names, 
            # which importantly comes from the input netcdf file
            if(compMean):
                diff2 = (varm[i] - compvarm[i])**2
            else:
                diff2 = (var[i] - compvar[i])**2
            num = np.sum(diff2 * wy * dp[i])
            den = np.sum(wy * dp[i])
            norm[i] = np.sqrt(num/den)
    
    return norm


# -------------------------------------------------------------


def ensemble_mean(ensemble, std=False, incl_vars=None, outFile=None, overwrite=False):
    '''
    Returns the mean of an ensemble for all variables included in the input ensemble members.
    Each member is expected as an xarray DataSet, and must have the same dimensions and 
    coordinates.

    Parameters
    ----------
    ens : list of datasets
        The ensemble members. Each entry can be a xarray Dataset, Datarray, or string giving 
        a file location
    std : bool, optional
        Whether or not to compute and also return the standard deviation of the ensemble 
        members,in addition to the mean.
        Defaults to False, in which case only the mean will be computed and returned.
    incl_vars : list of strings, optional
        A list of variables to include in the ensemble mean, in the case that it is not 
        desired to include all varibales present, or if all members variables match only for 
        this list. Defaults to None, in which case all ensemble members will be expected to 
        have identical variables, and all uppercase variables will be included in the mean
        computation.
    outFile : str, optional
        File location to write out the ensemble mean and std files in netcdf format.
        Default is None, in which case nothing is written out, and the ensemble mean and std
        is returned to the caller. If this file exists already, it will be read in, and the 
        data returned, rather than computing anything.
        If std==True and outFile is provided, a second file will also be written with an 
        extension '.std' to contain this information.
    overwrite : bool
        whether or not to force a recalculation of the ensemble mean, deleting any netcdf 
        files previously written by this function at outFile. 
        Defaults to False, in which case the result is read from file if already previously 
        computed and written out.

    Returns
    -------
    ensembleMean : xarray Dataset or list of xarray Datasets
        If std==False, the ensemble mean of the input ensemble members
        If std==True, a list of two elements. Item at position 0 is the ensemble
        mean of the inpit ensemble membersm, item at position 1 is the standard 
        deviation of the ensemble members
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
                print('--- Read data from file {}'.format(outFile_std.split('/')[-1]))
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
        assert np.sum([vv in sorted(list(ens[i+1].keys())) for vv in all_vars]) == \
                                                                  len(all_vars),   \
                'not all ensemble members contain these variables:\n{}'.format(all_vars)
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
        print('--- Wrote ensemble data to file {}'.format(outFile.split('/')[-1]))
        if(std):
            ensStd.to_netcdf(outFile_std)
            print('--- Wrote ensemble std data to file {}'.format(outFile.split('/')[-1]))
    if(std):    
        return ensMean, ensStd
    else:
        return ensMean
    


# -------------------------------------------------------------


def reduce_netcdf(runs, g):
    return


# -------------------------------------------------------------


def concat_run_outputs(run, sel={}, mean=[], outFile=None, overwrite=False, 
                       histnum=0, regridded=True, component='cam'):
    '''
    Concatenates netCDF data along the time dimension. Intended to be used to combine outputs of
    a single run.

    Parameters
    ----------
    run : str list
        CIME run directory contining output to be concatenated. It is assumed
        that the contents of this directory is from a run of a model through CIME
        that generated multiple history files of the same number (e.g. there are >1
        h0 files). Select varaiables are read from the history files and concatenated 
        across time
    sel : dict
        indexers to pass to data.sel(), where data is each history file opened via xarray
    mean : list
        list of dimensions along which to take the mean
    outFile : str
        File location to write out the concatenated file in netcdf format.
        Default is None, in which case nothing is written out, and the concatenated data 
        is returned to the caller. If this file exists already, it will be read in, and 
        the data returned, rather than any concatenation actually being performed.
    overwrite : bool
        whether or not to force a recalculation of the reduced data, deleting any netcdf 
        files previously written by this function at outFile. 
        Defaults to False, in which case the result is read from file if already previously 
        computed and written out.
    histnum : int
        hsitory file group to concatenate. Defaults to 0, in which case h0 files will
        be targeted.
    regridded : bool
        Whether or not to look for regridded versions of each history file (i.e. ones for 
        which the substring "regrid" appears after the header number "h0"). 
        Defaults to True
    component : string
        Concatenate data for this component (will look for files with [component].h[histnum]).
        Defaults to 'cam'
    '''
   
    run_name = run.rstrip('/').split('/')[-2]
 
    print('\n\n========== Concatenating data at {} =========='.format(run_name))
  
    if(regridded):
        hist = sorted(glob.glob('{}/*{}.h{}*regrid*.nc'.format(run, component, histnum)))
    else:
        hist = sorted(glob.glob('{}/*{}.h{}*0.nc'.format(run, component, histnum)))
    
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
        print('---------- working on file {} ----------'.format(hist[j].split('/')[-1]))
        skip=False

        # open dataset, select data
        with warnings.catch_warnings():  # temporary because of annoying xarray warning
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
