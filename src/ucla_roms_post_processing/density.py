"""Compute density, potential density and buoyancy from T and S.

This code is based on the implementation in xroms https://github.com/xoceanmodel/xroms/blob/main/xroms/roms_seawater.py
"""

import numpy as np
import numpy.typing as npt
import xarray as xr

from .grid import compute_z_rho

DEFAULT_RHO0 = 1025.0

def compute_density(
    temp: npt.NDArray[float] | xr.DataArray,
    salt: npt.NDArray[float] | xr.DataArray,
    z: npt.NDArray[float] | xr.DataArray | float = 0.0,
) -> xr.DataArray:
    """Calculate the density [kg/m^3] as calculated in ROMS.

    Parameters
    ----------
    temp: DataArray, ndarray
        Temperature [Celsius]
    salt: DataArray, ndarray
        Salinity
    z: DataArray, ndarray, int, float, optional
        Vertical coordinate array or reference value [m] (Default: 0.0).

    Notes
    -----
    Equation of state based on ROMS Nonlinear/rho_eos.F

    """

    A00 = +19092.56
    A01 = +209.8925
    A02 = -3.041638
    A03 = -1.852732e-3
    A04 = -1.361629e-5
    B00 = +104.4077
    B01 = -6.500517
    B02 = +0.1553190
    B03 = +2.326469e-4
    D00 = -5.587545
    D01 = +0.7390729
    D02 = -1.909078e-2
    E00 = +4.721788e-1
    E01 = +1.028859e-2
    E02 = -2.512549e-4
    E03 = -5.939910e-7
    F00 = -1.571896e-2
    F01 = -2.598241e-4
    F02 = +7.267926e-6
    G00 = +2.042967e-3
    G01 = +1.045941e-5
    G02 = -5.782165e-10
    G03 = +1.296821e-7
    H00 = -2.595994e-7
    H01 = -1.248266e-9
    H02 = -3.508914e-9
    Q00 = +999.842594
    Q01 = +6.793952e-2
    Q02 = -9.095290e-3
    Q03 = +1.001685e-4
    Q04 = -1.120083e-6
    Q05 = +6.536332e-9
    U00 = +0.824493e0
    U01 = -4.08990e-3
    U02 = +7.64380e-5
    U03 = -8.24670e-7
    U04 = +5.38750e-9
    V00 = -5.72466e-3
    V01 = +1.02270e-4
    V02 = -1.65460e-6
    W00 = +4.8314e-4

    sqrtS = np.sqrt(salt)
    den1 = (
        Q00
        + Q01 * temp
        + Q02 * temp**2
        + Q03 * temp**3
        + Q04 * temp**4
        + Q05 * temp**5
        + U00 * salt
        + U01 * salt * temp
        + U02 * salt * temp**2
        + U03 * salt * temp**3
        + U04 * salt * temp**4
        + V00 * salt * sqrtS
        + V01 * salt * sqrtS * temp
        + V02 * salt * sqrtS * temp**2
        + W00 * salt**2
    )
    K0 = (
        A00
        + A01 * temp
        + A02 * temp**2
        + A03 * temp**3
        + A04 * temp**4
        + B00 * salt
        + B01 * salt * temp
        + B02 * salt * temp**2
        + B03 * salt * temp**3
        + D00 * salt * sqrtS
        + D01 * salt * sqrtS * temp
        + D02 * salt * sqrtS * temp**2
    )
    K1 = (
        E00
        + E01 * temp
        + E02 * temp**2
        + E03 * temp**3
        + F00 * salt
        + F01 * salt * temp
        + F02 * salt * temp**2
        + G00 * salt * sqrtS
    )
    K2 = (
        G01
        + G02 * temp
        + G03 * temp**2
        + H00 * salt
        + H01 * salt * temp
        + H02 * salt * temp**2
    )
    bulk = K0 - K1 * z + K2 * z**2
    return (den1 * bulk) / (bulk + 0.1 * z)

def compute_potential_density(
    temp: npt.NDArray[float] | xr.DataArray,
    salt: npt.NDArray[float] | xr.DataArray,
    *,
    zref: float = 0.0,
    rho0: float = DEFAULT_RHO0
) -> xr.DataArray:
    """Calculate the potential density [kg/m^3] as calculated in ROMS.

    Parameters
    ----------
    temp: DataArray, ndarray
        Temperature [Celsius]
    salt: DataArray, ndarray
        Salinity
    zref: float, optional
        Vertical coordinate reference value [m] (Default: 0.0).
    rho0: float, optional
        Reference density [kg/m^3] (Default: 1025.0).
    Notes
    -----
    Equation of state based on ROMS Nonlinear/rho_eos.F

    """
    return compute_density(temp, salt, zref) - rho0

def compute_buoyancy(
    temp: npt.NDArray[float] | xr.DataArray,
    salt: npt.NDArray[float] | xr.DataArray,
    *,
    zref: float = 0.0,
    rho0: float = DEFAULT_RHO0
) -> xr.DataArray:
    """Calculate the buoyancy [m/s^2] as calculated in ROMS.

    Parameters
    ----------
    temp: DataArray, ndarray
        Temperature [Celsius]
    salt: DataArray, ndarray
        Salinity
    zref: float, optional
        Vertical coordinate reference value [m] (Default: 0.0).
    rho0: float, optional
        Reference density [kg/m^3] (Default: 1025.0).

    Notes
    -----
    Equation of state based on ROMS Nonlinear/rho_eos.F

    """
    g = 9.81
    return -compute_potential_density(temp, salt, zref=zref, rho0=rho0) * g / rho0

def add_density(ds: xr.Dataset) -> xr.Dataset:
    """Add density variable to the dataset."""
    z = compute_z_rho(ds)
    ds["density"] = xr.apply_ufunc(
        compute_density,
        ds["temp"], 
        ds["salt"], 
        z,
        input_core_dims=[[], [], []],
        output_core_dims=[[]],
        dask="allowed"
    )
    return ds

def add_potential_density(ds: xr.Dataset, zref: float = 0.0, rho0: float | None = None) -> xr.Dataset:
    """Add potential density variable to the dataset.
    
        If rho0 is not provided, it will be taken from the dataset attributes or default to 1025.0.
    """
    if rho0 is None:
        rho0 = ds.attrs.get("rho0", DEFAULT_RHO0)
    ds["potential_density"] = xr.apply_ufunc(
        compute_potential_density,
        ds["temp"], 
        ds["salt"], 
        input_core_dims=[[], []],
        output_core_dims=[[]],
        dask="allowed",
        kwargs={"zref": zref, "rho0": rho0})
    return ds

def add_buoyancy(ds: xr.Dataset, zref: float = 0.0, rho0: float | None = None) -> xr.Dataset:
    """Add buoyancy variable to the dataset.
    
        If rho0 is not provided, it will be taken from the dataset attributes or default to 1025.0.
    """
    if rho0 is None:
        rho0 = ds.attrs.get("rho0", DEFAULT_RHO0)
    ds["buoyancy"] = xr.apply_ufunc(
        compute_buoyancy,
        ds["temp"],
        ds["salt"],
        input_core_dims=[[], []],
        output_core_dims=[[]],
        dask="allowed",
        kwargs={"zref": zref, "rho0": rho0}
    )
    return ds