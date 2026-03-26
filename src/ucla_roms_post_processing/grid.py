"""Process ROMS grid files and add grid variables to output files."""

from typing import Literal

import numpy as np
import xarray as xr
import xgcm


def add_s_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Construct and add the s-coordinates."""
    s_w = np.linspace(-1, 0, ds.s_w.size, endpoint=True)
    s_rho = s_w[:-1] + np.diff(s_w) / 2
    ds = ds.assign_coords(s_w=s_w, s_rho=s_rho)
    return ds


def add_grid_stretching(ds: xr.Dataset, Vtransform: Literal[1, 2] = 2) -> xr.Dataset:
    """Extract the grid stretching out of attrs and add to dataset."""
    ds["Cs_r"] = ("s_rho", ds.attrs.pop("Cs_r"))
    ds["Cs_w"] = ("s_w", ds.attrs.pop("Cs_w"))
    ds["Vtransform"] = Vtransform
    return ds

def compute_z_rho(ds: xr.Dataset) -> xr.DataArray:
    """Compute z_rho from s-coordinates and grid stretching."""
    if "z_rho" in ds:
        return ds["z_rho"]
    hc = ds.attrs["hc"]
    S = (ds["s_rho"] * hc + ds["Cs_r"] * ds["h"]) / (hc + ds["h"])
    return ds["zeta"] + S * (ds["zeta"] + ds["h"])

def compute_z_w(ds: xr.Dataset) -> xr.DataArray:
    """Compute z_w from s-coordinates and grid stretching."""
    if "z_w" in ds:
        return ds["z_w"]
    hc = ds.attrs["hc"]
    S = (ds["s_w"] * hc + ds["Cs_w"] * ds["h"]) / (hc + ds["h"])
    return ds["zeta"] + S * (ds["zeta"] + ds["h"])

def add_z_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Add z_rho and z_w coordinates to the dataset."""
    ds["z_rho"] = compute_z_rho(ds)
    ds["z_w"] = compute_z_w(ds)
    return ds

def create_xgrid(ds: xr.Dataset) -> xgcm.Grid:
    """Create am xgcm grid."""
    return xgcm.Grid(
        ds,
        coords={
            "X": {"center": "xi_rho", "inner": "xi_u"},
            "Y": {"center": "eta_rho", "inner": "eta_v"},
            "Z": {"center": "s_rho", "outer": "s_w"}
        },
        autoparse_metadata=False
    )