"""Compute the potential vorticity from the velocity and density fields."""

import xarray as xr
import xgcm

from .grid import compute_z_w, create_xgrid


def _compute_dv_dxi(ds: xr.Dataset, xgrid: xgcm.Grid) -> xr.DataArray:
    """Compute the derivative of v with respect to xi."""
    dv_dxi = xgrid.diff(ds.v, "X")
    return xgrid.interp(dv_dxi, ("X", "Y"))


def _compute_du_deta(ds: xr.Dataset, xgrid: xgcm.Grid) -> xr.DataArray:
    """Compute the derivative of u with respect to eta."""
    du_deta = xgrid.diff(ds.u, "Y")
    return xgrid.interp(du_deta, ("X", "Y"))


def _compute_dv_dpi(ds: xr.Dataset, xgrid: xgcm.Grid) -> xr.DataArray:
    """Compute the derivative of v with respect to pi."""
    dv_dpi = xgrid.diff(ds.v, "Z")
    return xgrid.interp(dv_dpi, ("Z", "Y"))


def _compute_du_dpi(ds: xr.Dataset, xgrid: xgcm.Grid) -> xr.DataArray:
    """Compute the derivative of u with respect to pi."""
    du_dpi = xgrid.diff(ds.u, "Z")
    return xgrid.interp(du_dpi, ("Z", "X"))


def _compute_dtracer_dxi(
    ds: xr.Dataset, xgrid: xgcm.Grid, tracer: str = "b"
) -> xr.DataArray:
    """Compute the derivative of a tracer with respect to xi."""
    dtracer_dxi = xgrid.diff(ds[tracer], "X")
    return xgrid.interp(dtracer_dxi, "X")


def _compute_dtracer_deta(
    ds: xr.Dataset, xgrid: xgcm.Grid, tracer: str = "b"
) -> xr.DataArray:
    """Compute the derivative of a tracer with respect to eta."""
    dtracer_deta = xgrid.diff(ds[tracer], "Y")
    return xgrid.interp(dtracer_deta, "Y")


def _compute_dtracer_dpi(
    ds: xr.Dataset, xgrid: xgcm.Grid, tracer: str = "b"
) -> xr.DataArray:
    """Compute the derivative of a tracer with respect to pi."""
    dtracer_dpi = xgrid.diff(ds[tracer], "Z")
    return xgrid.interp(dtracer_dpi, "Z")


def _compute_dz(ds: xr.Dataset, xgrid: xgcm.Grid) -> xr.DataArray:
    """Compute the vertical grid spacing."""
    z = compute_z_w(ds)
    return xgrid.diff(z, "Z")

def _compute_vertical_vorticity(ds: xr.Dataset, xgrid: xgcm.Grid, tracer: str = "b") -> xr.DataArray:
    """Compute the vertical component of absolute vorticity."""
    dv_dxi = _compute_dv_dxi(ds, xgrid)
    du_deta = _compute_du_deta(ds, xgrid)
    return dv_dxi * ds["pm"] - du_deta * ds["pn"] + ds["f"]



def compute_potential_vorticity(
    ds: xr.Dataset,
    *,
    tracer: str = "b",
    xgrid: xgcm.Grid | None = None,
    scaling: float = 1.0,
) -> xr.DataArray:
    """Compute the potential vorticity."""
    if xgrid is None:
        xgrid = create_xgrid(ds)
    # first the vertical component of absolute vorticity times the vertical gradient of the tracer
    pv = _compute_vertical_vorticity(ds, xgrid, tracer)
    dtracer_dpi = _compute_dtracer_dpi(ds, xgrid, tracer)
    pv *= dtracer_dpi
    del dtracer_dpi

    # then the vertical shear of the zonal velocity times the meridional gradient of the tracer
    du_dpi = _compute_du_dpi(ds, xgrid)
    dtracer_deta = _compute_dtracer_deta(ds, xgrid, tracer)
    dtracer_deta *= ds["pn"] * du_dpi
    pv += dtracer_deta
    del dtracer_deta, du_dpi

    # finally the vertical shear of the meridional velocity times the zonal gradient of the tracer
    dv_dpi = _compute_dv_dpi(ds, xgrid)
    dtracer_dxi = _compute_dtracer_dxi(ds, xgrid, tracer)
    dtracer_dxi *= ds["pm"] * dv_dpi
    pv -= dtracer_dxi
    del dtracer_dxi, dv_dpi
    
    # divide by the vertical grid spacing to get potential vorticity
    dz = _compute_dz(ds, xgrid)
    return scaling * pv / dz


def add_potential_vorticity(
    ds: xr.Dataset,
    tracer: str = "b",
    xgrid: xgcm.Grid | None = None,
    scaling: float = 1.0,
) -> xr.Dataset:
    """Add potential vorticity variable to the dataset."""
    ds["potential_vorticity"] = xr.map_blocks(
        compute_potential_vorticity,
        ds, 
        kwargs={"tracer": tracer, "xgrid": xgrid, "scaling": scaling},
        template=ds.temp.isel(time=0).copy()
    )
    return ds
