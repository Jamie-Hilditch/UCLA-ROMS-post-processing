"""Process ROMS time variables and add time variables to output files."""

import numpy as np
import xarray as xr


def make_ocean_time_dim(ds: xr.Dataset) -> xr.Dataset:
    """Make ocean time the dimension."""
    return ds.swap_dims(time="ocean_time")

def add_time_coordinate(ds: xr.Dataset, origin: np.datetime64) -> xr.Dataset:
    """Add time coordinate to the dataset."""
    nanoseconds = ds["ocean_time"] * 1e9
    nanoseconds = nanoseconds.drop_attrs()
    time_since_origin = nanoseconds.astype("int64").astype("timedelta64[ns]")
    time = origin + time_since_origin
    ds = ds.assign_coords(time=time)
    return ds
