"""Minimal GeoTIFF geo-referencing support (no GDAL dependency).

Parses the standard GeoTIFF tags directly via tifffile and builds a
pixel -> (lat, lon) mapping using pyproj for CRS reprojection. Supports
the common single-tiepoint + pixel-scale case, and falls back to a
least-squares affine fit when multiple ground-control points (GCPs)
are present (the format real Sentinel-1 GRD products use).

Real satellite scenes can be hundreds of MB to a few GB (a Sentinel-1
GRD tile is commonly ~20000x8000 px). On a memory-constrained machine,
loading + float32-converting the full array can exceed available RAM.
To stay safe regardless of scene size, this reads the raster as a
memory-map from disk and immediately takes a strided, downsampled view
before any dtype conversion -- so peak memory stays proportional to the
downsampled output, not the source file.
"""

import numpy as np
import tifffile
from pyproj import Transformer

TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_TRANSFORMATION = 34264
TAG_GEOKEY_DIRECTORY = 34735

KEY_MODEL_TYPE = 1024
KEY_GEOGRAPHIC_CS = 2048
KEY_PROJECTED_CS = 3072

# Cap the retained raster's longest side. Plenty for on-screen preview and
# for a reasonable spill-centroid estimate; keeps memory use small and
# bounded no matter how large the source scene is.
MAX_DIM = 1600


def _parse_geokeys(geokey_values):
    """Extract EPSG code + whether the CRS is geographic vs projected."""
    if geokey_values is None or len(geokey_values) < 4:
        return None, None

    num_keys = geokey_values[3]
    epsg = None
    model_type = None

    for i in range(num_keys):
        offset = 4 + i * 4
        if offset + 4 > len(geokey_values):
            break
        key_id, tag_loc, count, value = geokey_values[offset:offset + 4]
        if tag_loc != 0:
            continue  # value stored elsewhere (ASCII/double params) -- not needed here
        if key_id == KEY_MODEL_TYPE:
            model_type = value
        elif key_id == KEY_GEOGRAPHIC_CS and value not in (0, 32767):
            epsg = value
        elif key_id == KEY_PROJECTED_CS and value not in (0, 32767):
            epsg = value

    return epsg, model_type


def _affine_from_tiepoint_scale(pixel_scale, tiepoints):
    scale_x, scale_y, _ = pixel_scale[:3]
    i0, j0, _, x0, y0, _ = tiepoints[:6]

    def pixel_to_native(col, row):
        x = x0 + (col - i0) * scale_x
        y = y0 - (row - j0) * scale_y
        return x, y

    return pixel_to_native


def _affine_from_gcps(tiepoints):
    points = np.array(tiepoints, dtype=np.float64).reshape(-1, 6)
    cols, rows = points[:, 0], points[:, 1]
    xs, ys = points[:, 3], points[:, 4]

    A = np.column_stack([cols, rows, np.ones_like(cols)])
    coef_x, _, _, _ = np.linalg.lstsq(A, xs, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)

    def pixel_to_native(col, row):
        x = coef_x[0] * col + coef_x[1] * row + coef_x[2]
        y = coef_y[0] * col + coef_y[1] * row + coef_y[2]
        return x, y

    return pixel_to_native


def read_geotiff(path):
    """Reads a GeoTIFF from a filesystem path.

    Returns (rgb_uint8_array, geo_info, downsample_factor). geo_info has
    'has_geo', and if True, 'pixel_to_latlon(col, row)' expecting pixel
    coordinates in the ORIGINAL (pre-downsample) raster, plus
    'orig_width'/'orig_height' for footprint computation.
    """
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        orig_height, orig_width = page.shape[:2]

        tags = page.tags
        pixel_scale = tags[TAG_PIXEL_SCALE].value if TAG_PIXEL_SCALE in tags else None
        tiepoints = tags[TAG_TIEPOINT].value if TAG_TIEPOINT in tags else None
        geokeys = tags[TAG_GEOKEY_DIRECTORY].value if TAG_GEOKEY_DIRECTORY in tags else None

        factor = max(1, -(-max(orig_width, orig_height) // MAX_DIM))  # ceil division

        # Memory-map the page and take a strided view BEFORE materializing
        # any full-size array -- only the downsampled subset gets copied
        # into real memory.
        mmap_arr = page.asarray(out="memmap")
        band = np.array(mmap_arr[::factor, ::factor], dtype=np.float32)
        del mmap_arr

    geo_info = {"has_geo": False, "pixel_to_latlon": None}
    epsg, _ = _parse_geokeys(geokeys)

    pixel_to_native = None
    if pixel_scale is not None and tiepoints is not None and len(tiepoints) == 6:
        pixel_to_native = _affine_from_tiepoint_scale(pixel_scale, tiepoints)
    elif tiepoints is not None and len(tiepoints) >= 12:
        pixel_to_native = _affine_from_gcps(tiepoints)
        # Sentinel-1/ALOS GCP grids are near-universally WGS84 lon/lat/height
        # and typically ship without an explicit GeoKeyDirectoryTag, since the
        # CRS is implied by convention rather than declared. Only fall back
        # to this default when no CRS tag was actually found.
        if epsg is None:
            epsg = 4326

    if pixel_to_native is not None and epsg is not None:
        transformer = None
        if epsg != 4326:
            transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

        def pixel_to_latlon(col, row):
            x, y = pixel_to_native(col, row)
            if transformer is not None:
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = x, y
            return lat, lon

        geo_info["has_geo"] = True
        geo_info["pixel_to_latlon"] = pixel_to_latlon
        geo_info["orig_width"] = orig_width
        geo_info["orig_height"] = orig_height

    if band.ndim == 3:
        band = band[..., 0]

    lo, hi = np.percentile(band, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((band - lo) / (hi - lo), 0, 1)
    gray_u8 = (normalized * 255).astype(np.uint8)
    rgb = np.stack([gray_u8] * 3, axis=-1)

    return rgb, geo_info, factor


def image_footprint(geo_info):
    """Returns the 4 corner [lat, lon] pairs of the original image, or None."""
    if not geo_info["has_geo"]:
        return None
    p2ll = geo_info["pixel_to_latlon"]
    w, h = geo_info["orig_width"], geo_info["orig_height"]
    corners_px = [(0, 0), (w, 0), (w, h), (0, h)]
    return [list(p2ll(c, r)) for c, r in corners_px]
