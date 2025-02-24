"""
The source code is adapted from https://github.com/aliaksandr960/segment-anything-eo. Credit to the author Aliaksandr Hancharenka.
"""

import shapely
import rasterio
import geopandas as gpd


def raster_to_vector(source, output, simplify_tolerance=None, dst_crs=None, **kwargs):
    """Vectorize a raster dataset.

    Args:
        source (str): The path to the tiff file.
        output (str): The path to the vector file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """
    from rasterio import features

    with rasterio.open(source) as src:
        band = src.read()

        mask = band != 0
        shapes = features.shapes(band, mask=mask, transform=src.transform)

    fc = [
        {"geometry": shapely.geometry.shape(shape), "properties": {"value": value}}
        for shape, value in shapes
    ]
    if simplify_tolerance is not None:
        for i in fc:
            i["geometry"] = i["geometry"].simplify(tolerance=simplify_tolerance)

    gdf = gpd.GeoDataFrame.from_features(fc)
    if src.crs is not None:
        gdf.set_crs(crs=src.crs, inplace=True)

    if dst_crs is not None:
        gdf = gdf.to_crs(dst_crs)

    gdf.to_file(output, **kwargs)


def raster_to_gpkg(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a gpkg file.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the gpkg file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    if not output.endswith(".gpkg"):
        output += ".gpkg"

    raster_to_vector(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def raster_to_shp(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a shapefile.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the shapefile.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    if not output.endswith(".shp"):
        output += ".shp"

    raster_to_vector(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def raster_to_geojson(tiff_path, output, simplify_tolerance=None, **kwargs):
    """Convert a tiff file to a GeoJSON file.

    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the GeoJSON file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    if not output.endswith(".geojson"):
        output += ".geojson"

    raster_to_vector(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)
