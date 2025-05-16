import geopandas as gpd

def reproject_shapefile(input_shp, output_shp, target_epsg=4326):
    """
    Reprojects a shapefile from its original CRS to the target CRS (default: EPSG:4326).
    
    Parameters:
    - input_shp: str
        Path to the input shapefile (e.g., 'path/to/your_shapefile.shp')
    - output_shp: str
        Path for the output shapefile (e.g., 'path/to/your_shapefile_4326.shp')
    - target_epsg: int, optional
        The EPSG code of the target CRS (default is 4326 for WGS84).
    
    The function reads the input shapefile, reprojects it to the target CRS,
    and writes out a new shapefile. The output shapefile will include the new .shp,
    .shx, .dbf, and .prj files.
    """
    # Load the original shapefile
    gdf = gpd.read_file(input_shp)
    
    # Reproject the GeoDataFrame to the target CRS
    gdf_reprojected = gdf.to_crs(epsg=target_epsg)
    
    # Save the reprojected shapefile to the output path
    gdf_reprojected.to_file(output_shp)
    
    print(f"Shapefile reprojected to EPSG:{target_epsg} and saved as {output_shp}")

# Example usage:
# reproject_shapefile('yearly_deforestation_biome.shp', 'yearly_deforestation_biome_4326.shp')
