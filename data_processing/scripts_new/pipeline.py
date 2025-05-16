from pathlib import Path
import numpy as np
import rasterio
import argparse
import sys
import os

# Import our modules
import create_mask
import process_sentinel

def parse_args():
    """Parse command line arguments with validation."""
    parser = argparse.ArgumentParser(description='Process Sentinel-2 images and create masks')
    parser.add_argument('--input-dirs', type=str, nargs='+', required=True,
                      help='List of input directories containing raw data')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Base output directory for processed data and masks')
    args = parser.parse_args()
    
    # Validate input directories
    for input_dir in args.input_dirs:
        path = Path(input_dir)
        if not path.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {args.output_dir}")
        except Exception as e:
            print(f"Error: Failed to create output directory {args.output_dir}: {str(e)}")
            sys.exit(1)
    
    return args

def process_single_region(input_base, output_base, bands_to_keep):
    """Process a single region's data with error handling."""
    try:
        # Get location name from input path
        location_name = input_base.name  # This will be "location_1", "location_2", etc.
        print(f"Processing location: {location_name}")
        
        # Check for required directories
        for year in ['july2022', 'july2023']:
            year_dir = input_base / 'sentinel2' / year
            if not year_dir.exists():
                print(f"Error: Missing directory: {year_dir}")
                return None
        
        # Check for PRODES shapefiles
        recent_defor = input_base.parent / 'prodes' / 'yearly_deforestation_biome.shp'
        accum_defor = input_base.parent / 'prodes' / 'accumulated_deforestation_2007' / 'accumulated_deforestation_2007.shp'
        
        if not recent_defor.exists():
            print(f"Error: Missing shapefile: {recent_defor}")
            return None
        
        if not accum_defor.exists():
            print(f"Error: Missing shapefile: {accum_defor}")
            return None
        
        # Define raw input directories
        input_dir_2022 = input_base / 'sentinel2' / 'july2022'
        input_dir_2023 = input_base / 'sentinel2' / 'july2023'
        
        # Create output directory structure
        temp_dir = output_base / 'temp' / location_name
        processed_dir = output_base / 'processed' / location_name
        
        temp_sentinel_dir_2022 = temp_dir / 'sentinel2' / 'july2022'
        temp_sentinel_dir_2023 = temp_dir / 'sentinel2' / 'july2023'
        
        # Create directories
        for directory in [temp_sentinel_dir_2022, temp_sentinel_dir_2023]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error: Failed to create directory {directory}: {str(e)}")
                return None
                
        print(f"Input dir 2022: {input_dir_2022}")
        print(f"Input dir 2023: {input_dir_2023}")
        
        # Step 1A: Filter the bands for each year
        try:
            print("Filtering Sentinel bands for 2022...")
            process_sentinel.filter_sentinel_bands(input_dir_2022, temp_sentinel_dir_2022, bands_to_keep)
            
            print("Filtering Sentinel bands for 2023...")
            process_sentinel.filter_sentinel_bands(input_dir_2023, temp_sentinel_dir_2023, bands_to_keep)
        except Exception as e:
            print(f"Error filtering Sentinel bands: {str(e)}")
            return None
        
        # Create directories for processed data
        combined_dir = processed_dir / 'sentinel2' / 'combined'
        sentinel_processed_dir = processed_dir / 'sentinel2'
        
        for directory in [combined_dir, sentinel_processed_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error: Failed to create directory {directory}: {str(e)}")
                return None
        
        # Step 1B: Create VRTs and save merged arrays
        vrt_paths = {}
        for year in ['july2022', 'july2023']:
            year_folder = temp_dir / 'sentinel2' / year
            try:
                tif_files = sorted(year_folder.glob('*.tif'))
                if not tif_files:
                    print(f"Warning: No filtered files found for {year}.")
                    continue
                combined_dir = sentinel_processed_dir / 'combined' 
                vrt_path = sentinel_processed_dir / f'Sentinel2_{year}.vrt'
                print(f"Creating VRT for {year} at {vrt_path}")
                process_sentinel.create_vrt(tif_files, vrt_path, combined_dir)
                vrt_paths[year] = vrt_path
                
            except Exception as e:
                print(f"Error processing {year}: {str(e)}")
                continue
        
        # Check if we have the necessary VRT for mask creation
        if 'july2023' not in vrt_paths:
            print("Error: Missing required 2023 VRT for mask creation. Aborting.")
            return None
        
        # Step 2: Create the Deforestation Mask
        ref_vrt_path = vrt_paths['july2023']
        
        mask_out_dir = processed_dir / 'masks'
        try:
            mask_out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Failed to create mask directory: {str(e)}")
            return None
            
        mask_tiff_path = mask_out_dir / 'prodes_mask_aug2022_jun2023.tif'
        mask_npy_path = mask_out_dir / 'prodes_mask_aug2022_jun2023.npy'
        
        # Create the mask from shapefile using location_name
        try:
            print("Creating deforestation mask...")
            create_mask.create_mask_from_shapefiles(
                recent_defor_shapefile_path = input_base.parent / 'prodes' / 'yearly_deforestation_biome.shp',
                accumulated_defor_shapefile_path = input_base.parent / 'prodes' / 'accumulated_deforestation_2007' / 'accumulated_deforestation_2007.shp',
                reference_vrt_path = ref_vrt_path,
                output_tiff_path = mask_tiff_path,
                output_npy_path = mask_npy_path,
                date_field = 'year',
                location_name = location_name  
            )
            print(f"Mask creation completed. Saved at {mask_tiff_path} and {mask_npy_path}")
        except Exception as e:
            print(f"Error creating mask: {str(e)}")
            return None
        
        print(f"Successfully processed region {location_name}")
        return processed_dir
        
    except Exception as e:
        print(f"Unexpected error processing region {input_base.name}: {str(e)}")
        return None

def run_pipeline(args):
    """Run the complete pipeline with error handling."""
    print("Starting Sentinel-2 processing pipeline")
    
    # Track success/failure for each input directory
    success_count = 0
    failed_dirs = []
    
    # Process each input directory
    for input_dir in args.input_dirs:
        print(f"\nProcessing input directory: {input_dir}")
        try:
            input_base = Path(input_dir)
            output_base = Path(args.output_dir)
            
            # Process the region and store its processed directory
            processed_dir = process_single_region(
                input_base=input_base,
                output_base=output_base,
                bands_to_keep=['B2', 'B3', 'B4', 'B8', 'MSK_CLDPRB', 'missing']
            )
            
            if processed_dir:
                print(f"Successfully processed {input_dir}")
                success_count += 1
            else:
                print(f"Failed to process {input_dir}")
                failed_dirs.append(input_dir)
                
        except Exception as e:
            print(f"Unexpected error processing {input_dir}: {str(e)}")
            failed_dirs.append(input_dir)
    
    # Print summary
    print("\n--- Processing Summary ---")
    print(f"Total locations: {len(args.input_dirs)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed processing: {len(failed_dirs)}")
    
    if failed_dirs:
        print(f"Failed locations: {', '.join(failed_dirs)}")
    
    print("Pipeline processing complete.")

if __name__ == "__main__":
    try:
        args = parse_args()
        run_pipeline(args)
    except Exception as e:
        print(f"Critical error in pipeline: {str(e)}")
        sys.exit(1)