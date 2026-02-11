import matplotlib
matplotlib.use('Agg')
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os

# CONFIGURATION - Edit these to visualize different patches
DEFAULT_PATCH_ID = 3  # Default patch ID if none provided
PATCHES_FOLDER = "river_patches"

def visualize_single_patch(patch_id, metadata_filename="metadata.json"):
    """Visualize a single patch with the query point marked - show all 4 bands separately"""
    
    # Load metadata
    metadata_file = os.path.join(PATCHES_FOLDER, metadata_filename)
    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)
    
    # Find the specific patch
    patch_meta = None
    for meta in all_metadata:
        if meta['patch_id'] == patch_id:
            patch_meta = meta
            break
    
    if patch_meta is None:
        print(f"Patch {patch_id} not found in metadata")
        return
    
    tif_path = patch_meta['file']
    query_lat = patch_meta['lat']
    query_lon = patch_meta['lon']
    target_date = patch_meta['target_date']
    actual_date = patch_meta.get('actual_date', target_date)  # Use actual if available
    days_diff = patch_meta.get('days_difference', 0)
    
    print(f"Visualizing patch {patch_id}")
    print(f"  Location: ({query_lat}, {query_lon})")
    print(f"  Target date: {target_date}")
    if days_diff > 0:
        print(f"  Actual date: {actual_date} ({days_diff:.0f} days from target)")
    
    # Extract scalar features if available
    rainfall_7day = None
    temperature = None
    temporal_change = None
    
    if 'scalar_features' in patch_meta:
        rainfall_7day = patch_meta['scalar_features'].get('rainfall_7day_mm')
        temperature = patch_meta['scalar_features'].get('temperature_celsius')
    
    if 'temporal_change' in patch_meta and patch_meta['temporal_change'] is not None:
        temporal_change = patch_meta['temporal_change']
    
    # Read the GeoTIFF
    with rasterio.open(tif_path) as src:
        # Read bands
        red = src.read(1)
        nir = src.read(2)
        swir1 = src.read(3)
        swir2 = src.read(4)
        ndvi = src.read(6) if src.count >= 6 else None
        ndmi = src.read(7) if src.count >= 7 else None
        water_mask = src.read(5) if src.count >= 5 else None
        ndvi_change_rate = src.read(8) if src.count >= 8 else None

        # If derived bands are not present in tif, compute locally.
        if ndvi is None:
            ndvi = (nir - red) / np.where((nir + red) == 0, np.nan, (nir + red))
        if ndmi is None:
            ndmi = (nir - swir1) / np.where((nir + swir1) == 0, np.nan, (nir + swir1))
        if water_mask is None:
            water_mask = ((ndvi < 0.1) & (ndmi < 0)).astype(np.float32)
        
        # Get image dimensions
        height, width = red.shape
        
        # The query point should be at the center of the patch
        # Since we created a square patch centered on the query point
        row = height // 2
        col = width // 2
    
    print(f"  Query point at image center: row={row}, col={col}")
    print(f"  Image shape: {red.shape}")
    print(f"  Number of bands: {src.count}")
    
    # Print scalar features if available
    if rainfall_7day is not None:
        print(f"  7-day Rainfall: {rainfall_7day:.2f} mm")
    if temperature is not None:
        print(f"  Temperature: {temperature:.2f} °C")
    if temporal_change is not None:
        print(f"  ΔNDVI/Δt available (gap: {temporal_change['time_gap_days']:.0f} days)")
    
    # Normalize bands for visualization
    def normalize(band):
        band = np.nan_to_num(band, nan=0)
        valid_pixels = band[band > -100]  # Remove nodata values
        if len(valid_pixels) == 0:
            return np.zeros_like(band)
        p2, p98 = np.percentile(valid_pixels, (2, 98))
        if p98 - p2 == 0:
            return band * 0
        return np.clip((band - p2) / (p98 - p2), 0, 1)
    
    # Create figure with 3x3 subplots for all bands
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    # Prepare band data
    bands_data = [
        (normalize(red), 'Red Band', 'Reds'),
        (normalize(nir), 'NIR Band', 'Greens'),
        (normalize(swir1), 'SWIR1 Band', 'YlOrBr'),
        (normalize(swir2), 'SWIR2 Band', 'copper'),
        (water_mask, 'Water Mask (local if missing)', 'Blues'),
        (ndvi, 'NDVI (local if missing)', 'RdYlGn'),
        (ndmi, 'NDMI (local if missing)', 'BrBG'),
    ]
    
    # Add NDVI change rate if available (key phosphorus indicator)
    if ndvi_change_rate is not None:
        bands_data.append((ndvi_change_rate, 'ΔNDVI/Δt (Change Rate)\n[Phosphorus Indicator]', 'RdYlGn'))
    
    # Plot each band
    for i, (band_data, title, cmap) in enumerate(bands_data):
        ax = axes[i]
        
        # Show band
        im = ax.imshow(band_data, cmap=cmap)
        
        # Mark the query point
        if 0 <= row < band_data.shape[0] and 0 <= col < band_data.shape[1]:
            # Draw a large visible marker
            ax.plot(col, row, 'r*', markersize=30, markeredgecolor='yellow', 
                    markeredgewidth=3, label='Query Point')
            # Add crosshairs
            ax.axhline(y=row, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=col, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Use remaining subplot(s) for scalar features text
    scalar_ax_idx = len(bands_data)
    if scalar_ax_idx < len(axes):
        scalar_ax = axes[scalar_ax_idx]
        scalar_ax.axis('off')
        
        # Display scalar features as text
        scalar_text = "Scalar Features:\n\n"
        if rainfall_7day is not None:
            scalar_text += f"7-day Rainfall: {rainfall_7day:.2f} mm\n\n"
        else:
            scalar_text += "7-day Rainfall: N/A\n\n"
        
        if temperature is not None:
            scalar_text += f"Temperature: {temperature:.2f} °C\n\n"
        else:
            scalar_text += "Temperature: N/A\n\n"
        
        if temporal_change is not None:
            scalar_text += "Temporal Change Metrics:\n"
            scalar_text += f"Time gap: {temporal_change['time_gap_days']:.0f} days\n"
            scalar_text += f"Mean ΔNDVI/Δt: {temporal_change['ndvi_change_mean']:.6f}/day\n"
            scalar_text += f"Rapid increase: {temporal_change['rapid_increase_pct']:.1f}%\n"
            scalar_text += f"Rapid decrease: {temporal_change['rapid_decrease_pct']:.1f}%\n"
        else:
            scalar_text += "Temporal Change: N/A"
        
        scalar_ax.text(0.1, 0.5, scalar_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide remaining unused subplots
    for i in range(len(bands_data) + 1, len(axes)):
        axes[i].axis('off')
    
    # Overall title with satellite info
    satellite = patch_meta.get('satellite', 'Unknown')
    cloud_cover = patch_meta.get('cloud_cover', 'N/A')
    title_text = f'Patch {patch_id} - All Bands & Features\n'
    title_text += f'Lat: {query_lat:.4f}, Lon: {query_lon:.4f} | Satellite: {satellite}\n'
    title_text += f'Target: {target_date}, Actual: {actual_date}'
    if isinstance(cloud_cover, (int, float)):
        title_text += f' | Cloud Cover: {cloud_cover:.1f}%'
    
    fig.suptitle(title_text, fontsize=14, y=0.99)
    
    plt.tight_layout()
    
    # Save
    output_path = f"patch_{patch_id}_viz.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a single river patch by patch_id.")
    parser.add_argument(
        "--id",
        type=int,
        default=DEFAULT_PATCH_ID,
        help=f"Patch id to visualize (default: {DEFAULT_PATCH_ID})"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.json",
        help="Metadata JSON filename inside PATCHES_FOLDER (default: metadata.json)"
    )
    args = parser.parse_args()
    visualize_single_patch(args.id, args.metadata)
