import sys
import spatialdata as sd
from napari_spatialdata import Interactive
from pathlib import Path
import numpy as np


def create_plot_function(combined_sdata, interactive):
    """Create a plot function that can be called from the napari console."""
    
    def plot(gene_name):
        """
        Add DAPI channel and cell-circles with gene expression to the viewer.
        
        Parameters
        ----------
        gene_name : str
            Name of the gene to visualize on the cell-circles
        """
        viewer = interactive.viewer
        
        # 1. Find and add DAPI channel (look for image layers)
        # DAPI is typically a 2D image, often the only image or labeled with DAPI/dapi
        dapi_layer = None
        for name, image in combined_sdata.images.items():
            # Add the first image as DAPI (or you can filter by name containing 'dapi')
            if dapi_layer is None or 'dapi' in name.lower():
                if name not in [layer.name for layer in viewer.layers]:
                    viewer.add_image(image.compute() if hasattr(image, 'compute') else image, 
                                    name=name, 
                                    colormap='gray',
                                    blending='additive')
                    print(f"Added DAPI layer: {name}")
                dapi_layer = name
        
        # 2. Find cell-circles (look for circular shapes)
        circles_layer = None
        for name, shapes in combined_sdata.shapes.items():
            # Shapes are typically stored as GeoDataFrame with circle/polygon geometries
            if 'circle' in name.lower() or 'cell' in name.lower():
                circles_layer = name
                break
        
        if circles_layer is None and len(combined_sdata.shapes) > 0:
            # If no obvious circles, take the first shapes element
            circles_layer = list(combined_sdata.shapes.keys())[0]
        
        if circles_layer is None:
            print("No cell shapes found in the data.")
            return
        
        # 3. Get gene expression data from the table
        shapes_gdf = combined_sdata.shapes[circles_layer]
        
        # Find the associated table
        table = None
        if hasattr(combined_sdata, 'tables') and len(combined_sdata.tables) > 0:
            # Get the first table (or find the one associated with the shapes)
            table_name = list(combined_sdata.tables.keys())[0]
            table = combined_sdata.tables[table_name]
        
        if table is None:
            print("No expression table found in the data.")
            return
        
        # 4. Extract gene expression values
        if gene_name not in table.var_names:
            print(f"Gene '{gene_name}' not found in the data. Available genes: {list(table.var_names[:10])}...")
            return
        
        # Get expression values for the gene
        gene_idx = list(table.var_names).index(gene_name)
        expression_values = table.X[:, gene_idx]
        
        # Convert to dense if sparse
        if hasattr(expression_values, 'toarray'):
            expression_values = expression_values.toarray().flatten()
        else:
            expression_values = np.asarray(expression_values).flatten()
        
        # 5. Extract coordinates from shapes
        coordinates = np.array([[geom.centroid.x, geom.centroid.y] for geom in shapes_gdf.geometry])
        
        # 6. Add as points layer with gene expression as colors
        layer_name = f"{circles_layer}_{gene_name}"
        
        # Remove existing layer if present
        if layer_name in [layer.name for layer in viewer.layers]:
            viewer.layers.remove(layer_name)
        
        viewer.add_points(
            coordinates,
            face_color=expression_values,
            name=layer_name,
            size=10,
            edge_width=0.5,
            edge_color='white',
            colormap='viridis'
        )
        
        print(f"Added {len(coordinates)} cells with {gene_name} expression")
    
    return plot


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: napari-all-slides <zarr_file1> <zarr_file2> ...")

    # 1. Get zarr file paths from command line arguments
    zarr_paths = [Path(arg) for arg in sys.argv[1:]]

    # 2. Load all .zarr files into a DICTIONARY
    # Key = filename (e.g., 'slide_A'), Value = the SpatialData object
    # This automatically prefixes all elements with the filename (e.g., 'slide_A_nuclei')
    sdata_dict = {f.stem: sd.read_zarr(f) for f in zarr_paths}

    if sdata_dict:
        print(f"Concatenating {len(sdata_dict)} datasets...")

        # 3. Concatenate using the dictionary
        combined_sdata = sd.concatenate(sdata_dict)

        print(combined_sdata)
        # 4. Store data in module-level storage BEFORE Interactive() blocks
        # Also store the order of samples (keys from sdata_dict preserve insertion order in Python 3.7+)
        from napari_spatial_tools._session import set_session_data
        sample_order = list(sdata_dict.keys())
        set_session_data(combined_sdata, None, sample_order=sample_order)
        
        print("\n" + "="*60)
        print("Starting napari viewer...")
        print("\nOnce the viewer opens, run this in the napari console:")
        print("\n  from napari_spatial_tools.console_helper import plot")
        print("  plot('CD8A')")
        print("\nThen you can visualize any gene:")
        print("  plot('PECAM1')")
        print("  plot('KRT8')")
        print("="*60 + "\n")
        
        # Interactive() blocks immediately in its constructor (starts Qt event loop)
        # Everything after this line will NOT execute until the viewer closes
        Interactive(combined_sdata)
    else:
        print("No .zarr files provided.")

if __name__ == "__main__":
    main()
