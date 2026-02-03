"""
Helper module for use in the napari console.

Import this in the napari console to get easy access to the plot function:
    from napari_spatial_tools.console_helper import plot, get_sdata
    
    # Plot genes
    plot('CD8A')
    
    # Access spatial data
    sdata = get_sdata()
"""

import numpy as np


def get_sdata():
    """
    Get the current SpatialData object.
    
    Returns
    -------
    SpatialData
        The concatenated spatial data object from all loaded zarr files.
        
    Examples
    --------
    >>> from napari_spatial_tools.console_helper import get_sdata
    >>> sdata = get_sdata()
    >>> print(sdata)
    >>> print(list(sdata.shapes.keys()))
    >>> print(list(sdata.tables.keys()))
    """
    from napari_spatial_tools._session import get_session_data
    return get_session_data()


def compare_samples(gene_name, samples, vmin=None, vmax=None, orientation='horizontal'):
    """
    Compare the same gene across multiple samples side-by-side in grid view.
    
    Parameters
    ----------
    gene_name : str
        Gene name to visualize across all samples
    samples : list of int or str
        List of samples to compare (can be indices or names)
    vmin : float or str, optional
        Minimum value for all samples (same as plot())
    vmax : float or str, optional
        Maximum value for all samples (same as plot())
    orientation : str, optional
        'horizontal' for side-by-side (default) or 'vertical' for top-bottom
        
    Examples
    --------
    >>> compare_samples('CD8A', samples=[0, 1])  # Compare Region 2 vs Region 3
    >>> compare_samples('EPCAM', samples=[0, 1, 2], vmin='p5', vmax='p95')
    """
    import napari
    from napari.layers import Points, Image
    from napari_spatialdata._viewer import SpatialDataViewer
    import math
    import numpy as np
    import pandas as pd
    from matplotlib import cm
    
    viewer = napari.current_viewer()
    if viewer is None:
        print("Error: No napari viewer found.")
        return
    
    # Get sdata and available samples
    from napari_spatial_tools._session import get_session_data, get_sample_order
    sdata = get_session_data()
    available_samples = get_sample_order()
    
    # Helper to parse vmin/vmax
    def parse_vlim(value, data):
        if value is None:
            return None
        if isinstance(value, str) and value.startswith('p'):
            percentile = float(value[1:])
            return np.percentile(data, percentile)
        return float(value)
    
    # Helper to convert sample index/name
    def _get_sample_name(sample_param):
        if isinstance(sample_param, int):
            if 0 <= sample_param < len(available_samples):
                return available_samples[sample_param]
            else:
                raise ValueError(f"Sample index {sample_param} out of bounds.")
        return sample_param
    
    # Remove ALL comparison layers (from both compare_samples and compare_genes)
    for layer in list(viewer.layers):
        if (layer.name.startswith('sample_') or 
            layer.name.startswith('dapi_sample_') or
            layer.name.startswith('gene_') or 
            layer.name.startswith('dapi_')):
            try:
                del viewer.layers[layer.name]
            except:
                pass
    
    # Hide all existing base layers
    for layer in viewer.layers:
        layer.visible = False
    
    # Create viewer model for layer creation
    viewer_model = SpatialDataViewer(viewer, [sdata])
    
    # Create layers for each sample
    dapi_layers = []
    gene_layers = []
    
    for sample_idx, sample_param in enumerate(samples):
        sample_name = _get_sample_name(sample_param)
        print(f"Creating layers for sample: {sample_name}")
        
        # Find DAPI layer name
        dapi_layer_name = None
        for name in sdata.images.keys():
            if sample_name in name and ('dapi' in name.lower() or 'morphology_focus' in name.lower()):
                dapi_layer_name = name
                break
        
        # Find shapes element name
        shapes_element_name = None
        for name in sdata.shapes.keys():
            if sample_name in name and 'cell_circles' in name.lower():
                shapes_element_name = name
                break
        
        # Find table name
        table_name = None
        for name in sdata.tables.keys():
            if sample_name in name:
                table_name = name
                break
        
        if not shapes_element_name or not table_name:
            print(f"Error: Could not find required data for sample {sample_name}")
            continue
        
        # Get the correct coordinate system for this element
        shapes_element = sdata.shapes[shapes_element_name]
        if hasattr(shapes_element, 'attrs') and 'transform' in shapes_element.attrs:
            available_cs = list(shapes_element.attrs['transform'].keys())
            selected_cs = available_cs[0] if available_cs else 'global'
        else:
            available_cs = list(sdata.coordinate_systems)
            selected_cs = available_cs[0] if available_cs else 'global'
        
        # Create DAPI layer
        if dapi_layer_name:
            dapi_layer = viewer_model.get_sdata_image(sdata, dapi_layer_name, selected_cs, False)
            dapi_layer.name = f"dapi_sample_{sample_idx}"
            dapi_layer.colormap = 'gray'
            dapi_layer.blending = 'additive'
            dapi_layers.append(dapi_layer)
        
        # Create gene layer
        gene_layer = viewer_model.get_sdata_circles(sdata, shapes_element_name, selected_cs, False)
        gene_layer.name = f"sample_{sample_idx}_{gene_name}"
        
        # Get expression data and update colors
        table = sdata.tables[table_name]
        if gene_name not in table.var_names:
            print(f"Error: Gene '{gene_name}' not found in sample {sample_name}")
            continue
        
        expression = table[:, gene_name].X.toarray().flatten()
        
        # Parse vmin/vmax
        vmin_val = parse_vlim(vmin, expression)
        vmax_val = parse_vlim(vmax, expression)
        if vmin_val is None:
            vmin_val = expression.min()
        if vmax_val is None:
            vmax_val = expression.max()
        if vmax_val <= vmin_val:
            vmax_val = vmin_val + 1
            print(f"Warning: vmax collapsed to vmin for {sample_name}, adjusted to {vmax_val:.2f}")
        
        # Apply colors
        norm_values = np.clip((expression - vmin_val) / (vmax_val - vmin_val), 0, 1)
        cmap_obj = cm.get_cmap('viridis')
        colors = cmap_obj(norm_values)
        gene_layer.face_color = colors
        
        # Store expression data in features for reference
        if not hasattr(gene_layer, 'features') or gene_layer.features is None:
            gene_layer.features = pd.DataFrame()
        gene_layer.features[gene_name] = expression
        
        gene_layers.append(gene_layer)
        print(f"  ✓ Created sample_{sample_idx}_{gene_name} (range: {expression.min():.2f}-{expression.max():.2f})")
    
    # Add all layers in correct order: dapi_sample_0, dapi_sample_1, ..., sample_0_gene, sample_1_gene, ...
    for dapi_layer in dapi_layers:
        viewer.add_layer(dapi_layer)
    for gene_layer in gene_layers:
        viewer.add_layer(gene_layer)
    
    # Make all comparison layers visible
    for layer in viewer.layers:
        if layer.name.startswith('sample_') or layer.name.startswith('dapi_sample_'):
            layer.visible = True
    
    # Set up grid view
    viewer.grid.enabled = True
    n_samples = len(samples)
    
    if orientation == 'horizontal':
        if n_samples <= 3:
            rows, cols = 1, n_samples
        elif n_samples == 4:
            rows, cols = 2, 2
        else:
            rows = int(math.sqrt(n_samples))
            cols = math.ceil(n_samples / rows)
            while rows * cols < n_samples:
                cols += 1
        viewer.grid.shape = (rows, cols)
    else:
        if n_samples <= 3:
            rows, cols = n_samples, 1
        elif n_samples == 4:
            rows, cols = 2, 2
        else:
            cols = int(math.sqrt(n_samples))
            rows = math.ceil(n_samples / cols)
            while rows * cols < n_samples:
                rows += 1
        viewer.grid.shape = (rows, cols)
    
    print(f"✓ Comparing {gene_name} across {len(samples)} samples in {rows}x{cols} grid")
    print(f"  Total layers: {len(dapi_layers) + len(gene_layers)} (should be {2 * len(samples)})")


def compare_genes(genes, sample=None, vmin=None, vmax=None, orientation='horizontal'):
    """
    Compare multiple genes side-by-side in grid view.
    
    Parameters
    ----------
    genes : list of str
        List of gene names to compare
    sample : str or int, optional
        Sample to visualize (same as plot())
    vmin : float or str, optional
        Minimum value for all genes (same as plot())
    vmax : float or str, optional
        Maximum value for all genes (same as plot())
    orientation : str, optional
        'horizontal' for side-by-side (default) or 'vertical' for top-bottom
        
    Examples
    --------
    >>> compare_genes(['CD8A', 'EPCAM', 'PTPRC'], sample=0)
    >>> compare_genes(['CD8A', 'EPCAM'], sample=0, vmin='p5', vmax='p95', orientation='vertical')
    """
    import napari
    from napari.layers import Points
    
    viewer = napari.current_viewer()
    if viewer is None:
        print("Error: No napari viewer found.")
        return
    
    # Remove ALL comparison layers (from both compare_samples and compare_genes)
    layers_to_remove = []
    for layer in viewer.layers:
        if (layer.name.startswith('gene_') or 
            layer.name.startswith('dapi_') or
            layer.name.startswith('sample_') or 
            layer.name.startswith('dapi_sample_')):
            layers_to_remove.append(layer.name)
    for name in layers_to_remove:
        try:
            del viewer.layers[name]
        except:
            pass
    
    # Plot each gene and create a copy with a unique name
    created_layers = []
    for gene in genes:
        # Plot the gene (updates the base layer) - use internal flag to prevent cleanup
        plot(gene, sample=sample, vmin=vmin, vmax=vmax, _internal_call=True)
        
        # Find the base layer that was just updated
        base_layer = None
        for layer in viewer.layers:
            if str(type(layer).__name__) == 'Points' and ('cell_circles' in layer.name):
                base_layer = layer
                break
        
        if base_layer is None:
            print(f"Warning: Could not find base layer for {gene}")
            continue
        
        # Create a copy with proper transforms and metadata
        new_layer = Points(
            base_layer.data,
            name=f"gene_{gene}",
            face_color=base_layer.face_color.copy(),
            size=base_layer.size,
            border_width=base_layer.border_width,
            border_color=base_layer.border_color,
            affine=base_layer.affine,
            metadata=base_layer.metadata.copy() if hasattr(base_layer, 'metadata') else {}
        )
        viewer.add_layer(new_layer)
        created_layers.append(new_layer)
        print(f"  Created layer: gene_{gene}")
    
    # Remove the original base layer completely (not just hide it)
    layers_to_remove = []
    for layer in viewer.layers:
        if 'cell_circles' in layer.name and not layer.name.startswith('gene_'):
            layers_to_remove.append(layer.name)
    for name in layers_to_remove:
        try:
            del viewer.layers[name]
        except:
            pass
    
    # Find the DAPI layer and duplicate it for each gene (so it appears in all grid panes)
    dapi_layer = None
    for layer in viewer.layers:
        if ('morphology' in layer.name.lower() or 'dapi' in layer.name.lower()) and not layer.name.startswith('dapi_'):
            dapi_layer = layer
            break
    
    # Remove old DAPI duplicates first
    for layer in list(viewer.layers):
        if layer.name.startswith('dapi_'):
            try:
                del viewer.layers[layer.name]
            except:
                pass
    
    if dapi_layer is not None:
        from napari.layers import Image
        
        # Create DAPI duplicates and store them
        dapi_copies = {}
        for gene in genes:
            dapi_copy = Image(
                dapi_layer.data,
                name=f"dapi_{gene}",
                colormap=dapi_layer.colormap,
                contrast_limits=dapi_layer.contrast_limits,
                blending=dapi_layer.blending,
                affine=dapi_layer.affine,
                metadata=dapi_layer.metadata.copy() if hasattr(dapi_layer, 'metadata') else {}
            )
            dapi_copies[gene] = dapi_copy
        
        # Remove the original DAPI layer
        try:
            del viewer.layers[dapi_layer.name]
        except:
            dapi_layer.visible = False
        
        # Reorder layers for grid view to ensure proper pairing
        # Order must be: dapi_gene0, dapi_gene1, ..., gene_gene0, gene_gene1, ...
        # This distributes: pane0 gets dapi_gene0+gene_gene0, pane1 gets dapi_gene1+gene_gene1
        
        # Clear all layers except keep track of gene layers
        gene_layers = []
        for gene in genes:
            for layer in viewer.layers:
                if layer.name == f"gene_{gene}":
                    gene_layers.append((gene, layer))
                    break
        
        # Remove all gene layers temporarily
        for gene, layer in gene_layers:
            viewer.layers.remove(layer)
        
        # Add all DAPI layers in order
        for gene in genes:
            viewer.add_layer(dapi_copies[gene])
        
        # Add all gene layers back in order
        for gene, layer in gene_layers:
            viewer.add_layer(layer)
    
    # Ensure gene and dapi layers are visible
    for layer in viewer.layers:
        if layer.name.startswith('gene_') or layer.name.startswith('dapi_'):
            layer.visible = True
    
    # Set up grid view with smart layout
    viewer.grid.enabled = True
    
    if orientation == 'horizontal':
        # Smart grid: try to make it as square as possible
        n_genes = len(genes)
        if n_genes <= 3:
            # For 1-3 genes, keep horizontal
            rows, cols = 1, n_genes
        else:
            # For 4+ genes, make it more square
            # Find the best rows/cols combination
            import math
            rows = int(math.sqrt(n_genes))
            cols = math.ceil(n_genes / rows)
            # Adjust if needed to ensure rows * cols >= n_genes
            while rows * cols < n_genes:
                cols += 1
        viewer.grid.shape = (rows, cols)
    else:
        # Vertical orientation: prioritize columns
        n_genes = len(genes)
        if n_genes <= 3:
            rows, cols = n_genes, 1
        else:
            import math
            cols = int(math.sqrt(n_genes))
            rows = math.ceil(n_genes / cols)
            while rows * cols < n_genes:
                rows += 1
        viewer.grid.shape = (rows, cols)
    
    print(f"✓ Comparing {len(genes)} genes in {rows}x{cols} grid view")
    print(f"  Genes: {', '.join(genes)}")
    print(f"  DAPI layer will appear in all panes")


def plot(gene_name, sample=None, vmin=None, vmax=None, _internal_call=False):
    """
    Plot gene expression on cells in the napari viewer.
    
    Parameters
    ----------
    gene_name : str
        Name of the gene to visualize
    sample : str or int, optional
        Sample/slide to visualize. Can be:
        - int: index of the zarr file (0 for first, 1 for second, etc.)
        - str: full sample name
        - None: auto-selects if only one sample exists
    vmin : float or str, optional
        Minimum value for color normalization. Can be:
        - float: explicit minimum value
        - str: percentile like "p5" for 5th percentile
        - None: uses data minimum (default)
    vmax : float or str, optional
        Maximum value for color normalization. Can be:
        - float: explicit maximum value
        - str: percentile like "p95" for 95th percentile
        - None: uses data maximum (default)
        
    Examples
    --------
    >>> plot('CD8A')  # Single sample
    >>> plot('EPCAM', sample=0)  # First zarr file
    >>> plot('EPCAM', sample=1)  # Second zarr file
    >>> plot('EPCAM', sample='output-XETG00117__0015978__Region_2__20240718__175145')  # Full name
    >>> plot('PTPRC', sample=0, vmin='p5', vmax='p95')
    """
    # Get the viewer from napari
    import napari
    viewer = napari.current_viewer()
    
    if viewer is None:
        print("Error: No napari viewer found. Make sure the viewer is open.")
        return
    
    # Get sdata from module-level storage
    sdata = get_sdata()
    
    if sdata is None:
        print("Error: Could not find spatial data. Make sure you launched with napari-all-slides.")
        return
    
    # Only cleanup comparison layers if this is a direct user call (not from compare_genes/compare_samples)
    if not _internal_call:
        viewer.grid.enabled = False
        for layer in list(viewer.layers):
            if (layer.name.startswith('gene_') or 
                layer.name.startswith('dapi_') or
                layer.name.startswith('sample_') or 
                layer.name.startswith('dapi_sample_')):
                try:
                    del viewer.layers[layer.name]
                except:
                    pass
    
    # Helper function to parse vmin/vmax
    def parse_vlim(value, data):
        """Parse vmin/vmax which can be numeric or percentile string like 'p95'"""
        if value is None:
            return None
        if isinstance(value, str) and value.startswith('p'):
            # Parse percentile, e.g., "p95" -> 95th percentile
            percentile = float(value[1:])
            return np.percentile(data, percentile)
        return float(value)
    
    # 1. Detect available samples from shape elements (will be used later)
    # This happens before determining which sample to use
    
    # 2. Determine which sample to use
    # Get all shape element names
    all_shape_names = list(sdata.shapes.keys())
    
    # Extract sample names (suffixes after the element type)
    # E.g., "cell_circles-output-XETG00117__0015978__Region_2__20240718__175145"
    # Sample is: "output-XETG00117__0015978__Region_2__20240718__175145"
    samples = {}
    for name in all_shape_names:
        # Look for the pattern: element_type-sample_name or element_type_sample_name
        if '-' in name:
            # Split on dash to get sample suffix
            parts = name.split('-', 1)
            if len(parts) == 2:
                sample_suffix = parts[1]
                if sample_suffix not in samples:
                    samples[sample_suffix] = []
                samples[sample_suffix].append(name)
    
    # If no samples with suffixes found, treat as single sample
    if not samples:
        samples = {'default': all_shape_names}
    
    # Get the sample order from session (if available)
    from napari_spatial_tools._session import get_sample_order
    sample_order = get_sample_order()
    
    # If sample is an integer, convert to sample name using the order
    if isinstance(sample, int):
        if sample_order and 0 <= sample < len(sample_order):
            # Convert index to actual sample name (stem of zarr file)
            zarr_stem = sample_order[sample]
            # Find the matching sample key (the suffix)
            sample_key = None
            for key in samples.keys():
                if zarr_stem in key or key in zarr_stem:
                    sample_key = key
                    break
            
            if sample_key is None:
                # If no match, try to find by index in samples dict
                sample_keys = list(samples.keys())
                if sample < len(sample_keys):
                    sample_key = sample_keys[sample]
                else:
                    print(f"Error: Sample index {sample} out of range. Available: 0-{len(sample_keys)-1}")
                    return
            
            sample = sample_key
            print(f"Using sample index {sample} -> {sample_key}")
        else:
            print(f"Error: Sample index {sample} out of range. Available indices: 0-{len(sample_order)-1 if sample_order else len(samples)-1}")
            return
    
    # Determine which sample to use
    if sample is None:
        if len(samples) == 1:
            # Only one sample, use it
            selected_sample = list(samples.keys())[0]
        else:
            # Multiple samples, need to specify
            print(f"Error: Multiple samples found: {list(samples.keys())}")
            print(f"Please specify which sample to plot using: plot('{gene_name}', sample='SAMPLE_NAME')")
            return
    else:
        # Use specified sample
        if sample not in samples:
            print(f"Error: Sample '{sample}' not found. Available samples: {list(samples.keys())}")
            return
        selected_sample = sample
    
    # 3. Find cell circles element for the selected sample
    shapes_element_name = None
    candidate_names = samples[selected_sample] if selected_sample != 'default' else all_shape_names
    
    for name in candidate_names:
        if 'circle' in name.lower():
            shapes_element_name = name
            break
    
    if shapes_element_name is None:
        for name in candidate_names:
            if 'cell' in name.lower() and 'boundar' not in name.lower():
                shapes_element_name = name
                break
    
    if shapes_element_name is None:
        shapes_element_name = candidate_names[0] if candidate_names else list(sdata.shapes.keys())[0]
    
    print(f"Using sample: {selected_sample}, layer: {shapes_element_name}")
    
    # 3. Hide layers from all other samples to avoid confusion (don't remove to avoid breaking napari-spatialdata tracking)
    all_samples = list(samples.keys())
    for layer in viewer.layers:
        layer_sample = None
        
        # Determine which sample this layer belongs to
        for sample_name in all_samples:
            if sample_name != 'default' and sample_name in layer.name:
                layer_sample = sample_name
                break
        
        # Hide layers from other samples, show layers from current sample
        if layer_sample is not None:
            if layer_sample == selected_sample:
                layer.visible = True
            else:
                layer.visible = False
    
    # 3a. Add DAPI/morphology image for this sample if available and not already present
    if hasattr(sdata, 'images') and len(sdata.images) > 0:
        for img_name in sdata.images.keys():
            # Check if this image belongs to the selected sample
            if selected_sample != 'default' and not img_name.endswith(selected_sample):
                continue
            
            if ('morphology' in img_name.lower() or 'dapi' in img_name.lower() or 'focus' in img_name.lower()):
                # Check if already in viewer
                if img_name not in [layer.name for layer in viewer.layers]:
                    try:
                        from napari_spatialdata._viewer import SpatialDataViewer
                        viewer_model = SpatialDataViewer(viewer, [sdata])
                        
                        # Get available coordinate systems for the image
                        img_element = sdata.images[img_name]
                        if hasattr(img_element, 'attrs') and 'transform' in img_element.attrs:
                            available_cs = list(img_element.attrs['transform'].keys())
                            img_cs = available_cs[0] if available_cs else list(sdata.coordinate_systems)[0]
                        else:
                            img_cs = list(sdata.coordinate_systems)[0]
                        
                        # Add the image using the viewer model
                        img_layer = viewer_model.get_sdata_image(sdata, img_name, img_cs, False)
                        viewer.add_layer(img_layer)
                        print(f"✓ Added DAPI/morphology layer: {img_name}")
                    except Exception as e:
                        print(f"Note: Could not auto-add image layer: {e}")
                break
    
    # 4. Check if layer already exists for THIS sample, if not ADD IT
    shapes_layer = None
    for layer in viewer.layers:
        # Check if layer name exactly matches the sample's element name
        if layer.name == shapes_element_name and str(type(layer).__name__) == 'Points':
            shapes_layer = layer
            break
    
    if shapes_layer is None:
        # ADD the layer using THE EXACT SAME METHOD as the UI
        print(f"Adding layer: {shapes_element_name}")
        
        try:
            # Import SpatialDataViewer and create/get the viewer model
            from napari_spatialdata._viewer import SpatialDataViewer
            
            # Create a SpatialDataViewer instance wrapping our viewer
            viewer_model = SpatialDataViewer(viewer, [sdata])
            
            # Find the correct coordinate system for this element
            shapes_element = sdata.shapes[shapes_element_name]
            
            # Get coordinate systems from the element
            if hasattr(shapes_element, 'attrs') and 'transform' in shapes_element.attrs:
                # Get available coordinate systems
                available_cs = list(shapes_element.attrs['transform'].keys())
                selected_cs = available_cs[0] if available_cs else 'global'
            else:
                # Fallback to sdata coordinate systems
                available_cs = list(sdata.coordinate_systems)
                selected_cs = available_cs[0] if available_cs else 'global'
            
            multi = False  # single sdata object
            
            # This returns a Points or Shapes layer with proper transforms
            shapes_layer = viewer_model.get_sdata_circles(sdata, shapes_element_name, selected_cs, multi)
            
            # Add it to the viewer
            viewer.add_layer(shapes_layer)
            
            print(f"✓ Added layer: {shapes_element_name}")
            
        except Exception as e:
            import traceback
            print(f"Error: Could not add layer automatically: {e}")
            print(traceback.format_exc())
            print(f"Please add the '{shapes_element_name}' layer manually from the UI first, then run plot() again.")
            return
    
    # 5. Get gene expression from table that matches this sample
    if not hasattr(sdata, 'tables') or len(sdata.tables) == 0:
        print("Error: No expression table found in the data.")
        return
    
    # Find the table that corresponds to this sample
    table_name = None
    for tbl_name in sdata.tables.keys():
        if selected_sample != 'default' and tbl_name.endswith(selected_sample):
            table_name = tbl_name
            break
    
    if table_name is None:
        # Fallback to first table if no match
        table_name = list(sdata.tables.keys())[0]
    
    table = sdata.tables[table_name]
    
    # Verify that the table size matches the shapes layer
    if len(table) != len(sdata.shapes[shapes_element_name]):
        print(f"Warning: Table has {len(table)} cells but shapes layer has {len(sdata.shapes[shapes_element_name])} cells")
        print("This may cause issues. Make sure the correct sample is selected.")
    
    # 6. Get gene expression values
    if gene_name not in table.var_names:
        print(f"Error: Gene '{gene_name}' not found.")
        print(f"Available genes (first 20): {list(table.var_names[:20])}")
        return
    
    gene_idx = list(table.var_names).index(gene_name)
    expression = table.X[:, gene_idx]
    
    # Convert to dense if sparse
    if hasattr(expression, 'toarray'):
        expression = expression.toarray().flatten()
    else:
        expression = np.asarray(expression).flatten()
    
    # 7. Update the shapes layer with gene colors (don't rename or remove to avoid breaking napari-spatialdata tracking)
    # Parse vmin and vmax
    vmin_val = parse_vlim(vmin, expression)
    vmax_val = parse_vlim(vmax, expression)
    
    if vmin_val is None:
        vmin_val = expression.min()
    if vmax_val is None:
        vmax_val = expression.max()
    
    # Validate and fix vmax if needed
    if vmax_val <= vmin_val:
        vmax_val = vmin_val + 1
        print(f"Warning: vmax collapsed to vmin, adjusted to vmax = {vmax_val:.2f}")
    
    # Normalize expression values to [0, 1]
    norm_values = np.clip((expression - vmin_val) / (vmax_val - vmin_val), 0, 1)
    
    # Apply colormap
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    colors = cmap(norm_values)
    
    # Update layer colors (keep original name to avoid breaking napari-spatialdata tracking)
    shapes_layer.face_color = colors
    
    # Store in features for interactive selection
    if not hasattr(shapes_layer, 'features'):
        shapes_layer.features = {}
    shapes_layer.features[gene_name] = expression
    
    print(f"✓ Updated {len(expression)} cells with {gene_name} expression")
    print(f"  Data range: {expression.min():.2f} - {expression.max():.2f}")
    print(f"  Display range: vmin={vmin_val:.2f}, vmax={vmax_val:.2f}")


# Also provide sdata for direct access
def get_sdata():
    """Get the spatial data object."""
    from napari_spatial_tools._session import get_session_data
    return get_session_data()
