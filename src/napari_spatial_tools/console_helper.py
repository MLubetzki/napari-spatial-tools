"""
Helper module for napari console.

Usage:
    from napari_spatial_tools.console_helper import plot, score, total_counts, compare_genes, compare_samples
    plot('CD8A', sample=0)
    score(['CD3D', 'CD3E'], sample=0, normalize='max', name='T_cell')
    total_counts(sample=0, log_scale=True)
    compare_genes(['CD8A', 'EPCAM'], sample=0)
    compare_samples('CD8A', samples=[0, 1])
"""

import numpy as np
import pandas as pd
import napari
from napari.layers import Points, Image
from napari_spatialdata._viewer import SpatialDataViewer
from matplotlib import cm
import math


# ============================================================================
# Helper Functions
# ============================================================================

def get_sdata():
    """Get the current SpatialData object."""
    from napari_spatial_tools._session import get_session_data
    return get_session_data()


def _get_viewer():
    """Get napari viewer or raise error."""
    viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("No napari viewer found.")
    return viewer


def _parse_vlim(value, data):
    """Parse vmin/vmax: numeric value or percentile string like 'p95'."""
    if value is None:
        return None
    if isinstance(value, str) and value.startswith('p'):
        return np.percentile(data, float(value[1:]))
    return float(value)


def _resolve_sample_index(sample_param, available_samples):
    """Convert sample index to name."""
    if isinstance(sample_param, int):
        if not 0 <= sample_param < len(available_samples):
            raise ValueError(f"Sample index {sample_param} out of bounds (0-{len(available_samples)-1}).")
        return available_samples[sample_param]
    return sample_param


def _find_element(sdata, sample_name, element_type, keywords=None):
    """Find element name by sample and optional keywords."""
    container = getattr(sdata, element_type)
    for name in container.keys():
        if sample_name not in name:
            continue
        if keywords is None:
            return name
        if any(kw.lower() in name.lower() for kw in keywords):
            return name
    return None


def _get_coordinate_system(sdata, element_name, element_type='shapes'):
    """Get correct coordinate system for an element."""
    container = getattr(sdata, element_type)
    element = container[element_name]
    
    if hasattr(element, 'attrs') and 'transform' in element.attrs:
        available_cs = list(element.attrs['transform'].keys())
    else:
        available_cs = list(sdata.coordinate_systems)
    
    return available_cs[0] if available_cs else 'global'


def _cleanup_comparison_layers(viewer):
    """Remove all comparison layers."""
    prefixes = ('sample_', 'dapi_sample_', 'gene_', 'dapi_')
    for layer in list(viewer.layers):
        if any(layer.name.startswith(p) for p in prefixes):
            try:
                del viewer.layers[layer.name]
            except:
                pass


def _compute_grid_shape(n_items, orientation='horizontal'):
    """Calculate optimal grid shape."""
    if n_items <= 3:
        return (1, n_items) if orientation == 'horizontal' else (n_items, 1)
    if n_items == 4:
        return (2, 2)
    
    rows = int(math.sqrt(n_items))
    cols = math.ceil(n_items / rows)
    while rows * cols < n_items:
        cols += 1
    
    return (rows, cols) if orientation == 'horizontal' else (cols, rows)


def _apply_gene_colors(layer, expression, vmin, vmax, gene_name):
    """Apply colormap to layer based on expression values."""
    vmin_val = _parse_vlim(vmin, expression) or expression.min()
    vmax_val = _parse_vlim(vmax, expression) or expression.max()
    
    if vmax_val <= vmin_val:
        vmax_val = vmin_val + 1
        print(f"Warning: vmax collapsed, adjusted to {vmax_val:.2f}")
    
    norm_values = np.clip((expression - vmin_val) / (vmax_val - vmin_val), 0, 1)
    layer.face_color = cm.get_cmap('viridis')(norm_values)
    
    if not hasattr(layer, 'features') or layer.features is None:
        layer.features = pd.DataFrame()
    layer.features[gene_name] = expression
    
    print(f"  ✓ {len(expression)} cells | range: {expression.min():.2f}-{expression.max():.2f} | display: {vmin_val:.2f}-{vmax_val:.2f}")


# ============================================================================
# Main Functions
# ============================================================================

def compare_samples(gene_name, samples, vmin=None, vmax=None, orientation='horizontal', global_scale=True):
    """
    Compare the same gene across multiple samples in grid view.
    
    Parameters
    ----------
    gene_name : str
        Gene name to visualize
    samples : list of int or str
        Sample indices or names to compare
    vmin, vmax : float or str, optional
        Color range limits (e.g., 'p95' for 95th percentile)
    orientation : str
        'horizontal' (default) or 'vertical' grid layout
    global_scale : bool
        If True (default), use same vmin/vmax for all samples for comparison.
        If False, compute vmin/vmax separately for each sample.
    
    Examples
    --------
    >>> compare_samples('CD8A', samples=[0, 1])  # Same scale for both
    >>> compare_samples('EPCAM', samples=[0, 1], global_scale=False)  # Independent scales
    >>> compare_samples('CD8A', samples=[0, 1, 2], vmin='p5', vmax='p95')
    """
    viewer = _get_viewer()
    from napari_spatial_tools._session import get_session_data, get_sample_order
    sdata = get_session_data()
    available_samples = get_sample_order()
    
    _cleanup_comparison_layers(viewer)
    for layer in viewer.layers:
        layer.visible = False
    
    viewer_model = SpatialDataViewer(viewer, [sdata])
    dapi_layers = []
    gene_layers = []
    sample_data = []  # Store (sample_idx, sample_name, dapi_name, shapes_name, table_name, cs, gene_layer, expression)
    
    # First pass: collect all expression data
    for idx, sample_param in enumerate(samples):
        sample_name = _resolve_sample_index(sample_param, available_samples)
        
        dapi_name = _find_element(sdata, sample_name, 'images', ['dapi', 'morphology_focus'])
        shapes_name = _find_element(sdata, sample_name, 'shapes', ['cell_circles'])
        table_name = _find_element(sdata, sample_name, 'tables')
        
        if not shapes_name or not table_name:
            print(f"Error: Missing data for sample {sample_name}")
            continue
        
        cs = _get_coordinate_system(sdata, shapes_name)
        
        table = sdata.tables[table_name]
        if gene_name not in table.var_names:
            print(f"Error: Gene '{gene_name}' not in sample {sample_name}")
            continue
        
        expression = table[:, gene_name].X.toarray().flatten()
        gene_layer = viewer_model.get_sdata_circles(sdata, shapes_name, cs, False)
        gene_layer.name = f"sample_{idx}_{gene_name}"
        
        sample_data.append((idx, sample_name, dapi_name, cs, gene_layer, expression))
    
    # Compute global vmin/vmax if needed
    if global_scale and sample_data:
        all_expression = np.concatenate([data[5] for data in sample_data])
        global_vmin = _parse_vlim(vmin, all_expression) or all_expression.min()
        global_vmax = _parse_vlim(vmax, all_expression) or all_expression.max()
        if global_vmax <= global_vmin:
            global_vmax = global_vmin + 1
        print(f"Using global scale: vmin={global_vmin:.2f}, vmax={global_vmax:.2f}")
    
    # Second pass: create layers and apply colors
    for idx, sample_name, dapi_name, cs, gene_layer, expression in sample_data:
        print(f"Creating layers for sample: {sample_name}")
        
        if dapi_name:
            dapi_layer = viewer_model.get_sdata_image(sdata, dapi_name, cs, False)
            dapi_layer.name = f"dapi_sample_{idx}"
            dapi_layer.colormap = 'gray'
            dapi_layer.blending = 'additive'
            dapi_layers.append(dapi_layer)
        
        if global_scale:
            _apply_gene_colors(gene_layer, expression, global_vmin, global_vmax, gene_name)
        else:
            _apply_gene_colors(gene_layer, expression, vmin, vmax, gene_name)
        
        gene_layers.append(gene_layer)
    
    for layer in dapi_layers + gene_layers:
        viewer.add_layer(layer)
        layer.visible = True
    
    viewer.grid.enabled = True
    viewer.grid.shape = _compute_grid_shape(len(samples), orientation)
    rows, cols = viewer.grid.shape
    scale_mode = "global scale" if global_scale else "independent scales"
    print(f"✓ Comparing {gene_name} across {len(samples)} samples ({rows}x{cols} grid, {scale_mode})")


def compare_genes(genes, sample=None, vmin=None, vmax=None, orientation='horizontal'):
    """
    Compare multiple genes side-by-side in grid view.
    
    Examples
    --------
    >>> compare_genes(['CD8A', 'EPCAM', 'PTPRC'], sample=0)
    >>> compare_genes(['CD8A', 'EPCAM'], sample=0, vmin='p5', vmax='p95')
    """
    viewer = _get_viewer()
    _cleanup_comparison_layers(viewer)
    
    for gene in genes:
        plot(gene, sample=sample, vmin=vmin, vmax=vmax, _internal_call=True)
        
        base_layer = None
        for layer in viewer.layers:
            if type(layer).__name__ == 'Points' and 'cell_circles' in layer.name:
                base_layer = layer
                break
        
        if not base_layer:
            print(f"Warning: Could not find base layer for {gene}")
            continue
        
        gene_layer = Points(
            base_layer.data,
            name=f"gene_{gene}",
            face_color=base_layer.face_color.copy(),
            size=base_layer.size,
            border_width=base_layer.border_width,
            border_color=base_layer.border_color,
            affine=base_layer.affine,
            metadata=base_layer.metadata.copy() if hasattr(base_layer, 'metadata') else {}
        )
        viewer.add_layer(gene_layer)
        print(f"  Created layer: gene_{gene}")
    
    # Remove base layers
    for layer in list(viewer.layers):
        if 'cell_circles' in layer.name and not layer.name.startswith('gene_'):
            try:
                del viewer.layers[layer.name]
            except:
                pass
    
    # Duplicate DAPI for each gene
    dapi_layer = None
    for layer in viewer.layers:
        if ('morphology' in layer.name.lower() or 'dapi' in layer.name.lower()) and not layer.name.startswith('dapi_'):
            dapi_layer = layer
            break
    
    if dapi_layer:
        gene_layers = [(g, l) for g in genes for l in viewer.layers if l.name == f"gene_{g}"]
        
        for gene, layer in gene_layers:
            viewer.layers.remove(layer)
        
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
            viewer.add_layer(dapi_copy)
        
        for gene, layer in gene_layers:
            viewer.add_layer(layer)
        
        try:
            del viewer.layers[dapi_layer.name]
        except:
            dapi_layer.visible = False
    
    for layer in viewer.layers:
        if layer.name.startswith('gene_') or layer.name.startswith('dapi_'):
            layer.visible = True
    
    viewer.grid.enabled = True
    viewer.grid.shape = _compute_grid_shape(len(genes), orientation)
    rows, cols = viewer.grid.shape
    print(f"✓ Comparing {len(genes)} genes ({rows}x{cols} grid)")


def score(gene_list, sample=None, vmin=None, vmax=None, name=None, normalize=None, percentile=95):
    """
    Plot a gene score (sum of multiple genes) on cells.
    
    Parameters
    ----------
    gene_list : list of str
        List of gene names to sum
    sample : int or str, optional
        Sample index (0, 1, ...) or full name
    vmin, vmax : float or str, optional
        Color range limits. Use 'p95' for 95th percentile
    name : str, optional
        Name for the score (default: "gene1+gene2+...")
    normalize : str, optional
        Normalization method before summing. Options:
        - None: no normalization, sum raw counts (default)
        - 'max': divide each gene by its maximum value
        - 'percentile': divide each gene by its percentile value
        - 'zscore': z-score normalization (value - mean) / std
    percentile : float, optional
        Percentile to use for 'percentile' normalization (default: 95)
    
    Examples
    --------
    >>> score(['CD3D', 'CD3E', 'CD8A'], sample=0, name='T_cell_score')
    >>> score(['MKI67', 'TOP2A'], sample=0, normalize='max', name='Prolif')
    >>> score(['CD3D', 'CD8A'], sample=0, normalize='percentile', percentile=95)
    >>> score(['EPCAM', 'KRT8'], sample=0, normalize='zscore', name='Epithelial')
    """
    viewer = _get_viewer()
    sdata = get_sdata()
    
    viewer.grid.enabled = False
    _cleanup_comparison_layers(viewer)
    
    # Use plot logic to get the correct sample and layer
    all_shape_names = list(sdata.shapes.keys())
    samples = {}
    for shape_name in all_shape_names:
        if '-' in shape_name:
            parts = shape_name.split('-', 1)
            if len(parts) == 2:
                sample_suffix = parts[1]
                if sample_suffix not in samples:
                    samples[sample_suffix] = []
                samples[sample_suffix].append(shape_name)
    
    if not samples:
        samples['default'] = all_shape_names
    
    if sample is None:
        from napari_spatial_tools._session import get_sample_order
        sample_order = get_sample_order()
        if sample_order and len(sample_order) == 1:
            sample = 0
        elif len(samples) == 1:
            selected_sample = list(samples.keys())[0]
        else:
            print(f"Error: Multiple samples found. Use: score({gene_list}, sample=INDEX_OR_NAME)")
            print(f"Available: {list(samples.keys())}")
            return
    
    if sample is not None:
        from napari_spatial_tools._session import get_sample_order
        sample_order = get_sample_order()
        
        if isinstance(sample, int):
            if 0 <= sample < len(sample_order):
                selected_sample = sample_order[sample]
            else:
                print(f"Error: Sample index {sample} out of bounds")
                return
        else:
            if sample not in samples:
                print(f"Error: Sample '{sample}' not found")
                return
            selected_sample = sample
    
    shapes_element_name = None
    candidate_names = samples[selected_sample] if selected_sample != 'default' else all_shape_names
    
    for shape_name in candidate_names:
        if 'circle' in shape_name.lower():
            shapes_element_name = shape_name
            break
    
    if shapes_element_name is None:
        for shape_name in candidate_names:
            if 'cell' in shape_name.lower() and 'boundar' not in shape_name.lower():
                shapes_element_name = shape_name
                break
    
    if shapes_element_name is None:
        shapes_element_name = candidate_names[0] if candidate_names else list(sdata.shapes.keys())[0]
    
    print(f"Using sample: {selected_sample}, layer: {shapes_element_name}")
    
    # Hide other samples
    all_samples = list(samples.keys())
    for layer in viewer.layers:
        layer_sample = None
        for sample_name in all_samples:
            if sample_name != 'default' and sample_name in layer.name:
                layer_sample = sample_name
                break
        
        if layer_sample is not None:
            layer.visible = (layer_sample == selected_sample)
    
    # Add DAPI if needed
    if hasattr(sdata, 'images') and len(sdata.images) > 0:
        for img_name in sdata.images.keys():
            if selected_sample != 'default' and not img_name.endswith(selected_sample):
                continue
            
            if any(kw in img_name.lower() for kw in ['morphology', 'dapi', 'focus']):
                if img_name not in [layer.name for layer in viewer.layers]:
                    try:
                        viewer_model = SpatialDataViewer(viewer, [sdata])
                        img_cs = _get_coordinate_system(sdata, img_name, 'images')
                        img_layer = viewer_model.get_sdata_image(sdata, img_name, img_cs, False)
                        viewer.add_layer(img_layer)
                        print(f"✓ Added DAPI layer: {img_name}")
                    except Exception as e:
                        print(f"Note: Could not add DAPI: {e}")
                break
    
    # Get or create shapes layer
    shapes_layer = None
    for layer in viewer.layers:
        if layer.name == shapes_element_name and type(layer).__name__ == 'Points':
            shapes_layer = layer
            break
    
    if shapes_layer is None:
        print(f"Adding layer: {shapes_element_name}")
        try:
            viewer_model = SpatialDataViewer(viewer, [sdata])
            cs = _get_coordinate_system(sdata, shapes_element_name)
            shapes_layer = viewer_model.get_sdata_circles(sdata, shapes_element_name, cs, False)
            viewer.add_layer(shapes_layer)
            print(f"✓ Added layer: {shapes_element_name}")
        except Exception as e:
            import traceback
            print(f"Error: Could not add layer: {e}")
            print(traceback.format_exc())
            return
    
    # Get table
    if not hasattr(sdata, 'tables') or len(sdata.tables) == 0:
        print("Error: No expression table found")
        return
    
    table_name = None
    for tbl_name in sdata.tables.keys():
        if selected_sample != 'default' and tbl_name.endswith(selected_sample):
            table_name = tbl_name
            break
    
    if table_name is None:
        table_name = list(sdata.tables.keys())[0]
    
    table = sdata.tables[table_name]
    
    # Sum expression for all genes (with optional normalization)
    score_values = np.zeros(table.shape[0])
    found_genes = []
    missing_genes = []
    
    for gene_name in gene_list:
        if gene_name in table.var_names:
            gene_idx = list(table.var_names).index(gene_name)
            expression = table.X[:, gene_idx]
            
            if hasattr(expression, 'toarray'):
                expression = expression.toarray().flatten()
            else:
                expression = np.asarray(expression).flatten()
            
            # Apply normalization
            if normalize == 'max':
                max_val = expression.max()
                if max_val > 0:
                    expression = expression / max_val
            elif normalize == 'percentile':
                percentile_val = np.percentile(expression, percentile)
                if percentile_val > 0:
                    expression = expression / percentile_val
            elif normalize == 'zscore':
                mean_val = expression.mean()
                std_val = expression.std()
                if std_val > 0:
                    expression = (expression - mean_val) / std_val
                else:
                    expression = expression - mean_val
            
            score_values += expression
            found_genes.append(gene_name)
        else:
            missing_genes.append(gene_name)
    
    if not found_genes:
        print(f"Error: None of the genes found in the data")
        print(f"Available (first 20): {list(table.var_names[:20])}")
        return
    
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes not found: {missing_genes}")
    
    # Apply colors
    vmin_val = _parse_vlim(vmin, score_values) or score_values.min()
    vmax_val = _parse_vlim(vmax, score_values) or score_values.max()
    
    if vmax_val <= vmin_val:
        vmax_val = vmin_val + 1
        print(f"Warning: vmax collapsed, adjusted to {vmax_val:.2f}")
    
    norm_values = np.clip((score_values - vmin_val) / (vmax_val - vmin_val), 0, 1)
    shapes_layer.face_color = cm.get_cmap('viridis')(norm_values)
    
    # Store score in features
    score_name = name if name else '+'.join(gene_list)
    if not hasattr(shapes_layer, 'features'):
        shapes_layer.features = {}
    shapes_layer.features[score_name] = score_values
    
    norm_method = normalize if normalize else 'raw counts'
    if normalize == 'percentile':
        norm_method = f'percentile (p{percentile})'
    
    print(f"✓ Updated {len(score_values)} cells with score '{score_name}'")
    print(f"  Genes used ({len(found_genes)}): {', '.join(found_genes)}")
    print(f"  Normalization: {norm_method}")
    print(f"  Score range: {score_values.min():.2f} - {score_values.max():.2f}")
    print(f"  Display range: vmin={vmin_val:.2f}, vmax={vmax_val:.2f}")


def total_counts(sample=None, vmin=None, vmax=None, log_scale=False):
    """
    Plot total RNA counts per cell.
    
    Parameters
    ----------
    sample : int or str, optional
        Sample index (0, 1, ...) or full name
    vmin, vmax : float or str, optional
        Color range limits. Use 'p95' for 95th percentile
    log_scale : bool
        If True, display log10(counts + 1) instead of raw counts
    
    Examples
    --------
    >>> total_counts(sample=0)
    >>> total_counts(sample=0, log_scale=True)
    >>> total_counts(sample=0, vmin='p5', vmax='p95', log_scale=True)
    """
    viewer = _get_viewer()
    sdata = get_sdata()
    
    viewer.grid.enabled = False
    _cleanup_comparison_layers(viewer)
    
    all_shape_names = list(sdata.shapes.keys())
    samples = {}
    for name in all_shape_names:
        if '-' in name:
            parts = name.split('-', 1)
            if len(parts) == 2:
                sample_suffix = parts[1]
                if sample_suffix not in samples:
                    samples[sample_suffix] = []
                samples[sample_suffix].append(name)
    
    if not samples:
        samples['default'] = all_shape_names
    
    if sample is None:
        from napari_spatial_tools._session import get_sample_order
        sample_order = get_sample_order()
        if sample_order and len(sample_order) == 1:
            sample = 0
        elif len(samples) == 1:
            selected_sample = list(samples.keys())[0]
        else:
            print(f"Error: Multiple samples found. Use: total_counts(sample=INDEX_OR_NAME)")
            print(f"Available: {list(samples.keys())}")
            return
    
    if sample is not None:
        from napari_spatial_tools._session import get_sample_order
        sample_order = get_sample_order()
        
        if isinstance(sample, int):
            if 0 <= sample < len(sample_order):
                selected_sample = sample_order[sample]
            else:
                print(f"Error: Sample index {sample} out of bounds")
                return
        else:
            if sample not in samples:
                print(f"Error: Sample '{sample}' not found")
                return
            selected_sample = sample
    
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
    
    all_samples = list(samples.keys())
    for layer in viewer.layers:
        layer_sample = None
        for sample_name in all_samples:
            if sample_name != 'default' and sample_name in layer.name:
                layer_sample = sample_name
                break
        
        if layer_sample is not None:
            layer.visible = (layer_sample == selected_sample)
    
    if hasattr(sdata, 'images') and len(sdata.images) > 0:
        for img_name in sdata.images.keys():
            if selected_sample != 'default' and not img_name.endswith(selected_sample):
                continue
            
            if any(kw in img_name.lower() for kw in ['morphology', 'dapi', 'focus']):
                if img_name not in [layer.name for layer in viewer.layers]:
                    try:
                        viewer_model = SpatialDataViewer(viewer, [sdata])
                        img_cs = _get_coordinate_system(sdata, img_name, 'images')
                        img_layer = viewer_model.get_sdata_image(sdata, img_name, img_cs, False)
                        viewer.add_layer(img_layer)
                        print(f"✓ Added DAPI layer: {img_name}")
                    except Exception as e:
                        print(f"Note: Could not add DAPI: {e}")
                break
    
    shapes_layer = None
    for layer in viewer.layers:
        if layer.name == shapes_element_name and type(layer).__name__ == 'Points':
            shapes_layer = layer
            break
    
    if shapes_layer is None:
        print(f"Adding layer: {shapes_element_name}")
        try:
            viewer_model = SpatialDataViewer(viewer, [sdata])
            cs = _get_coordinate_system(sdata, shapes_element_name)
            shapes_layer = viewer_model.get_sdata_circles(sdata, shapes_element_name, cs, False)
            viewer.add_layer(shapes_layer)
            print(f"✓ Added layer: {shapes_element_name}")
        except Exception as e:
            import traceback
            print(f"Error: Could not add layer: {e}")
            print(traceback.format_exc())
            return
    
    if not hasattr(sdata, 'tables') or len(sdata.tables) == 0:
        print("Error: No expression table found")
        return
    
    table_name = None
    for tbl_name in sdata.tables.keys():
        if selected_sample != 'default' and tbl_name.endswith(selected_sample):
            table_name = tbl_name
            break
    
    if table_name is None:
        table_name = list(sdata.tables.keys())[0]
    
    table = sdata.tables[table_name]
    
    # Calculate total counts (sum across all genes)
    total = np.asarray(table.X.sum(axis=1)).flatten()
    
    # Apply log scale if requested
    if log_scale:
        display_values = np.log10(total + 1)
        label = 'total_counts_log10'
    else:
        display_values = total
        label = 'total_counts'
    
    # Apply colors
    vmin_val = _parse_vlim(vmin, display_values) or display_values.min()
    vmax_val = _parse_vlim(vmax, display_values) or display_values.max()
    
    if vmax_val <= vmin_val:
        vmax_val = vmin_val + 1
        print(f"Warning: vmax collapsed, adjusted to {vmax_val:.2f}")
    
    norm_values = np.clip((display_values - vmin_val) / (vmax_val - vmin_val), 0, 1)
    shapes_layer.face_color = cm.get_cmap('viridis')(norm_values)
    
    if not hasattr(shapes_layer, 'features'):
        shapes_layer.features = {}
    shapes_layer.features[label] = total if not log_scale else display_values
    
    scale_info = " (log10 scale)" if log_scale else ""
    print(f"✓ Updated {len(total)} cells with total counts{scale_info}")
    print(f"  Raw counts range: {total.min():.0f} - {total.max():.0f}")
    if log_scale:
        print(f"  Log10 range: {display_values.min():.2f} - {display_values.max():.2f}")
    print(f"  Display range: vmin={vmin_val:.2f}, vmax={vmax_val:.2f}")


def plot(gene_name, sample=None, vmin=None, vmax=None, _internal_call=False):
    """
    Plot gene expression on cells.
    
    Parameters
    ----------
    gene_name : str
        Gene name to visualize
    sample : int or str, optional
        Sample index (0, 1, ...) or full name
    vmin, vmax : float or str, optional
        Color range limits. Use 'p95' for 95th percentile
    
    Examples
    --------
    >>> plot('CD8A')
    >>> plot('EPCAM', sample=0)
    >>> plot('PTPRC', sample=0, vmin='p5', vmax='p95')
    """
    viewer = _get_viewer()
    sdata = get_sdata()
    
    if not _internal_call:
        viewer.grid.enabled = False
        _cleanup_comparison_layers(viewer)
    
    all_shape_names = list(sdata.shapes.keys())
    samples = {}
    for name in all_shape_names:
        if '-' in name:
            parts = name.split('-', 1)
            if len(parts) == 2:
                sample_suffix = parts[1]
                if sample_suffix not in samples:
                    samples[sample_suffix] = []
                samples[sample_suffix].append(name)
    
    if not samples:
        samples['default'] = all_shape_names
    
    if sample is None:
        from napari_spatial_tools._session import get_sample_order
        sample_order = get_sample_order()
        if sample_order and len(sample_order) == 1:
            sample = 0
        elif len(samples) == 1:
            selected_sample = list(samples.keys())[0]
        else:
            print(f"Error: Multiple samples found. Use: plot('{gene_name}', sample=INDEX_OR_NAME)")
            print(f"Available: {list(samples.keys())}")
            return
    
    if sample is not None:
        from napari_spatial_tools._session import get_sample_order
        sample_order = get_sample_order()
        
        if isinstance(sample, int):
            if 0 <= sample < len(sample_order):
                selected_sample = sample_order[sample]
            else:
                print(f"Error: Sample index {sample} out of bounds")
                return
        else:
            if sample not in samples:
                print(f"Error: Sample '{sample}' not found")
                return
            selected_sample = sample
    
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
    
    all_samples = list(samples.keys())
    for layer in viewer.layers:
        layer_sample = None
        for sample_name in all_samples:
            if sample_name != 'default' and sample_name in layer.name:
                layer_sample = sample_name
                break
        
        if layer_sample is not None:
            layer.visible = (layer_sample == selected_sample)
    
    if hasattr(sdata, 'images') and len(sdata.images) > 0:
        for img_name in sdata.images.keys():
            if selected_sample != 'default' and not img_name.endswith(selected_sample):
                continue
            
            if any(kw in img_name.lower() for kw in ['morphology', 'dapi', 'focus']):
                if img_name not in [layer.name for layer in viewer.layers]:
                    try:
                        viewer_model = SpatialDataViewer(viewer, [sdata])
                        img_cs = _get_coordinate_system(sdata, img_name, 'images')
                        img_layer = viewer_model.get_sdata_image(sdata, img_name, img_cs, False)
                        viewer.add_layer(img_layer)
                        print(f"✓ Added DAPI layer: {img_name}")
                    except Exception as e:
                        print(f"Note: Could not add DAPI: {e}")
                break
    
    shapes_layer = None
    for layer in viewer.layers:
        if layer.name == shapes_element_name and type(layer).__name__ == 'Points':
            shapes_layer = layer
            break
    
    if shapes_layer is None:
        print(f"Adding layer: {shapes_element_name}")
        try:
            viewer_model = SpatialDataViewer(viewer, [sdata])
            cs = _get_coordinate_system(sdata, shapes_element_name)
            shapes_layer = viewer_model.get_sdata_circles(sdata, shapes_element_name, cs, False)
            viewer.add_layer(shapes_layer)
            print(f"✓ Added layer: {shapes_element_name}")
        except Exception as e:
            import traceback
            print(f"Error: Could not add layer: {e}")
            print(traceback.format_exc())
            return
    
    if not hasattr(sdata, 'tables') or len(sdata.tables) == 0:
        print("Error: No expression table found")
        return
    
    table_name = None
    for tbl_name in sdata.tables.keys():
        if selected_sample != 'default' and tbl_name.endswith(selected_sample):
            table_name = tbl_name
            break
    
    if table_name is None:
        table_name = list(sdata.tables.keys())[0]
    
    table = sdata.tables[table_name]
    
    if gene_name not in table.var_names:
        print(f"Error: Gene '{gene_name}' not found")
        print(f"Available (first 20): {list(table.var_names[:20])}")
        return
    
    gene_idx = list(table.var_names).index(gene_name)
    expression = table.X[:, gene_idx]
    
    if hasattr(expression, 'toarray'):
        expression = expression.toarray().flatten()
    else:
        expression = np.asarray(expression).flatten()
    
    vmin_val = _parse_vlim(vmin, expression) or expression.min()
    vmax_val = _parse_vlim(vmax, expression) or expression.max()
    
    if vmax_val <= vmin_val:
        vmax_val = vmin_val + 1
        print(f"Warning: vmax collapsed, adjusted to {vmax_val:.2f}")
    
    norm_values = np.clip((expression - vmin_val) / (vmax_val - vmin_val), 0, 1)
    shapes_layer.face_color = cm.get_cmap('viridis')(norm_values)
    
    if not hasattr(shapes_layer, 'features'):
        shapes_layer.features = {}
    shapes_layer.features[gene_name] = expression
    
    print(f"✓ Updated {len(expression)} cells with {gene_name} expression")
    print(f"  Data range: {expression.min():.2f} - {expression.max():.2f}")
    print(f"  Display range: vmin={vmin_val:.2f}, vmax={vmax_val:.2f}")
