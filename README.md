# napari-spatial-tools

Tools for visualizing spatial data in Napari.

## Installation

```bash
pip install .
```

## Usage

```bash
napari-all-slides /path/to/file1.zarr /path/to/file2.zarr /path/to/file3.zarr
```

### Visualizing genes from the napari console

Once the viewer is open, import the plot function in the napari console:

```python
# In the napari console - run this ONCE:
from napari_spatial_tools.console_helper import plot

# Then visualize genes:
plot('CD8A')     # Visualize CD8A gene expression on cells
plot('PECAM1')   # Visualize another gene
plot('KRT8')     # Visualize another gene
```

The `plot()` function will:
- Automatically find and add the DAPI channel (if available)
- Identify cell-circles/shapes in the data
- Extract gene expression from the associated table
- Display cells as points colored by the gene expression values

### Quick Start Example

```bash
# 1. Activate environment
conda activate napari-spatial

# 2. Launch viewer with your data
napari-all-slides /path/to/sample1.zarr /path/to/sample2.zarr

# 3. In the napari console that opens:
from napari_spatial_tools.console_helper import plot, get_sdata, compare_genes, compare_samples

# Single sample - auto-detected
plot('CD8A')

# Multiple samples - use index (0 = first file, 1 = second, etc.)
plot('CD8A', sample=0)  # First zarr file
plot('EPCAM', sample=1, vmin='p5', vmax='p95')  # Second zarr file

# Compare multiple genes on the same sample (split view)
compare_genes(['CD8A', 'EPCAM', 'PTPRC'], sample=0, vmin='p5', vmax='p95')

# Compare the same gene across different samples (split view)
compare_samples('CD8A', samples=[0, 1], vmin='p5', vmax='p95')

# Access the spatial data object for custom analysis
sdata = get_sdata()
print(sdata)
print(list(sdata.shapes.keys()))
print(list(sdata.tables.keys()))
```
