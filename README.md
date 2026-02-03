# napari-spatial-tools

Vibe-coded tool for visualizing spatial data in Napari.
Allows you to quickly zoom and pan on your slides, while visualizing different genes with custom cmap range.

## Installation

```bash
pip install .
```

## Usage

```bash
napari-all-slides /path/to/file1.zarr /path/to/file2.zarr /path/to/file3.zarr
```

### Visualizing genes from the napari console

Once the viewer is open, import the plotting functions in the napari console:

```python
from napari_spatial_tools.console_helper import *

# Then visualize genes:
plot('EPCAM', vmax='p95', sample=0)

# To visualize multiple genes in a grid view:
compare_genes(['EPCAM', 'PTPRC'], sample=0)

# To compare a gene between different samples:
compare_samples('EPCAM', samples=[0,1,2,3])
# I didn't figure out how to disable grid pan locking here, TODO
```
