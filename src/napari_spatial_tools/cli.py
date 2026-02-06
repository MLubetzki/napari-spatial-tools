import sys
import spatialdata as sd
from napari_spatialdata import Interactive
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: napari-all-slides <zarr_file1> <zarr_file2> ...")

    zarr_paths = [Path(arg) for arg in sys.argv[1:]]
    sdata_dict = {f.stem: sd.read_zarr(f) for f in zarr_paths}

    if not sdata_dict:
        sys.exit("No .zarr files provided.")

    print(f"Loading {len(sdata_dict)} datasets...")
    combined_sdata = sd.concatenate(sdata_dict)
    print(combined_sdata)

    from napari_spatial_tools._session import set_session_data
    sample_order = list(sdata_dict.keys())
    set_session_data(combined_sdata, None, sample_order=sample_order)
    
    print("\nStarting napari viewer...")
    print("In the console, run:")
    print("  from napari_spatial_tools.console_helper import plot, score, total_counts, compare_genes, compare_samples")
    print("  plot('CD8A', sample=0)")
    print("  total_counts(sample=0, log_scale=True)")
    
    Interactive(combined_sdata)


if __name__ == "__main__":
    main()
