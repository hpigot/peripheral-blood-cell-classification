# Data (not committed)

1. Download the PBC dataset from https://data.mendeley.com/datasets/snkd93bnjr/1
2. Unzip here so the layout is `data/PBC_dataset_normal_DIB/<class>/*.jpg`
3. `uv run bloodcell-split` drops non-images and duplicate images (it
   prints each one; see docs/decisions/0004-deduplicate-before-splitting.md)
   and writes `data/splits.csv`. That file is committed, so every run uses
   the same test images.
