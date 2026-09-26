# Data (not committed)

1. Download the PBC dataset from https://data.mendeley.com/datasets/snkd93bnjr/1
2. Unzip here so the layout is `data/PBC_dataset_normal_DIB/<class>/*.jpg`
3. `uv run bloodcell-split` drops non-images and duplicate images (it
   prints each one; see docs/decisions/0004-deduplicate-before-splitting.md)
   and writes `data/splits.csv`. That file is committed, so every run uses
   the same test images.

## Raabin-WBC (external validation only)

Used to test the trained models on another lab's images; never for
training.

1. Download the double-labelled cropped cells, `Raabin-WBC.rar` (1.75 GB),
   from the Google Drive link in https://github.com/nimaadmed/WBC_Feature.
   The official copy at raabindata.com now only serves the full 56 GB
   archive. Our copy had SHA-256
   `8d41ceb93eb343574dc95d96c482a53b9feabb65298169d7fcaef1e2c252d5eb`.
2. Extract the test set so the layout is
   `data/raabin/Raabin-WBC/Test/*.jpg` plus `data/raabin/Raabin-WBC/Test.json`:
   `7z x Raabin-WBC.rar "Raabin-WBC/Test/*" "Raabin-WBC/Test.json"`
3. `uv run bloodcell-external runs/<run>/best.pt` (after `bloodcell-eval`)
