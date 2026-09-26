# 4. Deduplicate before splitting

Status: accepted

## Context

The PBC download (Mendeley Data, v1) has 17,093 files, one more than the
17,092 images the paper reports. A check before the first split found:

- `neutrophil/.DS_169665.jpg`: a macOS `.DS_Store` file saved with a `.jpg`
  name. It isn't an image, and it's the extra file.
- 17 groups of byte-identical images (34 files). 16 of them are within one
  class. In one pair the labels conflict:
  `eosinophil/EO_225902.jpg` and `neutrophil/BNE_191112.jpg` are the same
  file.

With an image-level split (ADR 0001), an identical copy can land in train
while the other is in test, so the model is scored on an image it trained
on. The conflicting pair also gives one cell two different labels.

A perceptual-hash check (64-bit dHash, Hamming distance ≤ 4) on the
remaining images found no near-duplicates, so the problem seems limited to
exact copies.

## Decision

`bloodcell-split` cleans the file list before splitting (`data.clean`):

- Skip hidden files, and drop files that don't open as images.
- Group files by SHA-256. For a same-class group, keep the first copy by
  sorted path and drop the rest.
- For a group with conflicting labels, drop every copy: we can't tell which
  label is right, and guessing would put a label we invented into the
  test set.

The result is 17,074 images. `bloodcell-split` prints every dropped file:

| Dropped | Reason |
|---|---|
| basophil/BA_685720.jpg | duplicate of basophil/BA_546746.jpg |
| basophil/BA_786369.jpg | duplicate of basophil/BA_225540.jpg |
| basophil/BA_80990.jpg | duplicate of basophil/BA_381452.jpg |
| basophil/BA_809918.jpg | duplicate of basophil/BA_20201.jpg |
| basophil/BA_858229.jpg | duplicate of basophil/BA_435917.jpg |
| basophil/BA_988506.jpg | duplicate of basophil/BA_251042.jpg |
| eosinophil/EO_225902.jpg | same image labelled eosinophil and neutrophil |
| eosinophil/EO_433229.jpg | duplicate of eosinophil/EO_220125.jpg |
| eosinophil/EO_728356.jpg | duplicate of eosinophil/EO_621854.jpg |
| eosinophil/EO_738445.jpg | duplicate of eosinophil/EO_284748.jpg |
| eosinophil/EO_873756.jpg | duplicate of eosinophil/EO_65638.jpg |
| eosinophil/EO_918881.jpg | duplicate of eosinophil/EO_890163.jpg |
| eosinophil/EO_994040.jpg | duplicate of eosinophil/EO_941266.jpg |
| ig/PMY_225787.jpg | duplicate of ig/PMY_187173.jpg |
| ig/PMY_586364.jpg | duplicate of ig/PMY_497134.jpg |
| ig/PMY_753076.jpg | duplicate of ig/PMY_538541.jpg |
| monocyte/MO_997225.jpg | duplicate of monocyte/MO_145357.jpg |
| neutrophil/BNE_191112.jpg | same image labelled eosinophil and neutrophil |

The `.DS_Store` file is skipped by `discover` and isn't listed.

## Consequences

- Our numbers aren't directly comparable with work that split the raw
  17,092 images. The difference is small (18 images), but a leaked test
  image is a correctness problem, not a rounding error.
- Deduplication hashes file bytes, so a re-encoded or resized copy would
  get through. The dHash check says there are none today. If the dataset
  changes, run it again.
- The conflicting pair is worth reporting to the dataset authors.
