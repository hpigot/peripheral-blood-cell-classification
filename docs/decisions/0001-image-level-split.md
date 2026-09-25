# 1. Image-level train/val/test split

Status: accepted

## Context

The PBC dataset has 17,092 single-cell images in 8 class folders. It has no
patient, smear or slide IDs, so we can't group images by source. Cells from
the same smear share stain, illumination and the patient's morphology, so if
they land in both train and test, test scores come out optimistic.

## Decision

Split at image level: stratified by class, 70/15/15, fixed seed
(`bloodcell-split`). Commit the result as `data/splits.csv` so every run and
the edge evaluation use exactly the same test images.

## Consequences

- Internal test scores are an upper bound, not an estimate of performance on
  new patients. The README says so.
- Generalization is measured separately, with an external test on a dataset
  from a different lab and scanner (Raabin-WBC, the 5 shared WBC classes).
- If a future PBC release adds grouping IDs, switch to a grouped split and
  supersede this ADR.
