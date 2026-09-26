# Results

Both models are ImageNet-pretrained and fine-tuned for 15 epochs with one
recipe: AdamW, one-cycle learning rate peaking at 1e-3, batch 64, label
smoothing 0.05, seed 0. The checkpoint and the temperature are chosen on
the validation split, and the test split (2,562 images) is scored once.
One seed each, so small gaps are noise.

## Internal test (PBC)

| Model | Accuracy | Balanced acc. | Macro-F1 | ECE, raw → T-scaled | Errors |
|---|--:|--:|--:|--:|--:|
| MobileNetV3-Small | 98.8% | 98.9% | 98.9% | 4.4% → 0.6% | 31 |
| EfficientNet-B0 | 99.2% | 99.3% | 99.3% | 4.3% → 0.4% | 21 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/mobilenet_v3_small/confusion-matrix-dark.png">
  <img alt="MobileNetV3-Small confusion matrix on the test set: every class at 96% or higher on the diagonal; the largest error is 3.5% of immature granulocytes predicted as neutrophils." src="figures/mobilenet_v3_small/confusion-matrix.png" width="520">
</picture>

**Errors.** PBC's file names carry sub-types, and most errors sit between
two of them:

| Model | Metamyelocyte (IG) → neutrophil | Band neutrophil → IG | Other |
|---|--:|--:|--:|
| MobileNetV3-Small | 14 of 148 | 4 of 254 | 13 |
| EfficientNet-B0 | 5 of 148 | 8 of 254 | 8 |

These are neighbouring maturation stages, told apart by how deeply the
nucleus is indented. The models draw the line in different places. Neither
misclassifies a segmented neutrophil.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/mobilenet_v3_small/misclassified-dark.png">
  <img alt="MobileNetV3-Small's 12 most confident test mistakes, mostly immature granulocytes and band neutrophils with a kidney- or band-shaped nucleus." src="figures/mobilenet_v3_small/misclassified.png" width="680">
</picture>

**Calibration.** Both models are underconfident (T = 0.61 and 0.64), partly
from label smoothing. Temperature scaling fixes the average, not single
images: the worst mistakes still come at 97–100% confidence. Sending images
below 90% confidence to review would catch 14 of MobileNet's 31 errors for
1.6% of images flagged.

## External test (Raabin-WBC)

The same checkpoints, without retraining, on Raabin-WBC Test-A: 4,339 cells
from a lab in Iran with a different microscope, camera and stain, labelled
by two experts. It has 5 of PBC's 8 classes.

| Balanced accuracy | PBC test | Raabin, all 8 classes | Raabin, forced among 5 | Raabin ECE, raw → T |
|---|--:|--:|--:|--:|
| MobileNetV3-Small | 98.9% | 38.5% | 53.0% | 8.2% → 16.0% |
| EfficientNet-B0 | 99.3% | 56.5% | 63.5% | 12.1% → 19.7% |

"All 8 classes" is the model as deployed: predicting a class Raabin doesn't
have counts as wrong (25% of cells for MobileNet, 12% for EfficientNet).
"Forced among 5" isolates the change of lab. Chance is 20%.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/mobilenet_v3_small/external-reliability-dark.png">
  <img alt="MobileNetV3-Small reliability, temperature-scaled: close to the diagonal on PBC (ECE 0.6%), well below it on Raabin-WBC, where cells predicted at 98% confidence are right about 76% of the time (ECE 16.0%)." src="figures/mobilenet_v3_small/external-reliability.png" width="460">
</picture>

**Caveats.** Test-A was imaged on the same equipment as Raabin's own
training set; Test-B (another microscope) isn't in the copy used. The
numeric-label to class mapping is inferred from the paper's class counts
and checked by eye. Crops also differ in scale (575 px vs 360 px, both
resized to 224 px), so stain is the likeliest cause, not a proven one.
