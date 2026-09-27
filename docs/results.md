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
and checked by eye.

## Is it the stain?

Raabin's nuclei are magenta where PBC's are blue-purple, and its cells sit
about 7% larger in the frame (median lymphocyte nucleus 65 px vs 60 px at
224 px). `bloodcell-shift` corrects each at test time, towards PBC, with no
retraining: global colour (Reinhard, CIELAB mean and spread), the stain
colours themselves (Macenko), and cell size.

| Raabin balanced accuracy, all 8 classes | MobileNetV3-Small | EfficientNet-B0 |
|---|--:|--:|
| As is | 38.5% | 56.5% |
| Global colour (Reinhard) | 27.4% | 35.6% |
| Stain colours (Macenko) | 30.7% | 24.6% |
| Cell size (× 0.93) | 35.2% | 54.6% |
| Stain colours and size | 30.4% | 23.5% |

No correction helps. Colour is behind the shortcut: after either colour
correction, MobileNet calls fewer than 5% of Raabin's monocytes
eosinophils, down from 54%. But the errors move rather than go away:
after Macenko, 53% (MobileNet) and 66% (EfficientNet) of lymphocytes are
called erythroblasts. The corrections also cost 1.5–4.2 points on PBC's own
test split, so part of the loss is the correction's, not the lab's. Size
is too small a difference to matter.

Training for it instead doesn't help either. MobileNet retrained with
much stronger colour jitter (`--colour-jitter strong`: hue ±36° instead
of ±11°, saturation ±40% instead of ±15%; one seed) scores 98.7% on PBC
and 36.4% on Raabin, against 38.5% before. The pattern repeats: under 1%
of monocytes are called eosinophils, but 0.3% of eosinophils are
recognised, and 91% of them are called neutrophils when forced to one of
the five shared classes. Taught to ignore colour, the model loses the
cue it used for eosinophils, their pink-orange granules. Plain accuracy
rises from 52% to 68% only because Raabin is mostly neutrophils.

So the drop isn't a colour cast that preprocessing or augmentation can
remove. The likely rest (Raabin's images are softer, and camera and
smear preparation differ) can't be corrected image by image, which
points to training on more than one lab's data.
