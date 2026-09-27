#!/usr/bin/env bash
# Check that the C++ programs agree with Python, on synthetic JPEGs:
# 1. bloodcell_preprocess gives the same model input as torchvision, value by
#    value;
# 2. end to end, bloodcell_infer's labels and confidences match bloodcell-eval's
#    (split -> train -> eval -> export -> infer -> score). A briefly trained
#    model gives nearly the same output for any input, so step 2 checks the
#    plumbing (sidecar, temperature, output format) and step 1 the pixels.
# Usage: edge/parity.sh <cmake build dir>   (CI's edge-cpp job runs it)
set -euo pipefail

build="$(realpath "$1")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Noise JPEGs with a tint per class, most at PBC's size (360x363, so the resize
# shrinks) and some small (it enlarges).
uv run python - "$work/images" <<'EOF'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

tints = {"basophil": (90, 60, 150), "eosinophil": (220, 120, 110), "neutrophil": (170, 110, 170)}
for c, (cls, tint) in enumerate(tints.items()):
    (Path(sys.argv[1]) / cls).mkdir(parents=True)
    for i in range(14):
        rng = np.random.default_rng([c, i])
        h, w = (363, 360) if i % 5 else (100, 90)
        px = np.clip(rng.normal(tint, 50, (h, w, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(px).save(Path(sys.argv[1]) / cls / f"{i}.jpg", quality=90)
EOF

# 1. Model input
mkdir "$work/tensors"
for img in "$work"/images/*/*.jpg; do
  name="${img#"$work"/images/}"
  "$build/bloodcell_preprocess" "$img" "$work/tensors/${name//\//_}.f32"
done
uv run python - "$work" <<'EOF'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from bloodcell.model import INPUT_SIZE, transforms

work = Path(sys.argv[1])
tf = transforms(train=False)
worst = 0.0
for jpg in sorted((work / "images").glob("*/*.jpg")):
    ref = tf(Image.open(jpg).convert("RGB")).numpy()
    f32 = work / "tensors" / f"{jpg.parent.name}_{jpg.name}.f32"
    got = np.fromfile(f32, np.float32).reshape(3, INPUT_SIZE, INPUT_SIZE)
    worst = max(worst, float(np.abs(got - ref).max()))
print(f"model input: max |C++ - Python| = {worst:.1e}")
if worst > 1e-6:
    sys.exit("C++ preprocessing differs from torchvision's; see edge/cpp/preprocess.hpp")
EOF

# 2. End to end
common=(--root "$work/images" --splits "$work/splits.csv")
uv run bloodcell-split --root "$work/images" --out "$work/splits.csv" --test 0.5
uv run bloodcell-train "${common[@]}" --no-pretrained --epochs 1 --batch-size 8 \
  --workers 0 --out "$work/runs"
run="$(echo "$work"/runs/*)"
uv run bloodcell-eval "$run/best.pt" "${common[@]}"
uv run bloodcell-export "$run/best.pt" "${common[@]}"
"$build/bloodcell_infer" "$run/best.onnx" "$work/images" --threads 1 > "$run/preds.tsv"
uv run bloodcell-score "$run/preds.tsv" "${common[@]}" --against "$run/test_predictions.csv"
