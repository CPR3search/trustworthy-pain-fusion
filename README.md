# Trustworthy Multimodal Fusion of Physiological Signals for Pain Decoding

## Layout

```
trustworthy_pain_fusion/
  data/          dataset loaders (AI4Pain 2026, AI4Pain 2025, BioVid Part A), per-subject
                 normalization and features
  methods/       usable information, predictor-independent tests, fusion operators, encoders,
                 EDA representation search, splits and metrics
  experiments/   one module per analysis; each writes an aggregate JSON file to results/
third_party/     the NormWear files used to compute frozen embeddings (Apache License 2.0)
env/             locked requirements for the two environments
```

# Installation

Two environments are used: `main` for all analyses and `fm` (PyTorch with CUDA) for the frozen
NormWear encoder. Both use Python 3.13.

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r env/requirements-lock-main.txt
python3.13 -m venv .venv-fm && .venv-fm/bin/pip install -r env/requirements-lock-fm.txt \
    --extra-index-url https://download.pytorch.org/whl/cu132
```

## Data

The datasets are not redistributed. Obtain them from their providers and place them as follows:

| Dataset | Source |
|---|---|
| AI4Pain 2026 | challenge organizers, on request |
| AI4Pain 2025 | challenge organizers, on request |
| BioVid Heat Pain, Part A | Ulm University, under a data-use agreement |

```
data/
  train/{Bvp,Eda,Resp,SpO2}/<subject>.csv              AI4Pain 2026 (same file name under each signal)
  validation/{Bvp,Eda,Resp,SpO2}/<subject>.csv
  ai4pain2025/train/{Bvp,Eda,Resp,SpO2}/<subject>.csv  AI4Pain 2025
  ai4pain2025/validation/{Bvp,Eda,Resp,SpO2}/<subject>.csv
  biovid/PartA/biosignals_filtered/<subject>/<subject>-<class>-<trial>_bio.csv
  biovid/excluded_subjects.txt                         20 subject IDs, one per line (see below)
```

Example for AI4Pain 2026 presence:

```python
import numpy as np
from trustworthy_pain_fusion.data import ai4pain2026 as d
from trustworthy_pain_fusion.methods.decodability import UsableInformationEstimator, ai4pain_targets

segs, subj, lab, _ = d.load_segments("data", "train", d.SIGNALS)    # preprocessing
segs = d.normalize_per_subject(segs, subj, d.SIGNALS)
X = d.build_features(segs, d.SIGNALS)
_, si = np.unique(np.asarray(subj), return_inverse=True)
y, mask = ai4pain_targets(np.asarray(lab, int))["presence (NoPain vs Pain)"]
profile = UsableInformationEstimator(k=5, seed=0).profile(X[mask], y, si[mask])  # training and inference
print(profile["usable_info_bits"], profile["cv_balanced_accuracy"])
```

## Training and Evaluation

| Stage | Command | Article content |
|---|---|---|
| decodability | `.venv/bin/python -m trustworthy_pain_fusion decodability` | Tables 2 and 3 |
| predictor-independent tests | `.venv/bin/python -m trustworthy_pain_fusion independence` | Table 8(a) |
| fusion | `.venv/bin/python -m trustworthy_pain_fusion fusion` | Fig. 2 and Tables 4-6 (reference partition) |
| resampling | `.venv/bin/python -m trustworthy_pain_fusion resampling` | Sections 4.6 and 4.8, Fig. 3 |
| frozen NormWear | `.venv-fm/bin/python -m trustworthy_pain_fusion normwear` | Table 11 |
| evaluation | `.venv/bin/python -m trustworthy_pain_fusion evaluation` | Tables 8(b), 9 and 10, Section 4.10 |
| EDA feature bank | `.venv/bin/python -m trustworthy_pain_fusion eda-search features` | Section 4.8 |
| EDA search | `.venv/bin/python -m trustworthy_pain_fusion eda-search evaluate`, then `.venv/bin/python -m trustworthy_pain_fusion eda-search families` | Section 4.8, Fig. 4 |
| search permutation null | `.venv/bin/python -m trustworthy_pain_fusion eda-null observed`, then `.venv/bin/python -m trustworthy_pain_fusion eda-null null` (1000 within-subject draws) | Section 4.8 |
| repeated partitions | `.venv/bin/python -m trustworthy_pain_fusion partitions run1` | Tables 2, 4 and 5, Fig. 2 |
| predictor sensitivity | `.venv/bin/python -m trustworthy_pain_fusion partitions run3` | Table 2(b) |
| silent degradation | `.venv/bin/python -m trustworthy_pain_fusion partitions run4` | Table 7 |
| baseline-only normalization | `.venv/bin/python -m trustworthy_pain_fusion partitions run5` | Table 6(b) |
| frozen post-fusion calibration | `.venv/bin/python -m trustworthy_pain_fusion partitions run6` | Tables 5(b), 6(c) |

## Python API

The main building blocks are available as classes:

| Class | Module |
|---|---|
| `AI4Pain2026Dataset` | `trustworthy_pain_fusion.data.ai4pain2026` |
| `AI4Pain2025Dataset` | `trustworthy_pain_fusion.data.ai4pain2025` |
| `BioVidDataset` | `trustworthy_pain_fusion.data.biovid` |
| `UsableInformationEstimator` | `trustworthy_pain_fusion.methods.decodability` |
| `PredictorIndependentTests` | `trustworthy_pain_fusion.methods.independence` |
| `FusionOperators` | `trustworthy_pain_fusion.methods.fusion` |
| `NormWearEncoder` | `trustworthy_pain_fusion.methods.encoders` |
| `ModalityDropout`, `SignalCorruptor` | `trustworthy_pain_fusion.experiments.partitions` |

Results (summary metrics) of manuscript could be found in [here](https://drive.google.com/file/d/1VQN9GFgerG4lAqOBe593KMSQqXI7_v-V/view?usp=sharing)
