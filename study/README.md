# Frozen CelebA B and C study

This directory preserves the executed source and selected scientific records of
`generative_recovery_1500`. It reports a negative result: the selected signed
feature-interaction candidate C had higher mean rare-group test KID than the
balanced baseline B. It is a public source-and-record subset, **not the original
complete private research archive and not a self-contained numerical replay**.

## What was actually tested

- RGB64 CelebA; condition `g = 2*Male + Blond_Hair`; all four bins occur in TRAIN.
- Component-separated TRAIN/VAL/TEST sizes: 112,615 / 23,156 / 25,217 images.
- Five development trajectories at seed 909: B and C with
  `lambda in {0.1, 1}` crossed with `kappa in {0, 1}`. Two inference EMAs per
  trajectory produce all ten validation points.
- Validation selected EMA 0.99 for B and `(lambda=1, kappa=0, EMA=0.99)` for C.
  The selected tolerance was therefore inactive.
- Fresh confirmation seeds 101, 202, 303; 1,500 attempted iterations each.
  All six runs recorded 1,500 successful G/D updates and zero AMP skips.
- Mean rare-test KID × 1,000: B 166.663; C 185.544. C−B is +18.880, with
  nominal one-sided 98.75% **conditional** bootstrap upper bound +23.611.
  This comparison did not demonstrate improvement; it does not prove universal
  harm. All 24 model-by-group PRDC recalls were zero.

The cloud directory name `generative_recovery_2500` is a legacy path. The actual
study identifier and all reported development/confirmation budgets are 1,500.
No lost intermediate study is pooled into these results.

## Contents and provenance

`src/` contains the unchanged model, B/C trainer, fixed-teacher preparation,
interaction statistics/witness, KID/PRDC feature adapters and component bootstrap.
`audit/` contains the archived exact/identity recovery and near-component split
implementations. `runtime/` contains historical preparation, orchestration,
archiving and replay sources. Runtime source is included for inspection; old
operational gates are not fresh execution authorization.

`configs/`, the frozen plan/budget/selection files, `development/`,
`confirmation/`, and `teacher/` preserve actual settings and completion records.
The latter directory contains numerical statistics and metadata, not teacher
weights. `validation_evaluation/` and `test_evaluation/` retain safe original
commit/registry and completion metadata. `public_results/` contains aggregate-only
derivatives of all ten validation points, six test points and the bootstrap
report. The original point/CI JSONs contain condition-grouped real filenames and
are deliberately excluded. Each derivative records the original source SHA256,
the retained fields and this transformation; it is not called the original byte
string. `prior_audit/` and `data/` contain aggregate audit and split reports.

Every payload labeled preserved is byte-identical to its selected core source.
`provenance/SOURCE_INVENTORY.json` lists all 288 original payloads, their hashes,
and inclusion/exclusion decisions. `provenance/verified_core_MANIFEST.json` is the
unchanged original manifest, retained as evidence rather than used as the
manifest of this smaller public tree. `PUBLIC_MANIFEST.json` describes this tree.
No file is silently redacted and then represented as the original byte string.
The archived private-delivery notice is retained under `provenance/`; this README
states the current public-subset scope and does not enlarge any license.

Account/browser receipts, screenshots, session/process logs, filesystem logs,
and encoded transport chunks are excluded. Dataset pixels, source JPEG/ZIP/TXT
metadata, supplied identity-label arrays, per-image exposure vectors,
filename-to-condition lists, selected-bank support arrays, full split/component
arrays, synthetic grids, external Inception weights, optimizer state and full GAN
checkpoints are absent. This deliberately conservative public subset contains
code, configurations, aggregate values and provenance rather than any part of
the CelebA image or annotation database.

## Verify and run the small mathematical tests

Run these commands from this `study/` directory, using a Python environment you
control. The verifier uses only the standard library and performs no network,
model, or training operation:

```bash
python tools/verify_public_package.py
```

The statistics fixtures need NumPy. The autograd fixtures additionally need
PyTorch; their own skip status must not be described as a pass when Torch is
absent. These are synthetic implementation checks, not efficacy experiments:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_interaction_statistics.py' -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_interaction_witness_torch.py' -v
PYTHONDONTWRITEBYTECODE=1 python audit/audit_exact_identity_recovery.py --self-test
PYTHONDONTWRITEBYTECODE=1 python audit/build_clean_component_split.py --self-test
```

The recorded producer environment used Python 3.13.16, PyTorch 2.11.0+cu130,
torchvision 0.26.0+cu130, NumPy 2.1.3, Pillow 11.3.0 and clean-fid 0.1.35 on two
Tesla T4 GPUs. These version records are not a portable dependency lock or a
promise of bitwise deterministic kernels. Install an appropriate PyTorch build
for the intended hardware; do not assume any arbitrary version reproduces the
recorded tensors.

## Historical entrypoints and required inputs

The following are exact archived entrypoints, **not instructions to revive the
expired study clock or fabricate old approvals**. They require the original
bound data/teacher/checkpoint state and refuse missing or changed inputs:

```bash
python src/train_v3.py --root /kaggle/working/generative_recovery_2500 --config /kaggle/working/generative_recovery_2500/configs/development/B_seed909.json
python src/prepare_teacher_v3.py --root /kaggle/working/generative_recovery_2500
python src/evaluate_v3.py --root /kaggle/working/generative_recovery_2500 --phase validation --weights-dir /tmp --device cuda:0 --max-seconds 1200
python src/evaluate_v3.py --root /kaggle/working/generative_recovery_2500 --phase test --weights-dir /tmp --device cuda:0 --max-seconds 1200
```

`src/train_v3.py` can execute individual archived B/C configurations, but still
requires the exact canonical cache, clean split, component map and, for C, fixed
teacher artifact/statistics; confirmation also checks the saved selection.
`runtime/launch_v3_study.py --phase development|confirmation --root ...` and
`runtime/supervise_v3_study.py --root ...` implement the old source-approved
sequence and absolute deadline. A new experiment needs a new prospectively
specified protocol, output tree, data bindings and source approval. Changing
these records produces a new study, not an exact continuation of this one.

For licensed aligned CelebA images obtained independently, this is the exact
canonical preprocessing entrypoint; it refuses to overwrite an existing cache:

```bash
python src/cache_images.py --image-dir /path/to/licensed/img_align_celeba --output /path/to/NEW/images64.npy --workers 32
```

With matching image bytes and Pillow processing, the recorded complete NPY SHA256
is `98d39e0febb8c1946aba70eca2ca5cb96a657cdf6075f7ae32fc56d4c6f9bd5c`.
Preprocessing does not recreate the missing clean-index/component arrays. The
historical audit/recovery workers also require preserved earlier inputs and
failure/source records; their existence is not a one-command recipe for a new
user to reconstruct all past audits.

## Exact EMA replay and its present limit

The separate portable adapter reads selected EMA artifacts instead of faking
full checkpoints. If **all six selected models and the original hash-bound core,
reference ZIP and export record** are obtained, its exact modes are:

```bash
python runtime/replay_main_ema_v3.py --models-root /path/to/ALL_SELECTED_RUN_FOLDERS --core-zip /path/to/research_core.zip --references-zip /path/to/replay_references.zip --export-record /path/to/export_manifest.json --output /path/to/NEW/replay_preflight --preflight-only --max-seconds 1200
python runtime/replay_main_ema_v3.py --models-root /path/to/ALL_SELECTED_RUN_FOLDERS --core-zip /path/to/research_core.zip --references-zip /path/to/replay_references.zip --export-record /path/to/export_manifest.json --output /path/to/NEW/replay_smoke --smoke-only --weights-dir /path/to/inception_weights --device cuda:0 --max-seconds 180
python runtime/replay_main_ema_v3.py --models-root /path/to/ALL_SELECTED_RUN_FOLDERS --core-zip /path/to/research_core.zip --references-zip /path/to/replay_references.zip --export-record /path/to/export_manifest.json --output /path/to/NEW/replay_points --with-ci --weights-dir /path/to/inception_weights --device cuda:0 --max-seconds 1200
```

Preflight performs no Torch computation. Smoke uses one C101/condition-0 batch of
64 and no KID. Full mode regenerates every 512-request group; CI requires the
explicit flag. Fresh outputs are labeled replay records. This public subset is
not the `research_core.zip` accepted by those original byte checks. The adapter
retains its original gates; it is not claimed to run successfully on today's
incomplete delivery.

Complete EMA archives were recovered for B101 and C101/C202/C303, four of six.
B202 and B303 remain missing as complete models. Four of the eight original B202
text fragments were recovered and hash-checked; they do not form a usable fifth
model. The selected real-feature/latent reference archive, fixed-teacher tensor
snapshot and complete derived split-support archive were not retrieved. All ten
validation and six test scalar records and their bootstrap report were recovered;
they were not independently recomputed from the partially delivered weights.
The ten validation points cannot be regenerated from selected MAIN EMAs: that
would additionally require the development weights or feature banks. A reported
one-batch smoke completion does not establish full point/CI replay; its actual
machine-readable report was not retrieved.

## Data, licenses and responsible interpretation

Obtain CelebA from a permitted source and follow the dataset owner's current terms:
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html . This package redistributes no
CelebA face pixels or supplied identity database. Dataset access, a public mirror,
source availability and a code license are distinct matters; this release grants
no additional dataset, identity, publicity, commercial-use or trained-weight
rights. External dependencies and Inception weights retain their own terms.

Read `MODEL_CARD_B.md` and `MODEL_CARD_C.md`. Supplied annotation bins and feature
metrics do not establish semantic correctness, demographic fairness, privacy,
absence of memorization, state-of-the-art quality or a new general theorem.
The source/records support auditing this particular result and designing a fresh
study; they do not support claiming a complete portable reproduction was achieved.
