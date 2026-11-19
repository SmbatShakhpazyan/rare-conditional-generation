# Runnable references for generative model families

This is an **original pedagogical PyTorch library**, with ten runnable entries:
VAE, conditional GAN, Gaussian autoregression, affine-coupling flow, DDPM,
noise-conditional score matching, a reverse VE SDE / probability-flow ODE,
flow matching, standalone consistency training, and a latent-diffusion hybrid.
The score and SDE formulations are exposed separately. All are small vector
networks; the CLI flattens local image tensors and restores their shape when
saving samples.

These sources are **not the original authors' implementations, official
replications, or trained comparison arms of the accompanying paper**. No
pretrained network or external dataset is included or downloaded. The actual
research study used its separately archived residual conditional GAN, not the
MLP GAN implemented here. Tiny synthetic CPU runs below check software and
mathematical behavior; they do not measure model efficacy.

## Install and a complete local example

From this `reference_models` directory, use a private environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m genref.make_demo --output work/demo.pt --count 128 --dim 16 --classes 4
.venv/bin/python -m genref.train --model ddpm --data work/demo.pt --output work/ddpm_run --steps 20 --batch-size 32 --balanced --device cpu --max-seconds 120
.venv/bin/python -m genref.sample --checkpoint work/ddpm_run/checkpoint.pt --output work/ddpm_samples.pt --count 16 --device cpu
```

Use new output names. Training and sampling refuse to overwrite their outputs.
The demo is a **synthetic class-dependent sinusoid tensor**, not CelebA, a
substitute real dataset, or an image-quality benchmark. The training CLI defaults
to CPU and one Torch thread. GPU devices are optional user choices; no GPU is
needed for the tests. Windows users can invoke the corresponding Python in
`.venv/Scripts`.

`requirements.txt` gives the supported API floor. `requirements-tested.txt`
records the exact local validation versions; it is not a guarantee for every
OS, hardware backend or older supported release. Python 3.10 or newer is needed.
Installing into the surrounding document-generation runtime is unnecessary.

## Train on your own local tensors

Save a regular `.pt` file containing either a floating/uint8 tensor or a mapping
with `x` and optional `labels`. Loading uses `torch.load(..., weights_only=True)`.

```python
import torch
# x: [N,D], [N,H,W], or [N,C,H,W], N >= 2.
# Floating x must already be finite and in [-1,1]. uint8 is mapped to [-1,1].
torch.save({"x": x, "labels": labels}, "train_tensor.pt")
# labels, if present: torch.int64 [N], covering every value 0,...,K-1.
```

The loader caps the source file at 512 MiB and the loaded tensor at 100 million
elements. It treats the observations as continuous. A Gaussian likelihood on
scaled pixels is a density, not a probability mass for quantized images; no
dequantization is silently added. Supply **TRAIN data only**. This convenience
loader does not infer train/validation/test partitions, identity components,
near duplicates, permissions, dataset licenses or semantic labels. Those audits
remain the researcher's responsibility.

The class count is inferred from labels. `--unconditional` explicitly ignores
them. Natural sampling is uniform over observations with replacement;
`--balanced` instead samples the same number from each class, requiring a batch
size divisible by the class count. Neither repeated sampling nor balance creates
independent rare examples. Each algorithm uses the same declared tensor input;
there is no automatic validation sweep or selection.

## Methods and explicit sampling choices

`--model` chooses one entry. Every class has substantive objective and sampling
code; no method is a stub or an alias for an unrelated objective.

| Entry | Objective / implementation | Default generation | Important departure or limit |
|---|---|---|---|
| `vae` | Gaussian reconstruction NLL plus analytic Gaussian KL | Prior latent, Gaussian decoder draw | Small vector MLP; no perceptual encoder |
| `cgan` | Projection discriminator, hinge D loss, negative-score G loss | Conditional tanh generator | Original MLP adaptation, no study witness or EMA |
| `autoregressive` | Strictly masked normalized Gaussian conditional density | Sequential coordinate draws | Gaussian masked MLP, not binary NADE |
| `realnvp` | Exact affine-coupling change-of-variables log density | Invert all coupling layers | No multiscale image flow or dequantization |
| `ddpm` | Uniform-time epsilon MSE | Every ancestral reverse step | Short pedagogical schedule; no DDIM/step skipping |
| `score` | Discrete sigma-weighted denoising score loss | Annealed Langevin | Finite unadjusted corrector steps |
| `sde` | Continuous VE sigma-weighted denoising score loss | Reverse Euler–Maruyama | VE only; optional probability-flow Euler |
| `flow_matching` | Straight-path independent-endpoint velocity MSE | Heun ODE integration | Not optimal-transport endpoint coupling |
| `consistency` | Shared-noise adjacent-level L2 and detached EMA target | EMA one-step map | Standalone fixed-grid reference, no score distillation |
| `latent_diffusion` | VAE warmup, frozen compression/calibration, latent DDPM | Latent ancestral chain, decoder mean | No perceptual/adversarial autoencoder or cross-attention |

`MODEL_MANIFEST.json` maps each entry to primary-paper URLs, exact implemented
loss, sampling mechanism, original-code provenance, deviations and limits.
The degree-mask architecture is specifically related to Germain, Gregor, Murray and Larochelle, [MADE (ICML2015)](https://proceedings.mlr.press/v37/germain15.html), as well as the earlier NADE density-estimation mechanism. Our code is a Gaussian-output fixed-mask adaptation, not either paper's implementation.
Degree-zero hidden units receive conditions/biases but no observed coordinate,
so even the first conditional can depend on the requested class.
The monograph supplies fuller equations and related-work context. No universal
quality, fairness, privacy, support-coverage or convergence claim is made.

For iterative samplers, optional arguments have **method-specific meanings**:

```sh
# Two Langevin corrections PER noise level, not two total noise levels.
.venv/bin/python -m genref.sample --checkpoint work/score_run/checkpoint.pt --output work/score_draws.pt --sample-steps 2 --sampler langevin
# Continuous VE probability-flow Euler with 64 total integration steps.
.venv/bin/python -m genref.sample --checkpoint work/sde_run/checkpoint.pt --output work/sde_draws.pt --sample-steps 64 --sampler ode
# Heun or Euler transport over [0,1].
.venv/bin/python -m genref.sample --checkpoint work/fm_run/checkpoint.pt --output work/fm_draws.pt --sample-steps 32 --sampler heun
# One EMA consistency map, or additional declared noise reinjection/refinement.
.venv/bin/python -m genref.sample --checkpoint work/cm_run/checkpoint.pt --output work/cm_draws.pt --sample-steps 1 --sampler consistency
```

First create those example checkpoints using `genref.train` with `--model
score`, `sde`, `flow_matching` or `consistency` and distinct run directories.
VAE, GAN, autoregression and RealNVP reject iterative sampler arguments. DDPM
and latent DDPM accept only their configured number of ancestral steps; the CLI
does not silently substitute an invalid shortened reverse chain. Requested
conditional labels default to round-robin; `--class-label 2` requests one class.

Samples are not clipped and are saved as tensors, not proof of realistic images.
Continuous samplers can produce values outside the input range. The cGAN's
declared tanh is an exception built into its generator. If you clip or quantize
for a later image display or metric, treat that as an additional stated operation.
Noise-score and VE samplers stop at a positive minimum noise level and use a
finite high-noise Gaussian approximation. They do not exactly sample an
unsmoothed target distribution, even if the displayed score equations are exact.

## The latent-diffusion stage is explicit

```sh
.venv/bin/python -m genref.train --model latent_diffusion --data work/demo.pt --output work/latent_run --ae-steps 25 --steps 100 --batch-size 32 --balanced
```

`--ae-steps` are **additional** Gaussian-VAE updates. The compressor is then
frozen. A full pass over this same TRAIN tensor computes the posterior-mean
latent center and standard deviation, with a variance floor. Diffusion is fitted
to those standardized posterior means. Sampling reverses that standardization
and returns the frozen decoder mean. This teaches the two-stage mechanism; it
is not CompVis latent diffusion or an equivalent perceptual autoencoder.
Calling latent diffusion before calibration raises an error.

## Presets, optimization and records

`tiny` uses width 64, latent dimension 8, four couplings, 32 DDPM steps, eight
noise levels, and sigma range 0.01–2. `small` uses width 128, latent dimension 16,
six couplings, 128 DDPM steps and sixteen noise levels. The linear DDPM beta
endpoint is 0.2 for `tiny` and 0.05 for `small`. These unusually short schedules
are chosen for tractable reference computation, not recommended image-quality
hyperparameters. `Config` exposes all constants for explicit Python use.

The CLI uses Adam (GAN betas 0/0.9; otherwise PyTorch Adam defaults), a declared
learning rate, finite-gradient checks and gradient clipping at norm 10. GAN
updates freeze discriminator parameters and spectral-normalization state during
the generator step. Consistency training updates its frozen EMA target only
after an optimizer step. Latent diffusion does not update the compressor during
its denoising stage. There is no AMP, implicit checkpoint selection or resume.

Each successful run saves `checkpoint.pt`, `run.json` and `losses.jsonl`.
Records contain the model/configuration, input SHA-256, original observation
shape, per-source hashes, update counts, parameters, software version and elapsed
time. Sampling verifies that all current `genref` source bytes match the
checkpoint's source hashes before model construction/loading; an edited source
tree requires a newly trained reference checkpoint. The checkpoint contains model state and provenance; **optimizer state and
RNG state are not a restart guarantee**. Failed/interrupted runs preserve their
partial records and do not receive a success status. `--max-seconds` is a soft
cap checked between operations; it cannot interrupt an arbitrary long Torch
kernel. Use your process manager for an additional external deadline if needed.

## Meaningful CPU validation

```sh
.venv/bin/python -B run_cpu_validation.py --output work/cpu_validation_new
# Or standard unittest, without a new JSON record:
.venv/bin/python -B -m unittest tests.test_reference_models -v
```

The suite exercises actual CPU losses, finite nonzero gradients, conditional and
unconditional sampling, strict state round trips, every CLI train/sample path
with **two tiny updates**, masked autoregressive causality, nonidentity coupling
inversion and autodifferentiated Jacobian determinants, DDPM posterior variance,
VE reverse-time sign against a known Gaussian score, constant-velocity ODE
integration, consistency boundary/EMA behavior and compressor immutability.

Saved `validation/attempt*/report.json` and `unittest.log` are actual local test
records. Retained earlier attempts remain historical if a test source is later
clarified. They establish these bounded software checks on the recorded runtime;
they do not reproduce paper quality results, compare model families, measure
large-scale throughput, or validate arbitrary hyperparameters. Two synthetic
updates are not evidence that any trained model is useful.

## Attribution and code rights

The implementation was written afresh from standard equations and the cited
primary literature. No source file from an author repository was copied.
Original author implementations and their licenses remain separate. The parent
repository license governs this original reference code. The primary papers,
their figures, datasets and third-party weights retain their own rights. The
library redistributes none of those assets.

