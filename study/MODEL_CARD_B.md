# Model card for balanced baseline B

## Identity and intended role

B is the balanced conditional GAN baseline in `generative_recovery_1500`.
Three confirmation networks were trained from scratch at seeds 101, 202 and 303,
after a separate seed-909 development trajectory selected generator EMA decay
0.99. B is a family of these exact checkpoints, not a single model obtained by
averaging the three networks. It is a research baseline for the stated RGB64
experiment, not a production portrait generator or an official SNGAN reproduction.

The conditioning index is `2*Male + Blond_Hair` using CelebA's supplied binary
annotations. The four bins are 00, 01, 10, 11. This encoding should not be treated
as a definition of gender, a verified human-attribute classifier, or a complete
description of a person's appearance. Every bin occurs in training; this is not
an unseen-combination generalization test.

## Architecture and learning

The generator has 737,315 trainable parameters: latent dimension 128, actual base
width 32, a 4×4×128 dense projection, four nearest-neighbor residual upsampling
blocks with widths 128→128→64→32→16, eight conditional-BatchNorm affine tables,
and a final RGB convolution with tanh. The discriminator has 888,545 parameters,
five residual blocks, spectral normalization, spatial sum pooling, and a
conditional label-projection score. Source: `src/models.py`, SHA256
`f82bfdd31060fe357a3102fa4debd110da5989f86c5f9e61b6163036f579f5fa`.

B uses discriminator hinge loss and negative mean discriminator score for G.
Adam uses learning rate 0.0002, betas (0,0.9), epsilon 10^-8. A real batch has
64 images, 16 per group sampled with replacement. During each G update, sixteen
latents are shared across all conditions in a single forward pass; D fakes use a
separate independent-latent draw. Adversarial inputs receive random horizontal
flips. There is one D and one G optimizer attempt per iteration, with AMP and
initial scale 1024. Generator parameters are tracked by two EMAs, 0.99 and 0.999;
buffers are copied, not averaged. No external pretrained G or D is used.

Confirmation uses 1,500 attempted iterations and records 1,500 successful G/D
updates, zero AMP skips, 96,000 real presentations and 24,000 presentations per
group in each run. Presentations are not unique images. The same data streams
and seed values are used by the matched C runs. Training kernels were not
claimed bitwise deterministic. No quality metric was optimized during training.

## Data and evaluation

The canonical cache is a centered 178-pixel crop followed by Pillow bicubic RGB64
resizing. No face images are included here. The fresh graph-component partition
joins supplied identity labels, exact/reflected pixel equality and the declared
pHash≤4 plus RGB RMSE≤8 relation. Components touching original held-out data or
containing exact/flip annotation conflicts are excluded as specified in the
archived policy. TRAIN/VAL/TEST contain 112,615/23,156/25,217 images; the rare bin
has 942/253/173 images and 282/67/65 union components. Disjoint components do not
establish universal semantic deduplication or perfect identity ground truth.

Validation selected EMA 0.99 for B from two EMAs of the one seed-909 trajectory.
The candidate had four development trajectories, so total search compute was
not equalized across methods. The test used 512 shared-latent requests per bin
and at most 512 real references, with all 173 rare-test images. No generated
filtering, relabeling or attribute judge was used. KID was the recorded cubic
U-statistic form on Clean-FID clean-mode Inception features, not FID. PRDC used
strict feature-neighborhood membership with k=3.

Rare-group KID×1,000 was 150.910, 188.261 and 160.819 at seeds 101, 202 and 303;
the arithmetic mean was 166.663. Macro KID means were 159.807. Every B
model-by-group PRDC recall was zero. These are limited feature diagnostics, not
evidence of satisfactory coverage, semantic accuracy, fairness or privacy.

## Availability and reproducibility limits

Only B101 was recovered as a complete selected-EMA archive. Its EMA SHA256 is
`88d958f0226226d1bc720c03f0f2444ed94ab7a8451447e692bc6f18db3ebb67`.
B202 and B303 have complete saved evaluation records but missing complete weight
bytes. Four of eight B202 transport fragments were recovered; those fragments
are not a usable model. This source directory contains no model tensors.

All aggregate scalar point summaries, configuration and source hashes are
available here. Original point JSONs with selected real filenames remain excluded
from the public tree; each derivative identifies its exact source hash.
Numerical replay of all B models cannot be completed from the current weights.
The exact selected real-feature bank, full derived split/component arrays and
teacher tensor snapshot were not retrieved. The teacher is not needed to run an
already-trained B generator, but is needed to reproduce C training under the
recorded shared study. Full optimizer checkpoints are absent. Source consistency
checks and synthetic tests do not substitute for an independent metric rerun.

## Appropriate use and restrictions

Use B to inspect this implementation, evaluate the saved negative comparison or
design a separately specified research study. It is unsuitable as evidence that
balanced sampling solves long-tailed conditional generation. Do not infer that
synthetic outputs are guaranteed free of memorized training content. There is no
differential-privacy or demographic-fairness guarantee. Follow CelebA's terms,
dependency licenses and any separately specified trained-weight rights. Public
code availability does not itself grant dataset or commercial portrait rights.
