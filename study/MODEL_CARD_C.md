# Model card for signed-interaction candidate C

## Identity and intended role

C is the validation-selected signed teacher-feature interaction candidate in
`generative_recovery_1500`. It retains B's generator and adversarial
discriminator architecture and adds a training loss. It is not a new generator
architecture, a general solution to rare-group generation, or a first use of
feature matching or detached moving witnesses. It was tested as a research
candidate and produced a worse three-seed mean than B under this short recipe.

Four development configurations at seed 909 crossed λ∈{0.1,1} and κ∈{0,1}; each
was evaluated at generator EMA 0.99 and 0.999. Rare-validation KID selected
λ=1, κ=0, EMA=0.99. The selected κ=0 makes the proposed variability deadzone
inactive. Results of this selected model cannot establish a benefit of positive
uncertainty thresholding. Three fresh confirmation runs used seeds 101/202/303.

## Common networks and additional training cost

G has 737,315 trainable parameters and D has 888,545, exactly as in B. C adds no
trainable G parameter. Its fixed auxiliary teacher is a separate 888,545-parameter
discriminator whose 128-dimensional pre-projection feature vector is used by
the auxiliary term. The teacher is the final raw D of the fresh B seed-909
development run at 1,500 attempts; it is not chosen by validation and is not an
externally pretrained identity or semantic model. Its parameters and
spectral-normalization state remain frozen in evaluation mode.

A full TRAIN-only pass computes image-weighted feature means, signed contrast r,
within-group scale s with floor 0.001, and diagonal component-cluster variability
u. The signed per-component contributions are summed across groups before
squaring, retaining cross-group covariance. The teacher was learned on the same
training set; u is a variability heuristic, not a calibrated confidence radius.
Repeated training presentations do not create new independent components.

## Exact auxiliary mechanism

The alternating weights are c=(+1,−1,−1,+1)/4. Let δ̂ be the current fake
feature contrast and m the previous fake-contrast EMA. Define d=(m−r)/s and
q=sign(d)·relu(|d|−κu/s). The term optimized with G is the centered linear
surrogate `(2λ/128) sum_j stopgrad(q_j)*(δ̂_j-r_j)/s_j`, added to the ordinary
negative-critic generator objective. Teacher input gradients reach G; teacher
parameters receive no gradients. One existing joint G batch supplies the term,
without an additional G/BatchNorm-buffer forward pass.

The tracker beta is 0.9. Its first update copies the current detached contrast;
later successful updates form the EMA. The first 500 successful G updates
accumulate the tracker with zero auxiliary loss, and the next loss may use the
previous tracker. Updates occur after a successful G step, using the contrast
computed before that optimizer step. AMP-skipped G steps advance neither tracker
nor inference EMA. With training-mode BatchNorm, the functional depends on the
joint batch rather than only on separate evaluation-mode image marginals.

This is a tracking surrogate, not a nonnegative distance or an unbiased exact
gradient of the current population squared discrepancy. A stale witness can
point against the current discrepancy gradient. Matching one signed contrast
cannot identify four conditional distributions: common errors can cancel.
Output-based dependence does not imply parameterization-invariant SGD or EMA.
The code and model card claim neither convergence nor non-collapse.

## Training data and common controls

All groups use CelebA's supplied annotation index `2*Male+Blond_Hair`; these are
benchmark labels rather than verified human identities. The audited fresh
component split has 112,615/23,156/25,217 TRAIN/VAL/TEST images, with 942/253/173
rare-bin images. The rare test contains 65 union components, not 65 independently
verified people. The declared graph combines supplied identity, exact/reflection
duplicates and pHash≤4 plus RGB RMSE≤8; no universal semantic deduplication
claim follows. No validation/test image contributes teacher target statistics.

The optimizer, augmentation, batch, latent coupling and training horizon match
B: Adam 0.0002, betas(0,0.9), batch64 with16/group, shared16z quartets for G,
random adversarial flips and 1,500 attempts. Teacher features use unaugmented
generated images. Each confirmation record reports 1,500 successful G/D updates,
zero AMP skips and 96,000 real presentations. Matched iterations do not equalize
total compute: C has additional teacher operations and a larger development grid.

## Results and uncertainty

Selected rare-validation KID×1,000 was 143.568, compared with B's 140.856.
Rare-test values were 217.707, 166.280 and 172.645 for C101/C202/C303, with mean
185.544 and macro mean177.618. C−B rare differences were +66.797, −21.982 and
+11.826; only seed202 improved. Mean difference was +18.880 and the nominal
one-sided98.75% conditional upper bound was +23.611. The comparison did not
demonstrate improvement and does not prove that C is harmful in every setting.

The bootstrap uses 2,000 shared-component/shared-latent replicates over three
fixed model pairs. It does not resample training seeds, the teacher or validation
selection, and is not a training-seed population interval. KID uses recorded
Clean-FID features; all512 generated requests per group enter analysis. All12 C
model-by-group PRDC recalls were zero. These diagnostics do not establish
semantic adherence, privacy, fairness or adequate generative support.

## Availability and responsible interpretation

All three selected C EMA archives were recovered, with SHA256 values:

- C101: `bb8bce2417a99697576da0e6881bc43f4e838dab5b748f2b5565a5400c3b4518`
- C202: `f6d535a78523ed2386d78b5849b49ecb4b914648f3c85e8ade74d0149ccc2b35`
- C303: `b5519d6b067543d70dad36850b9e47b0782cee4794fd552dec5bbe8e38aa2399`

No tensors are embedded in this source directory. Complete six-model replay is
still blocked by two missing B models and missing selected reference banks.
Exact C retraining additionally lacks the original fixed-teacher snapshot,
full clean-index/component-support arrays and complete optimizer checkpoints.
The saved aggregate statistics JSON contains numerical r/s/u but cannot replace the missing
teacher function. The complete ten-point validation search cannot be recreated
from only selected MAIN EMAs. Saved record arithmetic was checked; full metrics
were not independently regenerated from the recovered partial delivery.

Use this model for bounded research inspection, with CelebA and dependency terms
respected. Do not deploy it for face identification, sensitive demographic
inference or fairness certification. No absence-of-memorization or differential
privacy guarantee is provided. The negative result and incomplete reproduction
inputs should remain visible whenever the candidate is described or reused.
