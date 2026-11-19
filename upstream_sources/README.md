# Primary source implementation library

This collection contains eight separately licensed, commit-pinned source
snapshots and two link-only records. It complements the new pedagogical reference
implementations; it does not claim that upstream code was executed or that any
paper was reproduced. A model name in the literature is not a claim that its
paper's original training setup has been run here.

The retained source, configuration, documentation and license files keep their
original Git-blob and SHA256 identities. Images, datasets, weights, fonts, media,
binaries, dependency trees and notebooks with outputs/attachments are omitted.
Every inclusion and exclusion appears in a `SNAPSHOT_MANIFEST.json`. This is a
filtered source snapshot, **not a full asset snapshot**. Upstream documentation
can still reference intentionally absent files or external dependencies.

## License boundaries

Each snapshot retains its own upstream license and copyright notices. No license
of the enclosing project overrides these terms:

- NCSN is **GPL-3.0**; consult its complete `source/LICENSE` and source notices.
- Facebook `flow_matching` is **CC BY-NC 4.0**, including its attribution and
  noncommercial restrictions; it is **not MIT**.
- GAN and the official PyTorch VAE example retain BSD 3-Clause licenses.
- Score-SDE and historical TensorFlow RealNVP retain Apache 2.0 licenses.
- OpenAI consistency models and CompVis latent diffusion retain MIT licenses.
- DDPM and MADE are link-only because no redistribution license file was located
  in their pinned repositories. No upstream code or documentation is copied for
  these two entries. Public visibility is not a redistribution license.

No dataset, model-weight, identity, commercial-use or publicity rights are added.
The VAE subtree is an **official PyTorch reference example**, not the VAE paper
authors' original implementation. Historical RealNVP includes only the specified
TensorFlow `research/real_nvp` subtree and the root license from the same commit.

## Verify without running upstream code

```bash
python verify_snapshots.py
```

This verifier uses only the standard library and checks membership, sizes,
SHA256 and Git blob SHA1 for the retained files. It performs no imports from
upstream packages, installation, training, sampling, evaluation or network work.
Do not interpret a source-integrity pass as an upstream test-suite pass. Any
future installation or execution needs its own environment, dependency, data and
resource decisions. The separate reference-model tests should be invoked by
their explicit paths rather than by indiscriminate discovery through this tree.

See `COVERAGE.json` for the repositories, complete commit pins, primary paper URLs,
license hashes and file counts. Only this filtered `upstream_sources/` tree is
intended for inclusion in the release. The separate `upstream_archives/` working
directory contains retrieval inputs and is excluded from publication.
