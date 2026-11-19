# Four recovered study generators

Complete selected EMA tensor archives: **B101, C101, C202, C303**. The actual
confirmation study has six models; B202/B303 are missing as complete files.
These bytes are the original recovered artifacts, not newly trained substitutes.
Each payload contains `ema_generator`, `generator_config` and producer metadata.
`models.py` is the exact archived architecture. The actual base width is 32.

From the repository root, after installing Torch:

```sh
python tools/verify_recovered_generators.py --weights-dir model_weights --output work/NEW_forward_report.json
```

This restricted-load CPU check generates eight outputs per archive without
optimization. It asserts strict state keys, parameter count 737,315, finite
shape/range, same-runtime repeat equality and unchanged inference state. The
actual completed check is in `verification/recovered_generator_cpu_forward.json`.
It does not reproduce KID, PRDC, reference features or the bootstrap. Output
hashes are runtime-specific, not promised across devices or library versions.

For sampling in Python:

```python
import sys, torch
sys.path.insert(0, "model_weights")
from models import Generator
p = torch.load("model_weights/B_seed101_EMA.pt", map_location="cpu", weights_only=True)
g = Generator(**{k:p["generator_config"][k] for k in ("z_dim","num_classes","base_channels")})
g.load_state_dict(p["ema_generator"], strict=True)
g.eval()
with torch.no_grad():
    images = g(torch.randn(4,128), torch.arange(4, dtype=torch.long))
# images: [4,3,64,64], tanh range; literal annotation bins 0,1,2,3.
```

Weights learned from licensed CelebA data do not grant dataset, identity,
publicity, privacy, commercial-use or unrestricted model-use rights. Follow
[the dataset agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
This public availability does not add a blanket license. No dataset images,
annotations, supplied identities, teacher tensors or external feature weights
are included. The labels are supplied binary annotations, not a comprehensive
gender taxonomy or certified identity. No fairness/privacy guarantee is claimed.
