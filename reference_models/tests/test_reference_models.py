"""CPU tests of actual Torch implementations and literal mathematical properties."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from genref.registry import Config, MODELS, create
from genref.data import load_training_tensor

RESULTS = {}
torch.set_num_threads(1)


def configuration():
    return Config(dim=6, classes=3, width=16, latent_dim=3, coupling_layers=4,
                  diffusion_steps=8, noise_levels=4)


def observations():
    torch.manual_seed(17)
    return .5 * torch.randn(9, 6).tanh(), torch.arange(9) % 3


class ReferenceTests(unittest.TestCase):
    def test_all_ten_losses_backward_sampling_and_checkpoint(self):
        x, labels = observations()
        for name in MODELS:
            with self.subTest(model=name):
                torch.manual_seed(23)
                model = create(name, configuration())
                if name == "latent_diffusion":
                    opt = torch.optim.Adam(model.autoencoder.parameters(), lr=1e-3)
                    ae_loss, _ = model.autoencoder.loss(x, labels)
                    ae_loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                    model.freeze_and_calibrate(x, labels)
                if name == "cgan":
                    dloss, _ = model.discriminator_loss(x, labels)
                    dloss.backward()
                    self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                                        for p in model.discriminator.parameters() if p.grad is not None))
                    model.zero_grad(set_to_none=True)
                    model.discriminator.eval().requires_grad_(False)
                    loss, _ = model.generator_loss(len(x), labels)
                else:
                    loss, _ = model.loss(x, labels)
                self.assertEqual(loss.ndim, 0)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
                norm = sum(float(g.square().sum()) for g in grads) ** .5
                self.assertGreater(norm, 0)
                model.eval()
                options = {"score": (2, "langevin"), "sde": (8, "ode"),
                           "flow_matching": (4, "heun"), "consistency": (2, "consistency")}
                steps, sampler = options.get(name, (None, None))
                torch.manual_seed(91)
                generated = model.sample(3, labels[:3], steps, sampler)
                self.assertEqual(generated.shape, (3, 6))
                self.assertTrue(bool(torch.isfinite(generated).all()))
                clone = create(name, configuration())
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "state.pt"
                    torch.save(model.state_dict(), path)
                    clone.load_state_dict(torch.load(path, weights_only=True), strict=True)
                clone.eval()
                torch.manual_seed(91)
                repeated = clone.sample(3, labels[:3], steps, sampler)
                self.assertTrue(torch.equal(generated, repeated))
                RESULTS[name] = {"finite_loss": float(loss.detach()), "nonzero_gradient_norm": norm,
                                 "sample_shape": list(generated.shape), "finite_samples": True,
                                 "state_roundtrip_sample_exact_same_runtime": True,
                                 "parameters": sum(p.numel() for p in model.parameters())}

    def test_unconditional_all_ten(self):
        x, _ = observations()
        for name in MODELS:
            with self.subTest(model=name):
                cfg = Config(dim=6, width=8, latent_dim=2, diffusion_steps=3, noise_levels=2)
                model = create(name, cfg)
                if name == "latent_diffusion": model.freeze_and_calibrate(x)
                value, _ = model.discriminator_loss(x) if name == "cgan" else model.loss(x)
                self.assertTrue(bool(torch.isfinite(value)))
                model.eval()
                kwargs = {"score": {"steps": 1}, "sde": {"steps": 2},
                          "flow_matching": {"steps": 2}}
                self.assertEqual(model.sample(2, **kwargs.get(name, {})).shape, (2, 6))

    def test_autoregressive_no_self_or_future_information(self):
        model = create("autoregressive", configuration()).double()
        x, y = observations(); x = x.double().requires_grad_()
        means, logstd = model.parameters_of_conditionals(x, y)
        for coordinate in range(6):
            derivative = torch.autograd.grad(means[:, coordinate].sum(), x, retain_graph=True)[0]
            self.assertTrue(torch.equal(derivative[:, coordinate:], torch.zeros_like(derivative[:, coordinate:])))
            derivative = torch.autograd.grad(logstd[:, coordinate].sum(), x, retain_graph=True)[0]
            self.assertTrue(torch.equal(derivative[:, coordinate:], torch.zeros_like(derivative[:, coordinate:])))
        changed = x.detach().clone(); changed[:, 3:] += 100
        second, _ = model.parameters_of_conditionals(changed, y)
        self.assertTrue(torch.equal(means[:, :4], second[:, :4]))
        identical = torch.zeros(2, 6, dtype=torch.float64)
        first_by_class, _ = model.parameters_of_conditionals(identical, torch.tensor([0, 1]))
        self.assertNotEqual(float(first_by_class[0, 0].detach()),
                            float(first_by_class[1, 0].detach()))
        scalar = create("autoregressive", Config(dim=1, width=8, classes=2))
        scalar_x = torch.zeros(2, 1, requires_grad=True)
        scalar_mean, _ = scalar.parameters_of_conditionals(scalar_x, torch.tensor([0, 1]))
        self.assertTrue(torch.equal(torch.autograd.grad(scalar_mean.sum(), scalar_x)[0],
                                    torch.zeros_like(scalar_x)))
        self.assertNotEqual(float(scalar_mean[0, 0].detach()), float(scalar_mean[1, 0].detach()))
        manual = (-.5 * ((x - means) * (-logstd).exp()).square()
                  - logstd - .5 * math.log(2 * math.pi)).sum(-1)
        self.assertTrue(torch.allclose(manual, model.log_prob(x, y)))

    def test_realnvp_nonidentity_inverse_and_jacobian(self):
        model = create("realnvp", Config(dim=4, width=12, classes=2)).double()
        torch.manual_seed(31)
        with torch.no_grad():
            for layer in model.layers:
                layer.net.net[-1].weight.normal_(0, .06)
                layer.net.net[-1].bias.normal_(0, .03)
        x = torch.randn(3, 4, dtype=torch.float64)
        y = torch.tensor([0, 1, 0])
        z, forward = model.transform(x, y)
        recovered, inverse = model.transform(z, y, inverse=True)
        self.assertLess(float((x - recovered).abs().max().detach()), 1e-10)
        self.assertLess(float((forward + inverse).abs().max().detach()), 1e-10)
        single = x[0].requires_grad_()
        jacobian = torch.autograd.functional.jacobian(
            lambda a: model.transform(a[None], y[:1])[0][0], single)
        expected = torch.linalg.slogdet(jacobian).logabsdet
        self.assertTrue(torch.allclose(expected, forward[0], atol=1e-10, rtol=1e-10))
        RESULTS["realnvp"]["nonidentity_inverse_max_error"] = float((x - recovered).abs().max().detach())
        RESULTS["realnvp"]["jacobian_logdet_max_error"] = float((expected - forward[0]).abs().detach())

    def test_ddpm_posterior_and_no_implicit_step_skipping(self):
        model = create("ddpm", configuration())
        previous = torch.cat([torch.ones(1), model.abar[:-1]])
        expected = model.beta * (1 - previous) / (1 - model.abar)
        self.assertTrue(torch.equal(expected, model.posterior_variance))
        self.assertEqual(float(model.posterior_variance[0]), 0.)
        self.assertTrue(bool((model.posterior_variance[1:] > 0).all()))
        with self.assertRaises(ValueError): model.sample(3, torch.arange(3), steps=2)

    def test_ve_probability_flow_time_direction(self):
        model = create("sde", Config(dim=2, width=8))
        class ExactSmoothedGaussian(nn.Module):
            def __init__(self, outer):
                super().__init__(); self.minimum = outer.sigma_min; self.ratio = outer.log_ratio
                self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
            def forward(self, x, t, labels=None):
                sigma = self.minimum * torch.exp(t * self.ratio)
                return -x * sigma[:, None] / (1 + sigma[:, None].square())
        model.net = ExactSmoothedGaussian(model)
        torch.manual_seed(19); expected = torch.randn(2, 2) * model.sigma_max
        initial = expected.clone()
        times = torch.linspace(1, .001, 64 + 1)
        for t, following in zip(times[:-1], times[1:]):
            sigma2 = model.sigma(t).square()
            expected *= 1 - (t - following) * model.log_ratio * sigma2 / (1 + sigma2)
        torch.manual_seed(19); actual = model.sample(2, steps=64, sampler="ode")
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertLess(float(actual.norm()), float(initial.norm()))

    def test_flow_matching_constant_velocity_integration(self):
        model = create("flow_matching", Config(dim=2, width=8))
        class Constant(nn.Module):
            def __init__(self): super().__init__(); self.value = nn.Parameter(torch.tensor([.25, -.5]))
            def forward(self, x, t, labels=None): return self.value.expand_as(x)
        model.velocity = Constant()
        torch.manual_seed(13); expected = torch.randn(3, 2) + model.velocity.value
        for method in ["euler", "heun"]:
            torch.manual_seed(13); actual = model.sample(3, steps=8, sampler=method)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_consistency_boundary_and_target_update(self):
        model = create("consistency", configuration())
        x, y = observations()
        sigma = x.new_full((len(x),), model.sigma_min)
        self.assertTrue(torch.equal(model.consistency(x, sigma, y), x))
        before = [p.detach().clone() for p in model.target.parameters()]
        with torch.no_grad():
            for p in model.net.parameters(): p.add_(.1)
        model.after_step()
        self.assertTrue(all(not p.requires_grad for p in model.target.parameters()))
        for old, actual in zip(before, model.target.parameters()):
            self.assertTrue(torch.allclose(actual, old + .1 * (1 - model.ema_decay)))
        loss, _ = model.loss(x, y); loss.backward()
        self.assertTrue(all(p.grad is None for p in model.target.parameters()))

    def test_latent_compression_frozen_during_diffusion(self):
        model = create("latent_diffusion", configuration())
        x, y = observations()
        with self.assertRaises(RuntimeError): model.loss(x, y)
        model.freeze_and_calibrate(x, y, batch_size=4)
        before = {k: v.clone() for k, v in model.autoencoder.state_dict().items()}
        optimizer = torch.optim.Adam(model.diffusion.parameters(), lr=.001)
        value, _ = model.loss(x, y); value.backward(); optimizer.step()
        self.assertTrue(all(p.grad is None for p in model.autoencoder.parameters()))
        self.assertTrue(all(torch.equal(v, model.autoencoder.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(bool((model.latent_std > 0).all()))

    def test_tensor_loader_rejects_nonfinite_and_bad_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x.pt"
            torch.save({"x": torch.full((3, 4), float("nan"))}, path)
            with self.assertRaises(ValueError): load_training_tensor(path)
            torch.save({"x": torch.zeros(3, 4), "labels": torch.tensor([0, 2, 2])}, path)
            with self.assertRaises(ValueError): load_training_tensor(path)

    def test_config_rejects_nonfinite_and_unknown_fields(self):
        with self.assertRaises(ValueError): create("score", Config(dim=4, sigma_max=float("nan")))
        with self.assertRaises(TypeError): create("vae", {"dim": 4, "unknown": 1})

    def test_cli_all_ten_two_step_train_and_sample(self):
        from genref.train import main as train
        from genref.sample import main as sample
        x, labels = observations()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp); data = root / "train.pt"
            torch.save({"x": x, "labels": labels}, data)
            for name in MODELS:
                with self.subTest(model=name):
                    output = root / name
                    train(["--model", name, "--data", str(data), "--output", str(output),
                           "--steps", "2", "--ae-steps", "2", "--batch-size", "6",
                           "--balanced", "--max-seconds", "30"])
                    report = json.loads((output / "run.json").read_text())
                    self.assertEqual(report["status"], "REFERENCE_TRAINING_COMPLETE")
                    self.assertEqual(report["steps_completed"], 2)
                    samples = root / f"{name}_samples.pt"
                    args = ["--checkpoint", str(output / "checkpoint.pt"), "--output", str(samples),
                            "--count", "3"]
                    if name in ["score", "sde", "flow_matching", "consistency"]:
                        args += ["--sample-steps", "2"]
                    sample(args)
                    result = torch.load(samples, weights_only=True)
                    self.assertEqual(result["samples"].shape, (3, 6))
                    self.assertTrue(bool(torch.isfinite(result["samples"]).all()))
                    RESULTS.setdefault(name, {})["cli_two_training_updates_and_sample"] = True
                    if name == "latent_diffusion":
                        self.assertEqual(report["autoencoder_steps_completed"], 2)
                    if name == "vae":
                        checkpoint_path = output / "checkpoint.pt"
                        checkpoint = torch.load(checkpoint_path, weights_only=True)
                        checkpoint["source_sha256"]["networks.py"] = "0" * 64
                        torch.save(checkpoint, checkpoint_path)
                        with self.assertRaisesRegex(ValueError, "source differs"):
                            sample(["--checkpoint", str(checkpoint_path), "--output", str(root / "reject.pt")])


if __name__ == "__main__":
    unittest.main(verbosity=2)

