"""CPU-only autograd fixtures. Not executed where Torch is unavailable."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.interaction_witness import Witness


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Torch unavailable')
class TorchWitnessFixtures(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.stats = {'target':[0.,0.], 'scale':[2.,1.], 'uncertainty':[.5,1.]}
        self.groups = torch.arange(4, dtype=torch.long)

    def test_literal_loss_and_gradient_use_previous_witness(self):
        t = self.torch
        w = Witness(self.stats, kappa=1, beta=.9, warmup_updates=0)
        w.update(t.tensor([6.,-3.]))
        x = t.tensor([[4.,2.],[0.,0.],[0.,0.],[0.,0.]], requires_grad=True)
        before = {k:v.clone() for k,v in w.state_dict().items()}
        rng = t.get_rng_state().clone()
        loss, diagnostics, contrast = w.loss(x,self.groups,strength=.4)
        # q=(2.75,-2), delta=(1,.5): .4*(2.75/2-1)=.15.
        t.testing.assert_close(loss, t.tensor(.15), rtol=1e-6, atol=1e-7)
        loss.backward()
        expected = t.tensor([[.1375,-.2],[-.1375,.2],[-.1375,.2],[.1375,-.2]])
        t.testing.assert_close(x.grad,expected,rtol=1e-6,atol=1e-7)
        self.assertFalse(contrast.requires_grad)
        self.assertEqual(len(list(w.parameters())),0)
        for key,value in before.items():
            self.assertTrue(t.equal(w.state_dict()[key],value))
        self.assertTrue(t.equal(rng,t.get_rng_state()))
        self.assertEqual(float(diagnostics['witness_active_fraction']),1.)

    def test_zero_strength_zero_gradient_and_no_state_mutation(self):
        t = self.torch
        w = Witness(self.stats,warmup_updates=0)
        w.update(t.tensor([6.,-3.]))
        x = t.ones(4,2,requires_grad=True)
        loss,_,_ = w.loss(x,self.groups,strength=0)
        loss.backward()
        self.assertTrue(t.equal(x.grad,t.zeros_like(x)))
        self.assertEqual(float(loss),0.)

    def test_warmup_ema_checkpoint_and_reset(self):
        t = self.torch
        w = Witness(self.stats,kappa=0,beta=.9,warmup_updates=2)
        x=t.tensor([[4.,2.],[0.,0.],[0.,0.],[0.,0.]],requires_grad=True)
        first,_,contrast=w.loss(x,self.groups)
        self.assertEqual(float(first),0.)
        w.update(contrast)
        second,_,_=w.loss(x,self.groups)
        self.assertEqual(float(second),0.)
        w.update(t.tensor([3.,1.5]))
        t.testing.assert_close(w.fake_ema,t.tensor([1.2,.6]))
        active,_,_=w.loss(x,self.groups)
        self.assertGreater(float(active),0.)
        copy=Witness(self.stats,kappa=0,beta=.9,warmup_updates=2)
        copy.load_state_dict(w.state_dict())
        t.testing.assert_close(copy.loss(x,self.groups)[0],active)
        copy.reset_statistics(self.stats)
        self.assertEqual(int(copy.updates),0)
        self.assertEqual(float(copy.loss(x,self.groups)[0]),0.)

    def test_tolerance_zeroes_small_old_discrepancy_and_rejects_attached_update(self):
        t=self.torch
        w=Witness(self.stats,kappa=1,warmup_updates=0)
        w.update(t.tensor([.4,.8]))
        x=t.ones(4,2,requires_grad=True)
        loss,diag,_=w.loss(x,self.groups)
        loss.backward()
        self.assertEqual(float(diag['witness_active_fraction']),0.)
        self.assertTrue(t.equal(x.grad,t.zeros_like(x)))
        with self.assertRaises(ValueError):
            w.update(t.zeros(2,requires_grad=True))


if __name__ == '__main__':
    unittest.main()
