import unittest
import torch
from model import EncoderCell

def tensordiff(t1, t2):
    return (t1 - t2).abs().sum().item()


class TestEncoderCell(unittest.TestCase):

    def setUp(self):
        self.input_size = 2
        self.hidden_size = 4
        self.ec = EncoderCell(self.input_size, self.hidden_size)

    def test__batch_mul(self):
        batch_size = 15
        x = torch.ones(batch_size, self.input_size)
        h = self.ec._batch_mul(self.ec.Wh, x)
        self.assertEqual(h.shape, (batch_size, self.hidden_size))

    def test_updates(self):

        for W in self.ec.embeddings + self.ec.updates:
            W.data.zero_()
        
        batch_size = 42
        hprev = torch.ones(batch_size, self.ec.hidden_size)
        x = torch.zeros(batch_size, self.ec.input_size)
        r = self.ec._update_r(x, hprev)
        self.assertEqual(r.shape, (batch_size, self.hidden_size))
        self.assertAlmostEqual(tensordiff(r, .5 * torch.ones_like(r)), 0)

        z = self.ec._update_z(x, hprev)
        self.assertEqual(z.shape, (batch_size, self.hidden_size))
        self.assertAlmostEqual(tensordiff(z, .5 * torch.ones_like(z)), 0)

        h_proposed = self.ec._update_h(x, r,  hprev)
        self.assertEqual(h_proposed.shape, (batch_size, self.hidden_size))
        self.assertAlmostEqual(tensordiff(h_proposed, torch.zeros_like(h_proposed)), 0)

    def test_forward(self):
        for W in self.ec.embeddings + self.ec.updates:
            W.data.zero_()

        batch_size = 32
        hprev = torch.ones(batch_size, self.ec.hidden_size)
        x = torch.zeros(batch_size, self.ec.input_size)
        h = self.ec(x, hprev)
        self.assertEqual(h.shape, hprev.shape)
        self.assertAlmostEqual(tensordiff(h, .5 * hprev), 0)


if __name__ == "__main__":
    unittest.main()