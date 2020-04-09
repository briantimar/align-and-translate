import unittest
import torch
from model import EncoderCell, batch_mul
from model import DecoderCell
from torch.autograd import gradcheck

def tensordiff(t1, t2):
    return (t1 - t2).abs().sum().item()


class TestDecoderCell(unittest.TestCase):

    def setUp(self):
        self.input_size = 2
        self.hidden_size = 4
        self.attention_size = 5
        self.dc = DecoderCell(self.input_size, self.hidden_size, self.attention_size)
    
    def test__attention_energies(self):
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        encoder_hiddens = torch.randn(batch_size, 2 * self.hidden_size, input_length)
        e = self.dc._attention_energies(s, encoder_hiddens)
        self.assertEqual(e.shape, (batch_size, input_length))
    
    def test__attention_weights(self):
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        encoder_hiddens = torch.randn(batch_size, 2 * self.hidden_size, input_length)
        e = self.dc._attention_weights(s, encoder_hiddens)
        self.assertEqual(e.shape, (batch_size, input_length))
        self.assertAlmostEqual(tensordiff(e.sum(dim=1), torch.ones(batch_size)), 0)

class TestEncoderCell(unittest.TestCase):

    def setUp(self):
        self.input_size = 2
        self.hidden_size = 4
        self.ec = EncoderCell(self.input_size, self.hidden_size)

    def test_batch_mul(self):
        batch_size = 15
        x = torch.ones(batch_size, self.input_size)
        h = batch_mul(self.ec.Wh, x)
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

    def test_backward(self):
        """Partial gradient checks"""
        input_size = 1
        hidden_size = 2
        ec = EncoderCell(input_size, hidden_size, dtype=torch.double)

        batch_size = 1
        x = torch.ones(batch_size, input_size).to(dtype=ec.dtype)
        hprev = torch.ones(batch_size, hidden_size, requires_grad=True).to(dtype=ec.dtype)

        def grad_fn(h):
            return ec(x, h).sum()
        self.assertTrue(gradcheck(grad_fn, hprev))

if __name__ == "__main__":
    unittest.main()