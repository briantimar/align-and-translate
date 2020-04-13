import unittest
import torch
from model import EncoderCell, batch_mul
from model import DecoderCell, DecoderLayer
from model import BiEncoder, flip_padded, pad_tokens
from torch.autograd import gradcheck
from torch.nn.utils.rnn import pad_sequence

def tensordiff(t1, t2):
    return (t1 - t2).abs().sum().item()


class TestDecoderLayer(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 10
        self.hidden_size = 11
        self.embedding_dim = 6
        self.attention_size = 4
        self.output_hidden_size = 7
        self.pad_token = 9
        self.dec = DecoderLayer(self.embedding_dim, self.hidden_size, self.attention_size, 
                                self.output_hidden_size, self.vocab_size, self.pad_token)
    
    def test_loss(self):
        targets = [[1, 2, 3], [0, 1]]
        batch_size = len(targets)
        max_inp_len = 8
        enc_hiddens = torch.randn(max_inp_len, batch_size, 2 * self.hidden_size)
        dec_hidden_init = torch.randn(batch_size, self.hidden_size)
        pad_token = 9
        
        loss = self.dec.loss(enc_hiddens, dec_hidden_init, targets, pad_token)
        self.assertEqual(loss.shape, ())
        


class TestBiEncoder(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 10
        self.hidden_size = 5
        self.embedding_dim = 3
        self.bienc = BiEncoder(self.vocab_size, self.embedding_dim, self.hidden_size)
    
        self.tokens = [[0, 4, 4, 2], [1,2]]
        self.lengths = [len(t) for t in self.tokens]
        self.padded_tokens = pad_tokens(self.tokens)
        self.batch_size = self.padded_tokens.shape[1]
        self.maxlen = self.padded_tokens.shape[0]

    def test__embed(self):
        e = self.bienc._embed(self.padded_tokens[0, :])
        self.assertEqual(e.shape, (self.batch_size,self.embedding_dim))

    def test__forward(self):
       h = self.bienc._forward(self.padded_tokens, self.lengths, self.bienc.ltr_cell)
       self.assertEqual(h.shape, (self.maxlen, self.batch_size, self.hidden_size))

    def test__ltr_forward(self):
        h = self.bienc._ltr_forward(self.padded_tokens, self.lengths)
        self.assertEqual(h.shape, (self.maxlen, self.batch_size, self.hidden_size))

    def test__rtl_forward(self):
        h = self.bienc._rtl_forward(self.padded_tokens, self.lengths)
        self.assertEqual(h.shape, (self.maxlen, self.batch_size, self.hidden_size))

    def test_forward(self):
        h, dec_init= self.bienc.forward(self.padded_tokens, self.lengths)
        self.assertEqual(h.shape, (self.maxlen, self.batch_size, 2 * self.hidden_size) )
        self.assertEqual(dec_init.shape, (self.batch_size, self.hidden_size))

    def test_flip_padded(self):
        h = pad_sequence([torch.tensor([1, 1, 2, 4]), torch.tensor([3, 3])])
        lengths=[4, 2]
        hf = flip_padded(h, lengths)
        self.assertEqual(hf.shape, h.shape)
        self.assertAlmostEqual(tensordiff(hf, torch.tensor([[4, 3], [2, 3], [1, 0], [1, 0]])), 0)

class TestDecoderCell(unittest.TestCase):
    """Most of these are just checking for correct tensor shapes"""


    def setUp(self):
        self.input_size = 2
        self.hidden_size = 4
        self.attention_size = 5
        self.output_hidden_size = 6
        self.vocab_size = 20
        self.dc = DecoderCell(self.input_size, self.hidden_size, self.attention_size, self.output_hidden_size, self.vocab_size)
    
    def test__attention_energies(self):
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        encoder_hiddens = torch.randn(input_length, batch_size, 2 * self.hidden_size)
        e = self.dc._attention_energies(s, encoder_hiddens)
        self.assertEqual(e.shape, (input_length, batch_size))
    
    def test__attention_weights(self):
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        encoder_hiddens = torch.randn(input_length, batch_size, 2 * self.hidden_size)
        e = self.dc._attention_weights(s, encoder_hiddens)
        self.assertEqual(e.shape, (input_length, batch_size))
        self.assertAlmostEqual(tensordiff(e.sum(dim=0), torch.ones(batch_size)), 0)

    def test__hidden_with_context(self):
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        x = torch.randn(batch_size, self.input_size)
        c = torch.randn(batch_size, 2 * self.hidden_size)
        s2 = self.dc._hidden_with_context(x, s, c)
        self.assertEqual(s2.shape, s.shape)

    def test__context(self):
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        enc_hiddens = torch.randn(input_length, batch_size, 2 * self.hidden_size)
        c = self.dc._context(s, enc_hiddens)
        self.assertEqual(c.shape, (batch_size, 2 * self.hidden_size))

    def test__hidden(self):
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        x = torch.randn(batch_size, self.input_size)
        enc_hiddens = torch.randn(input_length, batch_size, 2 * self.hidden_size)
        s2 = self.dc._hidden(x, s, enc_hiddens)
        self.assertEqual(s2.shape, s2.shape)

    def test__output_hidden(self):
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        x = torch.randn(batch_size, self.input_size)
        c = torch.randn(batch_size, 2 * self.hidden_size)
        t = self.dc._output_hidden(x, s, c)
        self.assertEqual(t.shape, (batch_size, self.output_hidden_size))

    def test__logits(self):
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        x = torch.randn(batch_size, self.input_size)
        c = torch.randn(batch_size, 2 * self.hidden_size)
        y = self.dc._logits(x, s, c)
        self.assertEqual(y.shape, (batch_size, self.vocab_size))

    def test_forward(self):    
        batch_size = 7
        input_length = 3
        s = torch.randn(batch_size, self.hidden_size)
        x = torch.randn(batch_size, self.input_size)
        enc_hiddens = torch.randn(input_length, batch_size, 2 * self.hidden_size)
        s2, logits = self.dc(x, s, enc_hiddens)
        self.assertEqual(s2.shape, s.shape)
        self.assertEqual(logits.shape, (batch_size, self.vocab_size))


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