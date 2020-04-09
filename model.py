"""Defines the RNN-plus-attention model described in the original paper."""
import torch
import torch.nn as nn

def batch_mul(W, x):
    """ Applies a matrix-vec mutliplication that respects batching.
        W: a (hidden, inp) matrix
        x: a (batch_size, inp) vector
        returns: (batch_size, hidden) vector"""
    return torch.matmul(W.unsqueeze(0), x.unsqueeze(-1)).squeeze()

class EncoderCell(nn.Module):
    """ Gated recurrent unit"""

    def __init__(self, input_size, hidden_size, dtype=torch.float):
        """input_size: int, dimensionality of the input vectors.
            hidden_size: int, dim of the hidden vectors.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        
        def make(*shape):
            return nn.Parameter(torch.Tensor(*shape).to(dtype=self.dtype))

        #embedding matrices, of shape (hidden_size, input_size)
        self.Wh = make(self.hidden_size, self.input_size)
        self.Wz = make(self.hidden_size, self.input_size)
        self.Wr = make(self.hidden_size, self.input_size)

        #update matrices, of shape (hidden size, hiddenn size)
        self.Uh = make(self.hidden_size, self.hidden_size)
        self.Uz = make(self.hidden_size, self.hidden_size)
        self.Ur = make(self.hidden_size, self.hidden_size)

        #bias vectors for the updates, of shape (hidden size,)
        self.bh = make(self.hidden_size)
        self.bz = make(self.hidden_size)
        self.br = make(self.hidden_size)
        

        self.embeddings = [self.Wh, self.Wz, self.Wr]
        self.updates = [self.Uh, self.Uz, self.Ur]
        self.biases = [self.bh, self.bz, self.br]

        #initialize everything
        for W in self.embeddings + self.updates:
            nn.init.xavier_normal_(W)
        for b in self.biases:
            b.data.zero_()
        
    
    def _update_h(self, x, r, hprev):
        """Compute the proposed hidden update from r and the previous hidden state."""
        return torch.tanh(batch_mul(self.Wh, x) + batch_mul(self.Uh, r * hprev) + self.bh)

    def _update_z(self, x, hprev):
        """Compute the new gate vector from input x and previous hidden state."""
        return torch.sigmoid(batch_mul(self.Wz, x) + batch_mul(self.Uz, hprev) + self.bz)
    
    def _update_r(self, x, hprev):
        """Compute the new reset vector from input x and previous hidden state."""
        return torch.sigmoid(batch_mul(self.Wr, x) + batch_mul(self.Ur, hprev) + self.br)

    def forward(self, x, hprev):
        """ Run forward pass on the given input vector.
            hprev = (batch_size, hidden_dim) hidden state from the previous timestep.
            Returns: the new hidden state, same shape as hprev."""
        
        r = self._update_r(x, hprev)
        z = self._update_z(x, hprev)
        h_proposed = self._update_h(x, r, hprev)
        return z * h_proposed + (1 - z) * hprev

class DecoderCell(nn.Module):
    """Decoder cell which conditions on previous hidden state as well as attention-context."""

    def __init__(self, input_size, hidden_size, attention_size, dtype=torch.float):
        """input_size: int, dimenionality of the input (or output) vectors
            hidden_size: int, dimensionality of the decoder hidden state
            attention_size: int, dimensionality of the vectors used to define attention scores.
            """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.dtype = dtype
    
        def make(*shape):
            return nn.Parameter(torch.Tensor(*shape).to(dtype=self.dtype))

        #embedding matrices, of shape (hidden_size, input_size)
        self.Wh = make(self.hidden_size, self.input_size)
        self.Wz = make(self.hidden_size, self.input_size)
        self.Wr = make(self.hidden_size, self.input_size)

        #update matrices, of shape (hidden size, hiddenn size)
        self.Uh = make(self.hidden_size, self.hidden_size)
        self.Uz = make(self.hidden_size, self.hidden_size)
        self.Ur = make(self.hidden_size, self.hidden_size)

        #context matrices, of shape (hidden_size, 2 * hidden_size)
        self.Ch = make(self.hidden_size, 2 * self.hidden_size)
        self.Cz = make(self.hidden_size, 2 * self.hidden_size)
        self.Cr = make(self.hidden_size, 2 * self.hidden_size)

        #bias vectors for the updates, of shape (hidden size,)
        self.bh = make(self.hidden_size)
        self.bz = make(self.hidden_size)
        self.br = make(self.hidden_size)

        #tensors for computing the attention scores
        self.va = make(self.attention_size)
        # this one couples to the decoder hidden state
        self.Wa = make(self.attention_size, self.hidden_size)
        # this one couples to the bidirectional encoder hidden state.
        self.Ua = make(self.attention_size, 2 * self.hidden_size)

        self.embeddings = [self.Wh, self.Wz, self.Wr]
        self.updates = [self.Uh, self.Uz, self.Ur]
        self.contexts = [self.Ch, self.Cz, self.Cr]
        self.biases = [self.bh, self.bz, self.br]

        #initialize everything
        for W in self.embeddings + self.updates + self.contexts:
            nn.init.xavier_normal_(W)
        for b in self.biases:
            b.data.zero_()
    
    def _update_h(self, x, r, hprev, c):
        """Compute the proposed hidden update from r, the previous hidden state, and the context c"""
        return torch.tanh(batch_mul(self.Wh, x) + batch_mul(self.Uh, r * hprev) + batch_mul(self.Ch, c) + self.bh)

    def _update_z(self, x, hprev, c):
        """Compute the new gate vector from input x, previous hidden state, and context c"""
        return torch.sigmoid(batch_mul(self.Wz, x) + batch_mul(self.Uz, hprev) + batch_mul(self.Cz, c) + self.bz)
    
    def _update_r(self, x, hprev, c):
        """Compute the new reset vector from input x, previous hidden state, and context c"""
        return torch.sigmoid(batch_mul(self.Wr, x) + batch_mul(self.Ur, hprev) + batch_mul(self.Cr, c) + self.br)

    def _attention_energies(self, decoder_hidden, encoder_hiddens):
        """ Computes vector of attention energies.
            decoder_hidden: (batch_size, hidden_size) hidden state vector from the previous timestep
            encoder_hidden: (batch_size, 2 * hidden, Lx) tensor holding all hidden states from the input encoding, stacked along the final dimension. 
                Lx = length of the input sequence.
            returns: (batch_size, Lx) vector of energy scores, one for each token in the input.
        """
        #(batch, attn_size, Lx)
        enc_scores = torch.matmul(self.Ua.unsqueeze(0), encoder_hiddens)
        #batch, attn_size
        dec_score = batch_mul(self.Wa, decoder_hidden)
        #batch, Lx, attn size
        v = torch.tanh(dec_score.unsqueeze(-1) + enc_scores).permute(0, 2, 1)
        #batch, Lx
        return torch.matmul(v, self.va)





    def forward_with_context(self, x, hprev, c):
        """ Run forward pass on the given input vector.
            x = (batch_size, input_dim) input vector
            hprev = (batch_size, hidden_dim) hidden state from the previous timestep.
            c = (batch_size, 2 * hidden_dim) context vector for the current timestep
            Returns: the new hidden state, same shape as hprev."""
        
        r = self._update_r(x, hprev, c)
        z = self._update_z(x, hprev, c)
        h_proposed = self._update_h(x, r, hprev, c)
        return z * h_proposed + (1 - z) * hprev


if __name__ == "__main__":
    pass