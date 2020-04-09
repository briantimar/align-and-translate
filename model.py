"""Defines the RNN-plus-attention model described in the original paper."""
import torch
import torch.nn as nn

class EncoderCell(nn.Module):
    """ Gated recurrent unit"""

    def __init__(self, input_size, hidden_size):
        """input_size: int, dimensionality of the input vectors.
            hidden_size: int, dim of the hidden vectors.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        #embedding matrices, of shape (hidden_size, input_size)
        self.Wh = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wz = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Wr = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))

        #update matrices, of shape (hidden size, hiddenn size)
        self.Uh = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.Uz = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.Ur = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))

        #bias vectors for the updates, of shape (hidden size,)
        self.bh = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bz = nn.Parameter(torch.Tensor(self.hidden_size))
        self.br = nn.Parameter(torch.Tensor(self.hidden_size))

        self.embeddings = [self.Wh, self.Wz, self.Wr]
        self.updates = [self.Uh, self.Uz, self.Ur]
        self.biases = [self.bh, self.bz, self.br]

        #initialize everything
        for W in self.embeddings + self.updates:
            nn.init.xavier_normal_(W)
        for b in self.biases:
            b.data.zero_()

    def _batch_mul(self, W, x):
        """ Applies a matrix-vec mutliplication that respects batching.
            W: a (hidden, inp) matrix
            x: a (batch_size, inp) vector
            returns: (batch_size, hidden) vector"""
        return torch.matmul(W.unsqueeze(0), x.unsqueeze(-1)).squeeze()
    
    def _update_h(self, x, r, hprev):
        """Compute the proposed hidden update from r and the previous hidden state."""
        return torch.tanh(self._batch_mul(self.Wh, x) + self._batch_mul(self.Uh, r * hprev) + self.bh)

    def _update_z(self, x, hprev):
        """Compute the new gate vector from input x and previous hidden state."""
        return torch.sigmoid(self._batch_mul(self.Wz, x) + self._batch_mul(self.Uz, hprev) + self.bz)
    
    def _update_r(self, x, hprev):
        """Compute the new reset vector from input x and previous hidden state."""
        return torch.sigmoid(self._batch_mul(self.Wr, x) + self._batch_mul(self.Ur, hprev) + self.br)

    def forward(self, x, hprev):
        """ Run forward pass on the given input vector.
            hprev = (batch_size, hidden_dim) hidden state from the previous timestep.
            Returns: the new hidden state, same shape as hprev."""
        
        r = self._update_r(x, hprev)
        z = self._update_z(x, hprev)
        h_proposed = self._update_h(x, r, hprev)
        return z * h_proposed + (1 - z) * hprev


if __name__ == "__main__":
    c = EncoderCell(5, 10)