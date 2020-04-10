"""Defines the RNN-plus-attention model described in the original paper."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

def batch_mul(W, x):
    """ Applies a matrix-vec mutliplication that respects batching.
        W: a (hidden, inp) matrix
        x: a (batch_size, inp) vector
        returns: (batch_size, hidden) vector"""
    return torch.matmul(W.unsqueeze(0), x.unsqueeze(-1)).squeeze()

def flip_padded(h, lengths):
    """ Flip a (max_len, batch_size) padded tensor h.
        lengths: length of each batch element."""
    maxlen, batch_size = h.shape[0], h.shape[1]
    idx = torch.tensor(list(range(maxlen-1, -1, -1))).long()
    flipped = h.index_select(0, idx)
    for i in range(batch_size):
        flipped[:lengths[i], i, ...] = flipped[maxlen - lengths[i]:, i, ...]
        flipped[lengths[i]:, i, ...] = 0
    return flipped

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
        
        def make(*shape, init='xavier_normal'):
            inits = {'xavier_normal': nn.init.xavier_normal_, 
                    'zero': lambda t: t.data.zero_()}
            if init not in inits:
                raise ValueError(f"Unknown init {init}")
            
            w = nn.Parameter(torch.Tensor(*shape).to(dtype=self.dtype))
            inits[init](w)
            return w

        #embedding matrices, of shape (hidden_size, input_size)
        self.Wh = make(self.hidden_size, self.input_size)
        self.Wz = make(self.hidden_size, self.input_size)
        self.Wr = make(self.hidden_size, self.input_size)

        #update matrices, of shape (hidden size, hiddenn size)
        self.Uh = make(self.hidden_size, self.hidden_size)
        self.Uz = make(self.hidden_size, self.hidden_size)
        self.Ur = make(self.hidden_size, self.hidden_size)

        #bias vectors for the updates, of shape (hidden size,)
        self.bh = make(self.hidden_size, init='zero')
        self.bz = make(self.hidden_size, init='zero')
        self.br = make(self.hidden_size, init='zero')

        self.embeddings = [self.Wh, self.Wz, self.Wr]
        self.updates = [self.Uh, self.Uz, self.Ur]
        self.biases = [self.bh, self.bz, self.br]
    
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


class BiEncoder(nn.Module):
    """A bidirectional GRU layer, along with a shared word embedding."""

    def __init__(self, vocab_size, embedding_size, hidden_size):
        """vocab_size = number of words in the vocabulary
        embedding_size = dimension of the word embeddings
        hidden_size = dimension of the hidden states."""
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.ltr_cell = EncoderCell(self.embedding_size, self.hidden_size)
        self.rtl_cell = EncoderCell(self.embedding_size, self.hidden_size)

    def _embed(self, tokens):
        """ tokens: (batch_size,) tensor of integer tokens.
            returns: (batch_size, embed_dim) embedded tensor. """
        return self.embedding(tokens)

    def _forward(self, padded_tokens, lengths, cell):
        """Compute forward pass for the left-to-right GRU
            padded_tokens: (length, batch_size) padded tensor of input tokens.
            lengths: (batch_size,) tensor of integer lengths, SHOULD BE SORTED
            cell: an EncoderCell instance
            returns: 
                hiddens - (length, batch_size) tensor of hidden states; the length of each hidden sequence in the batch
            is the same as that of the corresponding token sequence.
            """
        L, batch_size = padded_tokens.shape
        packed_tokens = pack_padded_sequence(padded_tokens, lengths)
        #init hidden state with zeros
        h = torch.zeros(batch_size, self.hidden_size)
        hidden_steps = []
        
        batch_start = 0

        for i in range(L):
            cur_batch_size = packed_tokens.batch_sizes[i]
            batch_end = batch_start + cur_batch_size
            token_batch = packed_tokens.data[batch_start:batch_end]
            batch_start = batch_end

            #perform cell computation at this timestep
            inp = self._embed(token_batch)
            h = h[:cur_batch_size, ...]
            h = cell(inp, h)
            hidden_steps.append(h)

        #stack all the hidden outputs.
        return pad_sequence(hidden_steps, batch_first=True)

    def _pad_tokens(self, list_of_sequences):
        """Pad list of int sequences into long tensor."""
        return pad_sequence([torch.tensor(s) for s in list_of_sequences]).long()

    def _ltr_forward(self, list_of_sequences):
        """ Obtain left-to-right hidden states from the given list of sequences.
            Each seq is list of integers.
                return shape: (maxlen, batch_size, hidden_dim)"""
        lengths = [len(s) for s in list_of_sequences]
        padded_tokens = self._pad_tokens(list_of_sequences)
        return self._forward(padded_tokens, lengths, self.ltr_cell)

    def _rtl_forward(self, list_of_sequences):
        """ Right-to-left hidden states from the given list.
            The padding mask for these will match that of the ltr hidden states.
            return shape: (maxlen, batch_size, hidden_dim)""" 
        lengths = [len(s) for s in list_of_sequences]
        padded_tokens = self._pad_tokens(list(reversed(s)) for s in list_of_sequences)
        h = self._forward(padded_tokens, lengths, self.rtl_cell)     
        return flip_padded(h, lengths)

class DecoderCell(nn.Module):
    """Decoder cell which conditions on previous hidden state as well as attention-context."""

    def __init__(self, input_size, hidden_size, attention_size, output_hidden_size, vocab_size, dtype=torch.float):
        """input_size: int, dimenionality of the input (or output) vectors
            hidden_size: int, dimensionality of the decoder hidden state
            attention_size: int, dimensionality of the vectors used to define attention scores.
            output_hidden_size: int, size of the hidden layer before the output logits
            vocab_size: int: number of tokens in the vocabulary.
            """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.output_hidden_size = output_hidden_size
        self.vocab_size = vocab_size
        self.dtype = dtype
    
        def make(*shape, init='xavier_normal'):
            inits = {'xavier_normal': nn.init.xavier_normal_, 
                    'zero': lambda t: t.data.zero_()}
            if init not in inits:
                raise ValueError(f"Unknown init {init}")
            
            w = nn.Parameter(torch.Tensor(*shape).to(dtype=self.dtype))
            inits[init](w)
            return w

        #embedding matrices, of shape (hidden_size, input_size)
        self.Ws = make(self.hidden_size, self.input_size)
        self.Wz = make(self.hidden_size, self.input_size)
        self.Wr = make(self.hidden_size, self.input_size)

        #update matrices, of shape (hidden size, hiddenn size)
        self.Us = make(self.hidden_size, self.hidden_size)
        self.Uz = make(self.hidden_size, self.hidden_size)
        self.Ur = make(self.hidden_size, self.hidden_size)

        #context matrices, of shape (hidden_size, 2 * hidden_size)
        self.Cs = make(self.hidden_size, 2 * self.hidden_size)
        self.Cz = make(self.hidden_size, 2 * self.hidden_size)
        self.Cr = make(self.hidden_size, 2 * self.hidden_size)

        #bias vectors for the updates, of shape (hidden size,)
        self.bs = make(self.hidden_size, init='zero')
        self.bz = make(self.hidden_size, init='zero')
        self.br = make(self.hidden_size, init='zero')

        #tensors for computing the attention scores
        self.va = make(self.attention_size, init='zero')
        # this one couples to the decoder hidden state
        self.Wa = make(self.attention_size, self.hidden_size)
        # this one couples to the bidirectional encoder hidden state.
        self.Ua = make(self.attention_size, 2 * self.hidden_size)

        # tensors for computing the output logits
        self.Uo = make(2 * self.output_hidden_size, self.hidden_size)
        self.Vo = make(2 * self.output_hidden_size, self.input_size)
        self.Co = make(2 * self.output_hidden_size, 2 * self.hidden_size)
        self.Wo = make(self.vocab_size, self.output_hidden_size)

        self.maxout = nn.MaxPool1d(kernel_size=2)

        self.embeddings = [self.Ws, self.Wz, self.Wr]
        self.updates = [self.Us, self.Uz, self.Ur]
        self.contexts = [self.Cs, self.Cz, self.Cr]
        self.biases = [self.bs, self.bz, self.br]
    
    def _update_s(self, x, r, sprev, c):
        """Compute the proposed hidden update from r, the previous hidden state, and the context c"""
        return torch.tanh(batch_mul(self.Ws, x) + batch_mul(self.Us, r * sprev) + batch_mul(self.Cs, c) + self.bs)

    def _update_z(self, x, sprev, c):
        """Compute the new gate vector from input x, previous hidden state, and context c"""
        return torch.sigmoid(batch_mul(self.Wz, x) + batch_mul(self.Uz, sprev) + batch_mul(self.Cz, c) + self.bz)
    
    def _update_r(self, x, sprev, c):
        """Compute the new reset vector from input x, previous hidden state, and context c"""
        return torch.sigmoid(batch_mul(self.Wr, x) + batch_mul(self.Ur, sprev) + batch_mul(self.Cr, c) + self.br)

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

    def _attention_weights(self, decoder_hidden, encoder_hiddens):
        """Compute vector of attention weights given: 
            (batch_size, hidden_size) previous hidden state
            (batch_size, 2 * hidden, Lx) tensor of all input bidirectional hidden states.
        returns: (batch_size, Lx) vector of attention scores, normalized as probs along dim 1."""
        return self._attention_energies(decoder_hidden, encoder_hiddens).softmax(dim=1)

    def _hidden_with_context(self, x, sprev, c):
        """ 
            x = (batch_size, input_dim) input vector
            sprev = (batch_size, hidden_dim) hidden state from the previous timestep.
            c = (batch_size, 2 * hidden_dim) context vector for the current timestep
            Returns: the new hidden state, same shape as hprev."""
        
        r = self._update_r(x, sprev, c)
        z = self._update_z(x, sprev, c)
        s_proposed = self._update_s(x, r, sprev, c)
        return z * s_proposed + (1 - z) * sprev

    def _context(self, sprev, encoder_hiddens):
        """ Computes the context vector c
            sprev = (batch_size, hidden_dim) hidden state from the previous timestep.
            encoder_hiddens = (batch_size, 2 * hidden_dim, Lx) tensor of bilstm hidden states.
            returns: (batch, 2 * hidden_dim) context vec
        """
        #(batch_size, Lx) set of attention weights onto inputs.
        alpha = self._attention_weights(sprev, encoder_hiddens)
        # attention-averaged context, (batch_size, 2 * hidden_dim)
        return (alpha.unsqueeze(1) * encoder_hiddens).sum(2)

    def _hidden(self, x, sprev, encoder_hiddens):
        """ Compute new hidden state for the decoder.
            x = (batch_size, input_dim) input vector
            sprev = (batch_size, hidden_dim) decoder hidden state from previous timestep.
            encoder_hiddens = (batch_size, 2 * hidden_dim, Lx) tensor of bilstm hidden states.
            returns: new (batch_size, hidden_dim) decoder hidden state.
            """
        c = self._context(sprev, encoder_hiddens)
        return self._hidden_with_context(x, sprev, c)

    def _output_hidden(self, x, sprev, c):
        """Compute the output hidden layer values."""
        #(batch_size, 2 * output_hidden_size)
        t_tilde = batch_mul(self.Uo, sprev) + batch_mul(self.Vo, x) + batch_mul(self.Co, c)
        return self.maxout(t_tilde.unsqueeze(1)).squeeze()

    def _logits(self, x, sprev, c):
        """ Compute logits for the decoder output at the current timestep.
            x = (batch, input_size) embedded input
            sprev = (batch, hidden_size) prev decoder hidden state
            c = (batch, 2 * hidden_size) current context
            returns: (batch, vocab_size) logits.
            """
        #batch, 2* output_hidden_size
        t = self._output_hidden(x, sprev, c)
        return batch_mul(self.Wo, t)

    def forward(self, x, sprev, encoder_hiddens):
        """ Performs the full decoder cell hidden pass.
         x = (batch, input_size) embedded input
        sprev = (batch, hidden_size) prev decoder hidden state
        encoder_hiddens = (batch, 2*hidden_size, Lx) tensor of encoder hidden states.
        returns:
            new_hidden, logits
            where
                new_hidden = (batch, hidden_size)
                logits = (batch, vocab_size)
        """
        c = self._context(sprev, encoder_hiddens)
        s = self._hidden_with_context(x, sprev, c)
        logits = self._logits(x, sprev, c)
        return s, logits

if __name__ == "__main__":
    pass