import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.Ws = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.Wout = nn.Linear(2 * hidden_size, hidden_size)
        
        # raise NotImplementedError("Add your implementation.")

    def forward(self, query, encoder_outputs, src_lengths):
        """
        query:          (batch_size, max_tgt_len, hidden_size)
        encoder_outputs:(batch_size, max_src_len, hidden_size)
        src_lengths:    (batch_size)
        Returns:
            attn_out:   (batch_size, max_tgt_len, hidden_size) - attended vector
        """
        batch_size, max_src_len, _ = encoder_outputs.size()
        print(f"batch_size: {batch_size}")
        print(f"max_src_len: {max_src_len}")
        print(f"encoder_outputs shape: {encoder_outputs.shape}")

        # Pad encoder_outputs to match query length
        encoder_outputs = F.pad(encoder_outputs, (0, 0, 0, 1, 0, 0))  # (batch_size, max_src_len + 1, hidden_size)

        # Compute alignment scores
        # query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        print(f"query shape: {query.shape}")
        scores = self.v(torch.tanh(self.Ws(query).unsqueeze(1) + self.Wh(encoder_outputs)))  # (batch_size, max_src_len, 1)
        print(f"scores shape: {scores.shape}")
        scores = scores.squeeze(-1)  # (batch_size, max_src_len)
        print(f"scores shape 2: {scores.shape}")

        # Mask padding positions
        mask = self.sequence_mask(src_lengths).to(encoder_outputs.device)
        mask = F.pad(mask, (0, 1))  # (batch_size, max_src_len + 1)
        scores.data.masked_fill_(~mask, -float("inf"))

        # Normalize scores with softmax
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, max_src_len)

        # Debug prints to check tensor shapes
        print(f"attention_weights shape: {attention_weights.shape}")
        print(f"encoder_outputs shape: {encoder_outputs.shape}")

        # Ensure attention_weights is 2D and encoder_outputs is 3D
        assert len(attention_weights.shape) == 2, "attention_weights must be 2D"
        assert len(encoder_outputs.shape) == 3, "encoder_outputs must be 3D"

        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_size)
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_size)

        # Concatenate context vector with query
        combined = torch.cat((context_vector, query), dim=1)  # Concatenate context vector and decoder state
        attn_out = torch.tanh(self.Wout(combined))           # Compute attention-enhanced state

        return attn_out, attention_weights

        raise NotImplementedError("Add your implementation.")

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################
        embedded = self.dropout(self.embedding(src))
        packed = pack(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        enc_output, _ = unpack(packed_output, batch_first=True)
        return enc_output, (hidden, cell)

        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        raise NotImplementedError("Add your implementation.")


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        #############################################
        # TODO: Implement the forward pass of the decoder
        # Hints:
        # - the input to the decoder is the previous target token,
        #   and the output is the next target token
        # - New token representations should be generated one at a time, given
        #   the previous token representation and the previous decoder state
        # - Add this somewhere in the decoder loop when you implement the attention mechanism in 3.2:
        # if self.attn is not None:
        #     output = self.attn(
        #         output,
        #         encoder_outputs,
        #         src_lengths,
        #     )
        #############################################
        
        # Remove the last token from the target sequence
        print(f"tgt shape: {tgt.shape}")
        if tgt.size(1) > 1:
            tgt = tgt[:, :-1]
        print(f"tgt shape: {tgt.shape}")
        
        # Apply embedding and dropout
        embed = self.embedding(tgt)
        embed = self.dropout(embed)

        # Output and dropout
        out1, dec_state = self.lstm(embed, dec_state)
        print(f"out1 shape: {out1.shape}")
        print(f"dec_state shape: {dec_state[0].shape}")
        
        # Attention layer (for 3.1b only)
        if self.attn is not None:
            out1, attn_weights = self.attn(
                out1,
                encoder_outputs,
                src_lengths,
            )
        
        print(f"out1 shape 2: {out1.shape}")

        outputs = self.dropout(out1)

        return outputs, dec_state

        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)
        raise NotImplementedError("Add your implementation.")


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
