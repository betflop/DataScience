import random
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)

        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear(enc_hid_dim, dec_hid_dim)
        
    def forward(self, hidden, encoder_outputs):
        attention_output = torch.einsum(
            "bij, kij -> kib",
            self.attn(encoder_outputs),
            hidden
        )
        #attention_output = [n_layers_dec, batch_size, scr sent len]

        attention_output = nn.functional.softmax(attention_output, -1)

        return attention_output
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(num_layers=1,
                           hidden_size=self.dec_hid_dim,
                           input_size=self.emb_dim + self.dec_hid_dim + self.enc_hid_dim)
        #prev dec hidden state + emb + attention
        
        self.out = nn.Linear(self.enc_hid_dim + self.dec_hid_dim + self.emb_dim,
                             self.output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):

        input = input.unsqueeze(0)

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # embedded = [1, batch size, emb dim]
        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # attention_vec = [n_layers_dec, batch_size, scr sent len]

        # Attention part
        attention_vec = self.attention(hidden, encoder_outputs)

        weighted = torch.einsum('kij, bik -> bij', encoder_outputs, attention_vec)

        # weighted = weighted[0]  # 1 layer case
        # weighted = [n_layers, batch_size, enc_hid_dim]

        # LSTM unit prediction
        output, (hidden, cell) = self.rnn(torch.cat((embedded, weighted, hidden), dim=-1))
        # assert False

        output = torch.cat((embedded, weighted, output), dim=-1)

        prediction = self.out(output.squeeze(0))

        return prediction, hidden



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):

            output, hidden = self.decoder(input, hidden, enc_states)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs