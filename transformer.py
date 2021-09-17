class Transformer(nn.Module):
    def __init__(self, d_model, nhead, nlayers):
        super(Transformer, self).__init__()
        self.enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0)
        self.enc = nn.TransformerEncoder(self.enc_layer, num_layers=nlayers)

    def forward(self, input):
        enc_out = self.enc(input) # input [bs , seq_len , d_model]
        return enc_out # enc_out [bs , seq_len , d_model]