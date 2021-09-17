class MLP(nn.Module):
    def __init__(self, input_size, hid_size, output_size):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, output_size)
        )

    def forward(self, input):
        output = self.mlp(input)
        return output