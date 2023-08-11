from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_features, output_dim):
        super(NeuralNetwork, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_features)
        self.linear2 = nn.Linear(hidden_features, hidden_features)
        self.linear3 = nn.Linear(hidden_features, hidden_features)
        self.linear4 = nn.Linear(hidden_features, output_dim)

        self.relu = nn.ReLU(True)

    def forward(self, x):

        x = x.reshape(-1, 3 * 64 * 64)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)

        return x
