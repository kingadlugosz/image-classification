from src.neural_network import NeuralNetwork


def load_model():

    model = NeuralNetwork(input_dim=3*64*64, hidden_features=25, output_dim=10)

    return model


if __name__ == "__main__":
    load_model()
