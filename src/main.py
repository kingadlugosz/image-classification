from src.load_data import load_data
from src.load_model import load_model


def main():
    train_dataloader, test_dataloader = load_data()
    model = load_model()
    # model(next(iter(train_dataloader))[0])
    # print('a')


if __name__ == "__main__":
    main()
