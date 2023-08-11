import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression


#
# dataset = torchvision.datasets.Country211('D:/Programming/PycharmProjects/image-classification/datasets',
#                                           download=True,
#                                           )
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = datasets.EuroSAT('D:/Programming/PycharmProjects/image-classification/datasets',
                               download=True,
                               transform=transform
                               )

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, test_dataloader = load_data()

    train_dataset = next(iter(train_dataloader))
    test_dataset = next(iter(test_dataloader))

    X = train_dataset[0].reshape(-1, 3 * 64 * 64)
    y = train_dataset[1]

    x_test = test_dataset[0].reshape(-1, 3 * 64 * 64)
    y_test = test_dataset[1]

    model = LogisticRegression(max_iter=10000)
    model.fit(X, y)
    print(model.score(X, y))
    print(model.score(x_test, y_test))