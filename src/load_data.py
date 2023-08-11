import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
