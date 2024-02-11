import os
import splitfolders
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from config import Config as CFG


class ASLDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CFG.label_names)

        # Load data
        for label in os.listdir(directory):
            label_dir = os.path.join(directory, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    self.images.append(os.path.join(label_dir, img_file))
                    self.labels.append(self.label_encoder.transform([label]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def check_image_sizes(dataset, num_samples=10):
    """
    Prints the size of 'num_samples' images from the dataset.

    Parameters:
    - dataset: An instance of the dataset.
    - num_samples: Number of samples to check (default is 10).
    """
    for i in range(min(num_samples, len(dataset))):
        image, _ = dataset[i]
        # Convert from PyTorch tensor to PIL Image to get size
        image = transforms.ToPILImage()(image).convert("RGB")
        print(f"Image {i} size: {image.size}")


def get_transformations():
    common_transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),])

    train_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomRotation(90),
            transforms.RandomRotation(30),
            transforms.RandomRotation(60)
        ], p=0.25),
        transforms.Resize((50, 50)),
        transforms.ToTensor()
    ])

    return train_transform, common_transform, common_transform


if __name__ == '__main__':
    train_transform, val_transform, test_transform = get_transformations()
    # splitfolders.ratio(input="../data/asl_alphabet_train/asl_alphabet_train",
    #                    output="../data/split_data", seed=42, ratio=(0.8, 0.1, 0.1))

    train_dataset = ASLDataset(directory='../data/split_data/train',
                               transform=train_transform)
    val_dataset = ASLDataset(directory="../data/split_data/val",
                             transform=val_transform)
    test_dataset = ASLDataset(directory="../data/split_data/test",
                              transform=test_transform)

    check_image_sizes(train_dataset, num_samples=5)
    check_image_sizes(test_dataset, num_samples=5)
    check_image_sizes(val_dataset, num_samples=5)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
