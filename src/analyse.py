import os
import matplotlib.pyplot as plt

from collections import Counter
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Compose, RandomRotation


def visualize_images(image_paths, n_images=5):
    plt.figure(figsize=(20, 4))
    for i, image_path in enumerate(image_paths[:n_images]):
        image = read_image(image_path)
        plt.subplot(1, n_images, i + 1)
        plt.imshow(to_pil_image(image))
        plt.axis('off')
    plt.show()


def class_distribution(directory):
    labels = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    counts = Counter(labels)
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.show()


def visualize_augmentation(image_path):
    transform = Compose([RandomRotation(30)])
    image = read_image(image_path)
    augmented_image = transform(image)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(to_pil_image(image))
    ax[0].set_title('Original')
    ax[1].imshow(to_pil_image(augmented_image))
    ax[1].set_title('Augmented')
    plt.show()


if __name__ == '__main__':
    # visualize_images([
    #     "/home/cognaize/Desktop/ASL-Alphabet-Classifier/data/split_data/test/H/H34.jpg",
    # "/home/cognaize/Desktop/ASL-Alphabet-Classifier/data/split_data/test/Q/Q24.jpg",
    # "/home/cognaize/Desktop/ASL-Alphabet-Classifier/data/split_data/test/O/O101.jpg",
    # "/home/cognaize/Desktop/ASL-Alphabet-Classifier/data/split_data/test/space/space95.jpg",
    # "/home/cognaize/Desktop/ASL-Alphabet-Classifier/data/split_data/test/V/V39.jpg",
    # "/home/cognaize/Desktop/ASL-Alphabet-Classifier/data/split_data/test/H/H34.jpg"])

    visualize_augmentation("/home/cognaize/Desktop/ASL-Alphabet-Classifier/data/split_data/test/H/H34.jpg")