import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config as CFG
from src.model import ASLClassifier
from src.data_preprocessing import ASLDataset, get_transformations


def plot_images_with_labels(images, true_labels, predicted_labels, label_map, n=5):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        plt.imshow(img)
        ax.set_title(f"True: {label_map[int(true_labels[i][0])]}\nPred: {label_map[int(predicted_labels[i])]}")
        plt.axis("off")
    plt.show()


def plot_results_in_random_imgs(model: pl.LightningModule, test_loader: DataLoader):
    images, labels = next(iter(test_loader))
    images, labels = images.to(CFG.device), labels.to(CFG.device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    indices = torch.randint(0, len(labels), (5,))
    plot_images_with_labels(images[indices], labels[indices], preds[indices], CFG.label_map)


def evaluate_on_test_set(model: pl.LightningModule, test_loader: DataLoader):
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(CFG.device), labels.to(CFG.device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.view(-1).cpu().numpy())
            actuals.extend(labels.view(-1).cpu().numpy())

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='macro')
    recall = recall_score(actuals, predictions, average='macro')
    f1 = f1_score(actuals, predictions, average='macro')

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def predict_from_img_path(model: pl.LightningModule, image_path: str):
    _, _, transform = get_transformations()
    image = Image.open(image_path)

    image_tensor = transform(image).unsqueeze(0).to(CFG.device)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).item()

    print(f'Predicted class: {CFG.label_map[prediction]}')


if __name__ == '__main__':
    ckp_path_="../checkpoints/asl-epoch=97-val_loss=0.06-val_acc=0.98-train_loss=0.32-train_acc=0.90.ckpt"
    _, _, test_transform = get_transformations()
    test_dataset = ASLDataset(directory="../data/split_data/test",
                              transform=test_transform)
    test_loader_ = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=15)

    model_ = ASLClassifier.load_from_checkpoint(ckp_path_)
    model_.to(CFG.device)
    model_.eval()

    # evaluate_on_test_set(model=model_, test_loader=test_loader_)
    plot_results_in_random_imgs(model=model_, test_loader=test_loader_)
    # predict_from_img_path(model=model_, image_path='../data/istockphoto-1182224876-612x612.jpg')
