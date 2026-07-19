import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def train_model(model, loader, criterion, optimiser, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(loader, desc='Training', leave=False) as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad(set_to_none=True)

            # Forward pass
            outputs = model(inputs)
            loss1 = criterion(outputs.logits, labels)
            loss2 = criterion(outputs.aux_logits, labels)
            loss = loss1 + 0.4 * loss2

            # Backward pass and optimisation
            loss.backward()
            optimiser.step()

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate_model(model, loader, criterion, device):
    """validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(), tqdm(loader, desc='Evaluation', leave=False) as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# splits and generates dataloaders for each data splits
def data_generator(parent_directory: str, transform: transforms.Compose, batch_size: int, subset: str):
    if subset not in ["Train", "Val", "Test"]:
        raise ValueError('Subset must be one of "Train", "Val", "Test"')

    if subset == "Test":
        dataset = datasets.ImageFolder(
            parent_directory,
            transform=transform
        )

        test_generator = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return test_generator

    elif subset in ["Train", "Val"]:
        dataset = datasets.ImageFolder(
            parent_directory,
            transform=transform
        )

        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            lengths = [train_size, val_size],
            generator = torch.Generator().manual_seed(42)
        )

        if subset == "Train":
            train_generator = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            return train_generator

        else:
            val_generator = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            return val_generator

    return None


# resizes val and test images, and augment training images.
def image_gen_w_aug(train_parent_directory, test_parent_directory, batch_size: int):
    train_batch_size = batch_size
    val_batch_size = batch_size*2
    test_batch_size = batch_size*2

    # Image resizing and augmentation (training only)
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomRotation(30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2)
        ),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    # train, val, test generators
    train_generator = data_generator(train_parent_directory, train_transform, train_batch_size, subset="Train")
    val_generator = data_generator(train_parent_directory, test_transform, val_batch_size, subset="Val")
    test_generator = data_generator(test_parent_directory, test_transform, test_batch_size, subset="Test")

    return train_generator, val_generator, test_generator


def build_model(pre_trained_model, num_classes: int):
    pre_trained_model.fc = nn.Sequential(
        nn.Linear(pre_trained_model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return pre_trained_model


# isle_of_gods (2022) Answer to: 'early stopping in PyTorch'. Stack Overflow.
# Available at: https://stackoverflow.com/a/73704579 (Accessed: 17 July 2026).
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False