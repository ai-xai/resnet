import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm import tqdm


def pil_loader_safe(path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except OSError:
        print(f"Skipping corrupted image: {path}")
        return None


def get_transforms() -> v2.Transform:
    return v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def get_dataset(path: Path, transforms: nn.Module | None = None) -> Dataset:
    return ImageFolder(path, loader=pil_loader_safe, transform=transforms)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        if self.device == "cuda":
            self.model = nn.DataParallel(self.model)

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9
        )

    def train(
        self,
        n_epochs: int,
        final_model_path: Path,
        checkpoint_dir: Path,
        plot_loss: bool = True,
        checkpoint_path: Path | None = None,
        checkpoint_interval: int = 10,
    ) -> None:
        """
        Runs training for a given number of epochs, with optional checkpointing
        and loss visualization.

        Args:
            n_epochs: Number of epochs to train the model.
            final_model_path: Path where the final model checkpoint will be saved.
            checkpoint_dir: Directory where intermediate checkpoints will be saved.
            plot_loss: Whether to plot the loss curve after training. Default is True.
            checkpoint_path: Optional path to a checkpoint to resume training from.
            checkpoint_interval: Number of epochs between checkpoint saves.

        Returns:
            None
        """
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.model.train()

        loss_data = {"train": [], "test": []}
        for epoch in range(n_epochs):
            total_loss = 0

            for X, y in tqdm(
                self._train_dataloader, desc=f"epoch {epoch + 1}/{n_epochs}"
            ):
                X = X.to(self.device)
                y = y.to(self.device)

                logits = self.model(X)

                loss = self.criterion(logits, y)
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            logging.info(
                f"epoch: {epoch + 1} | loss: {total_loss / len(self._train_dataloader):.5f}"
            )
            loss_data["train"].append(total_loss / len(self._train_dataloader))

            loss_data["test"].append(self._test_epoch())

            if (epoch + 1) % checkpoint_interval == 0 and (epoch + 1) != n_epochs:
                filename = f"{datetime.now().strftime('%Y%m%d')}_{epoch + 1}_{final_model_path.stem}.pth"
                self._save_model(checkpoint_dir / filename)

        self._save_model(final_model_path)

        if plot_loss:
            self._plot_loss(loss_data)

    def _test_epoch(self) -> float:
        """
        Evaluates the model on the test/validation dataset for one epoch.

        Performs a forward pass on all batches, computes the loss and accuracy,
        and logs the accuracy. No gradient updates are performed.

        Returns:
            float: Average loss over the entire test dataset.
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X, y in tqdm(self._test_dataloader, desc="validation: "):
                X = X.to(self.device)
                y = y.to(self.device)

                logits = self.model(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * X.size(0)
                total_correct += (torch.argmax(logits, dim=-1) == y).sum().item()
                total_samples += X.size(0)

        accuracy = total_correct / total_samples
        logging.info(f"Accuracy: {accuracy:.4f}")

        avg_loss = total_loss / total_samples
        return avg_loss

    def _plot_loss(self, data: dict[str, list[float]]) -> None:
        """
        Plots the training loss curve.

        Args:
            data: Dictionary containing loss values with the following keys:
                - "train": list of floats representing training loss per epoch
                - "test": list of floats representing testing loss per epoch

        Returns:
            None
        """
        plt.plot(data["train"], label="train")
        plt.plot(data["test"], label="test")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Train/test loss")
        plt.show()

    def _save_model(self, path: Path) -> None:
        """
        Saves the model and optimizer state to a checkpoint file.

        This method will automatically create the parent directory of `path`
        if it does not already exist.

        Args:
            path: Filesystem path where the model checkpoint should be saved.

        Returns:
            None
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "optimizer": self.optimizer.state_dict(),
        }
        if self.device == "cuda":
            model = self.model.module
        else:
            model = self.model

        data["weights"] = model.state_dict()  # type: ignore

        torch.save(data, path)
        logging.info(f"Model state saved in: {path}")

    def _load_checkpoint(self, path: Path) -> None:
        """
        Loads model and optimizer state from a previously saved checkpoint.

        The checkpoint is loaded using `map_location=self.device` to ensure
        compatibility when moving between GPU and CPU environments.

        Args:
            path: Path to a checkpoint file created with `_save_model`.

        Returns:
            None
        """
        if not path.exists():
            logging.error(f"File not found: {path}")
            return

        data = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(data["optimizer"])

        if self.device == "cuda":
            self.model.module.load_state_dict(data["weights"])  # type: ignore
        else:
            self.model.load_state_dict(data["weights"])

        logging.info(f"Checkpoint: {path} successfully loaded")
