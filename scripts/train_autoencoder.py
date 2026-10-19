
from pathlib import Path
import csv
import json
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "bottle" / "train" / "good"
SPLIT_PATH = ROOT / "data" / "processed" / "bottle_split.json"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 25
SEED = 42


class BottleDataset(Dataset):
    def __init__(self, filenames):
        self.paths = [DATA_DIR / name for name in filenames]

        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Images missing from split: {missing[:5]}"
            )

        if not self.paths:
            raise RuntimeError("Dataset split is empty.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )
            array = np.asarray(image, dtype=np.float32) / 255.0

        return torch.from_numpy(array.copy()).permute(2, 0, 1)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def calculate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            total_loss += loss.item() * batch.size(0)
            count += batch.size(0)

    return total_loss / count


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")

    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Create the dataset split first: {SPLIT_PATH}"
        )

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_dataset = BottleDataset(split["train"])
    val_dataset = BottleDataset(split["validation"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    history = []

    print("=== VisionGuard AI: Train / Validation ===")
    print(f"Training images   : {len(train_dataset)}")
    print(f"Validation images : {len(val_dataset)}")
    print(f"Image size        : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Device            : {device}")
    print(f"Epochs            : {EPOCHS}\n")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        count = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            count += batch.size(0)

        train_loss = total_loss / count
        val_loss = calculate_loss(
            model, val_loader, criterion, device
        )

        history.append((epoch + 1, train_loss, val_loss))

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- train_loss: {train_loss:.6f} "
            f"- val_loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model_path = ARTIFACTS_DIR / "baseline_autoencoder.pth"

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": IMAGE_SIZE,
            "architecture": "ConvAutoencoder",
            "training_images": len(train_dataset),
            "validation_images": len(val_dataset),
            "epochs": EPOCHS,
            "best_validation_loss": best_val_loss,
            "seed": SEED,
        },
        model_path,
    )

    history_path = REPORTS_DIR / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "training_reconstruction_loss",
            "validation_reconstruction_loss"
        ])
        writer.writerows(history)

    print("\nTraining completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved: {model_path}")
    print(f"Training history: {history_path}")
    print("This checkpoint was selected by validation reconstruction loss.")


if __name__ == "__main__":
    main()
