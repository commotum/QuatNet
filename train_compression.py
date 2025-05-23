import logging
from pathlib import Path
import time
import torch
import torch.nn as nn
from quatnet_pytorch_layer import QuatNetPytorchDenseLayer
from data_loader import create_data_loaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompressionAutoencoder(nn.Module):
    """Simple quaternion autoencoder using Parcollet-style layers."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = QuatNetPytorchDenseLayer(input_dim, hidden_dim, activation="split_tanh")
        self.decoder = QuatNetPytorchDenseLayer(hidden_dim, input_dim, activation="pure_imag_sigmoid")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)


def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    *,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "models",
    resume: bool = True,
    save_all: bool = False,
):
    """Train the autoencoder with checkpointing.

    Args:
        model: The neural network to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for validation.
        num_epochs: Number of additional epochs to train.
        learning_rate: Optimizer learning rate.
        device: Torch device to use.
        save_dir: Directory where checkpoints are stored.
        resume: If True, resume from ``latest.pth`` in ``save_dir`` when present.
        save_all: If True, keep a checkpoint for every epoch in addition to
            ``latest.pth`` and ``best.pth``.
    """

    logger.info("Starting training on %s", device)
    if device.type == "cuda":
        # free any cached memory from previous runs
        torch.cuda.empty_cache()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    latest_file = save_path / "latest.pth"
    best_file = save_path / "best.pth"
    start_epoch = 0
    best_loss = float("inf")

    if resume and latest_file.exists():
        logger.info("Resuming from %s", latest_file)
        ckpt = torch.load(latest_file, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_loss = ckpt.get("best_loss", best_loss)
        logger.info("Resumed at epoch %d with best loss %.6f", start_epoch, best_loss)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            # Resolved conflict: Keep the batch reshaping line
            batch = batch.view(-1, batch.size(-2), 4)
            batch[:, :, 0] = 0.0  # ensure purely imaginary
            out = model(batch)
            loss = criterion(out[:, :, 1:], batch[:, :, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info("Epoch %d training loss %.6f (%.2fs)", epoch + 1, epoch_loss / len(train_loader), time.time() - start)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                # Resolved conflict: Keep the batch reshaping line
                batch = batch.view(-1, batch.size(-2), 4)
                batch[:, :, 0] = 0.0
                out = model(batch)
                val_loss += criterion(out[:, :, 1:], batch[:, :, 1:]).item()
        val_loss /= len(test_loader)
        logger.info("Epoch %d validation loss %.6f", epoch + 1, val_loss)

        # Save checkpoints
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "best_loss": best_loss},
                best_file,
            )
            logger.info("Saved new best checkpoint to %s", best_file)

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "best_loss": best_loss},
            latest_file,
        )

        if save_all:
            torch.save(model.state_dict(), save_path / f"checkpoint_epoch_{epoch+1}.pth")

        if device.type == "cuda":
            # release cached memory after each epoch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    PATCH_SIZE = 128
    INPUT_Q = PATCH_SIZE * PATCH_SIZE
    HIDDEN_Q = 16

    parser = argparse.ArgumentParser(description="Train quaternion compression autoencoder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory for checkpoints")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint")
    parser.add_argument("--save-all", action="store_true", help="Keep checkpoint for every epoch")
    args = parser.parse_args()

    train_loader, test_loader = create_data_loaders("data/train", "data/test", batch_size=256, patch_size=PATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompressionAutoencoder(INPUT_Q, HIDDEN_Q)
    train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        resume=not args.no_resume,
        save_all=args.save_all,
    )
