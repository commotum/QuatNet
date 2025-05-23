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
    target_mse: float = 1e-4,  # Target MSE to reach
    max_epochs: int = 10000,   # Maximum epochs to prevent infinite training
    learning_rate: float = 1e-3,
    lr_decay_step: int = 50,
    lr_decay_gamma: float = 0.5,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "models",
    resume: bool = True,
    save_all: bool = False,
):
    """Train the autoencoder until target MSE is reached or max epochs.

    Args:
        model: The neural network to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for validation.
        target_mse: Target mean squared error to reach before stopping.
        max_epochs: Maximum number of epochs to train (prevents infinite training).
        learning_rate: Optimizer learning rate.
        lr_decay_step: Step size for :class:`torch.optim.lr_scheduler.StepLR`.
            If ``0``, no scheduler is used.
        lr_decay_gamma: Multiplicative factor of learning rate decay.
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
    scheduler = None
    if lr_decay_step > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma
        )
        logger.info(
            "Using StepLR scheduler: step_size=%d, gamma=%s",
            lr_decay_step,
            lr_decay_gamma,
        )
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
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError as err:
            logger.warning("Failed to load checkpoint: %s", err)
            logger.warning("Starting from scratch with randomly initialized model")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            start_epoch = ckpt.get("epoch", -1) + 1
            best_loss = ckpt.get("best_loss", best_loss)
            logger.info("Resumed at epoch %d with best loss %.6f", start_epoch, best_loss)

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            batch = batch.view(-1, batch.size(-2), 4)
            batch[:, :, 0] = 0.0  # ensure purely imaginary
            out = model(batch)
            loss = criterion(out[:, :, 1:], batch[:, :, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info("Epoch %d training loss %.6f (%.2fs)", epoch + 1, avg_epoch_loss, time.time() - start)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch = batch.view(-1, batch.size(-2), 4)
                batch[:, :, 0] = 0.0
                out = model(batch)
                val_loss += criterion(out[:, :, 1:], batch[:, :, 1:]).item()
        val_loss /= len(test_loader)
        logger.info("Epoch %d validation loss %.6f", epoch + 1, val_loss)

        if scheduler is not None:
            scheduler.step()
            logger.info("Updated learning rate to %.6e", optimizer.param_groups[0]['lr'])

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

        # Check if we've reached target MSE
        if avg_epoch_loss <= target_mse:
            logger.info("Reached target MSE of %.6f at epoch %d", target_mse, epoch + 1)
            break

        if device.type == "cuda":
            # release cached memory after each epoch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    PATCH_SIZE = 4
    INPUT_Q = PATCH_SIZE * PATCH_SIZE
    HIDDEN_Q = 4

    parser = argparse.ArgumentParser(description="Train quaternion compression autoencoder")
    parser.add_argument("--target-mse", type=float, default=1e-4, help="Target MSE to reach before stopping")
    parser.add_argument("--max-epochs", type=int, default=10000, help="Maximum number of epochs to train")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory for checkpoints")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint")
    parser.add_argument("--save-all", action="store_true", help="Keep checkpoint for every epoch")
    parser.add_argument("--lr-decay-step", type=int, default=0, help="Step size for learning rate decay (0 to disable)")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.1, help="Decay factor for learning rate")
    args = parser.parse_args()

    train_loader, test_loader = create_data_loaders("data/train", "data/test", batch_size=128, patch_size=PATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompressionAutoencoder(INPUT_Q, HIDDEN_Q)
    train_model(
        model,
        train_loader,
        test_loader,
        target_mse=args.target_mse,
        max_epochs=args.max_epochs,
        device=device,
        save_dir=args.save_dir,
        resume=not args.no_resume,
        save_all=args.save_all,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
    )
