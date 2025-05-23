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


def train_model(model: nn.Module, train_loader, test_loader, *, num_epochs: int = 10, learning_rate: float = 1e-3, device: torch.device = torch.device('cpu'), save_dir: str = 'models'):
    logger.info("Starting training on %s", device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for batch in train_loader:
            batch = batch.to(device)
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
                batch[:, :, 0] = 0.0
                out = model(batch)
                val_loss += criterion(out[:, :, 1:], batch[:, :, 1:]).item()
        logger.info("Epoch %d validation loss %.6f", epoch + 1, val_loss / len(test_loader))
        torch.save(model.state_dict(), save_path / f"checkpoint_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    PATCH_SIZE = 4
    INPUT_Q = PATCH_SIZE * PATCH_SIZE
    HIDDEN_Q = 4

    train_loader, test_loader = create_data_loaders('data/train', 'data/test', batch_size=256, patch_size=PATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CompressionAutoencoder(INPUT_Q, HIDDEN_Q)
    train_model(model, train_loader, test_loader, num_epochs=10, device=device)
