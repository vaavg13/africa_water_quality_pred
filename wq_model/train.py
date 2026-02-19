"""Training and evaluation utilities for water quality regression"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def run_epoch(model, loader, device, optimizer=None, criterion=None):
    """
    Run one epoch of training or evaluation
    
    Args:
        model: The neural network model
        loader: DataLoader for the data
        device: Device to run on (cuda/cpu)
        optimizer: Optimizer (None for evaluation)
        criterion: Loss function
        
    Returns:
        Average loss for the epoch
    """
    if criterion is None:
        criterion = nn.SmoothL1Loss()
        
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n = 0
    
    for x_img, x_tab, y in tqdm(loader, desc="Batch", leave=False):
        x_img = x_img.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.set_grad_enabled(is_train):
            out = model(x_img, x_tab)
            loss = criterion(out, y)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item() * len(y)
        n += len(y)
    
    return total_loss / n if n > 0 else 0.0


def predict(model, loader, device):
    """
    Get predictions from a model on a data loader
    
    Args:
        model: The neural network model
        loader: DataLoader for the data
        device: Device to run on (cuda/cpu)
        
    Returns:
        Tuple of (predictions, true_values) as numpy arrays
    """
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for x_img, x_tab, y in loader:
            x_img = x_img.to(device, non_blocking=True)
            x_tab = x_tab.to(device, non_blocking=True)
            out = model(x_img, x_tab).cpu().numpy()
            preds.append(out)
            trues.append(y.numpy())
    
    return np.vstack(preds), np.vstack(trues)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    max_epochs,
    patience,
    lr,
    weight_decay
):
    """
    Train model with early stopping based on validation set performance.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for held-out test data (evaluated once at the end)
        device: Device to run on (cuda/cpu)
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        
    Returns:
        Dictionary with training history, best model state, and final test loss
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer=optimizer, criterion=criterion)
        val_loss = run_epoch(model, val_loader, device, optimizer=None, criterion=criterion)
        
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch:02d}/{max_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs (val loss not improving)")
                break

    # Evaluate test loss exactly once using the best validation checkpoint
    best_test_loss = float("nan")
    if best_state is not None:
        model.load_state_dict(best_state)
        if test_loader is not None:
            best_test_loss = run_epoch(model, test_loader, device, optimizer=None, criterion=criterion)
    
    return {
        "history": history,
        "best_state": best_state,
        "best_test_loss": best_test_loss
    }
