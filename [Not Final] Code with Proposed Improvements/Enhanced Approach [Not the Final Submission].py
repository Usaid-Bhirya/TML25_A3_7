"""
We formalized this code by adding some imporvements to our submitted code.
The following changes were added: 
    - Trades Loss
    - Improved Training
    - Multi-epsilon Training.

However, due to limited computational resources. We were not fully able to analyse
the performance of this enhanced approach.

We tried running this on Kaggle and each epoch was taking around 30 minutes. 

We expect this code could out-perform our submitted code. 

HOWEVER, THIS IS NOT OUR FINAL SUBMITTED CODE :D
"""




import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import numpy as np
import random
import requests
from collections import defaultdict
import json
from tqdm import tqdm
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TaskDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # load raw object; could be dict, list, or saved TaskDataset
        data = torch.load(data_path, weights_only=False)

        # If it's already a TaskDataset instance, reuse its attributes
        if isinstance(data, TaskDataset):
            self.ids = data.ids
            self.imgs = data.imgs
            self.labels = data.labels

        # If it's a dict with keys 'ids','imgs','labels', unpack it:
        elif isinstance(data, dict) and all(k in data for k in ('ids','imgs','labels')):
            self.ids = list(data['ids'])
            self.imgs = list(data['imgs'])
            self.labels = list(data['labels'])

        # If it's a list of triplets, unzip them:
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) == 3:
            self.ids, self.imgs, self.labels = map(list, zip(*data))

        else:
            raise RuntimeError(f"Unrecognized Train.pt format: got {type(data)}")

        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


def fgsm_attack(model, loss_fn, images, labels, epsilon):
    """FGSM attack implementation"""
    images = images.clone().detach().requires_grad_(True)
    
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    
    # Zero gradients
    model.zero_grad()
    
    # Calculate gradients
    loss.backward()
    
    # Create adversarial examples
    grad_sign = images.grad.data.sign()
    adv_images = images + epsilon * grad_sign
    
    return torch.clamp(adv_images, 0, 1).detach()


def pgd_attack(model, loss_fn, images, labels, epsilon, alpha, iters):
    """PGD attack implementation"""
    orig_images = images.clone().detach()
    
    # Start with random noise
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(delta, 0-images, 1-images)
    
    for _ in range(iters):
        delta.requires_grad = True
        
        # Forward pass
        outputs = model(orig_images + delta)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update delta
        grad = delta.grad.detach()
        delta = delta + alpha * grad.sign()
        
        # Project back to epsilon ball
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = torch.clamp(delta, 0-orig_images, 1-orig_images)
        delta = delta.detach()
    
    return (orig_images + delta).detach()


def trades_loss(model, x_natural, y, optimizer, step_size, epsilon, perturb_steps, beta):
    """TRADES loss for adversarial training"""
    model.eval()
    batch_size = len(x_natural)
    
    # Generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                               F.softmax(model(x_natural), dim=1),
                               reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    
    x_adv = x_adv.detach()
    
    # Calculate loss
    optimizer.zero_grad()
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                                 F.softmax(model(x_natural), dim=1),
                                                 reduction='batchmean')
    loss = loss_natural + beta * loss_robust
    
    return loss, loss_natural, loss_robust


def train_epoch_improved(model, device, train_loader, optimizer, epoch, total_epochs, 
                        epsilon, alpha, pgd_iters, use_trades=True):
    """Improved training with TRADES and multi-epsilon training"""
    model.train()
    total_loss = 0
    total_natural_loss = 0
    total_robust_loss = 0
    correct = 0
    total_samples = 0
    
    # TRADES parameters
    beta = 8.0  # Increased weight for robust loss
    step_size = 2/255
    
    # Multi-epsilon training - use different epsilons
    epsilon_list = [4/255, 6/255, 8/255, 10/255, 12/255, 14/255]
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
    
    for batch_idx, (_, imgs, labels) in enumerate(train_bar):
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.size(0)
        
        if use_trades and epoch > 5:  # Start TRADES after a few epochs
            # TRADES loss
            loss, loss_natural, loss_robust = trades_loss(
                model=model,
                x_natural=imgs,
                y=labels,
                optimizer=optimizer,
                step_size=step_size,
                epsilon=random.choice(epsilon_list),  # Random epsilon
                perturb_steps=10,
                beta=beta
            )
            
            total_natural_loss += loss_natural.item()
            total_robust_loss += loss_robust.item()
        else:
            # Standard adversarial training with mixed epsilon
            optimizer.zero_grad()
            
            # Split batch for different training strategies
            # Ensure minimum batch size of 2 for each type to avoid batch norm issues
            min_size = 2
            clean_size = max(min_size, batch_size // 4)
            fgsm_size = max(min_size, batch_size // 4)
            pgd_size = max(min_size, batch_size // 4)
            multi_eps_size = batch_size - clean_size - fgsm_size - pgd_size
            
            # Adjust sizes if batch is too small
            if batch_size < 8:
                clean_size = max(2, batch_size // 2)
                fgsm_size = batch_size - clean_size
                pgd_size = 0
                multi_eps_size = 0
            
            all_imgs = []
            all_labels = []
            
            # Clean examples
            if clean_size > 0:
                all_imgs.append(imgs[:clean_size])
                all_labels.append(labels[:clean_size])
            
            # FGSM examples
            if fgsm_size > 0:
                fgsm_eps = random.choice(epsilon_list)
                fgsm_imgs = fgsm_attack(model, nn.CrossEntropyLoss(), 
                                       imgs[clean_size:clean_size+fgsm_size], 
                                       labels[clean_size:clean_size+fgsm_size], 
                                       fgsm_eps)
                all_imgs.append(fgsm_imgs)
                all_labels.append(labels[clean_size:clean_size+fgsm_size])
            
            # PGD examples
            if pgd_size > 0:
                pgd_eps = random.choice(epsilon_list)
                pgd_imgs = pgd_attack(model, nn.CrossEntropyLoss(),
                                     imgs[clean_size+fgsm_size:clean_size+fgsm_size+pgd_size],
                                     labels[clean_size+fgsm_size:clean_size+fgsm_size+pgd_size],
                                     pgd_eps, alpha, pgd_iters)
                all_imgs.append(pgd_imgs)
                all_labels.append(labels[clean_size+fgsm_size:clean_size+fgsm_size+pgd_size])
            
            # Multi-epsilon PGD examples (process as a batch to avoid batch norm issues)
            if multi_eps_size > 0:
                start_idx = clean_size + fgsm_size + pgd_size
                multi_eps_imgs = imgs[start_idx:]
                multi_eps_labels = labels[start_idx:]
                
                # Use a different random epsilon for robustness
                eps = random.choice(epsilon_list)
                multi_pgd_imgs = pgd_attack(model, nn.CrossEntropyLoss(),
                                           multi_eps_imgs,
                                           multi_eps_labels,
                                           eps, alpha, random.randint(7, 15))
                all_imgs.append(multi_pgd_imgs)
                all_labels.append(multi_eps_labels)
            
            # Combine all examples
            combined_imgs = torch.cat(all_imgs, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            
            # Forward pass
            outputs = model(combined_imgs)
            loss = nn.CrossEntropyLoss()(outputs, combined_labels)
            
            # Add L2 regularization
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss = loss + 1e-4 * l2_reg
            
            loss_natural = loss
            loss_robust = torch.tensor(0)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        with torch.no_grad():
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
        
        # Update progress bar
        train_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'nat_loss': f'{loss_natural.item():.4f}',
            'rob_loss': f'{loss_robust.item():.4f}',
            'acc': f'{correct/total_samples:.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total_samples
    return avg_loss, accuracy


def evaluate_model(model, device, data_loader, loss_fn, epsilon, alpha, pgd_iters):
    """Evaluate model on clean, FGSM, and PGD examples"""
    model.eval()
    
    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0
    
    eval_bar = tqdm(data_loader, desc="Evaluating")
    
    with torch.no_grad():
        for _, imgs, labels in eval_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)
            
            # Clean accuracy
            clean_outputs = model(imgs)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_correct += clean_preds.eq(labels).sum().item()
            
            total += batch_size
            
            eval_bar.set_postfix({
                'clean': f'{clean_correct/total:.4f}',
                'total': total
            })
    
    # Reset for adversarial evaluation
    model.eval()
    fgsm_correct = 0
    pgd_correct = 0
    total = 0
    
    eval_bar = tqdm(data_loader, desc="Evaluating Adversarial")
    
    for _, imgs, labels in eval_bar:
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.size(0)
        
        # FGSM attack
        fgsm_imgs = fgsm_attack(model, loss_fn, imgs, labels, epsilon)
        with torch.no_grad():
            fgsm_outputs = model(fgsm_imgs)
            fgsm_preds = fgsm_outputs.argmax(dim=1)
            fgsm_correct += fgsm_preds.eq(labels).sum().item()
        
        # PGD attack
        pgd_imgs = pgd_attack(model, loss_fn, imgs, labels, epsilon, alpha, pgd_iters)
        with torch.no_grad():
            pgd_outputs = model(pgd_imgs)
            pgd_preds = pgd_outputs.argmax(dim=1)
            pgd_correct += pgd_preds.eq(labels).sum().item()
        
        total += batch_size
        
        eval_bar.set_postfix({
            'fgsm': f'{fgsm_correct/total:.4f}',
            'pgd': f'{pgd_correct/total:.4f}',
            'total': total
        })
    
    clean_acc = clean_correct / len(data_loader.dataset)
    fgsm_acc = fgsm_correct / total
    pgd_acc = pgd_correct / total
    
    return clean_acc, fgsm_acc, pgd_acc


def submit_model(token, model_name, model_path):
    """Submit model to evaluation server"""
    try:
        response = requests.post(
            "http://34.122.51.94:9090/robustness",
            files={"file": open(model_path, "rb")},
            headers={"token": token, "model-name": model_name}
        )
        print("Submission response:", response.json())
        return response.json()
    except Exception as e:
        print(f"Submission failed: {e}")
        return None


if __name__ == '__main__':
    # Configuration
    data_path = '/kaggle/input/traindata/Train.pt'
    model_name = 'resnet50'  # Using ResNet-50 for better capacity
    batch_size = 64  # Smaller batch size to ensure we have enough samples per type
    epochs = 150  # More epochs for better convergence
    lr = 0.1  # Higher initial learning rate
    epsilon = 8/255
    alpha = 2/255
    pgd_iters = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Epsilon: {epsilon:.4f}, Alpha: {alpha:.4f}, PGD iters: {pgd_iters}")
    
    # Data augmentation - stronger augmentation for robustness
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'convert') else img),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'convert') else img),
        transforms.ToTensor(),
    ])

    def collate_fn(batch):
        ids, imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        labels = torch.tensor(labels)
        return list(ids), imgs, labels

    # Create train/validation split
    full_dataset = TaskDataset(data_path, transform=train_transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create separate dataset for validation with different transform
    val_dataset_clean = TaskDataset(data_path, transform=val_transform)
    val_indices = val_dataset.indices
    val_subset = torch.utils.data.Subset(val_dataset_clean, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Model setup
    model = getattr(models, model_name)(weights='DEFAULT')
    
    # Replace final layer
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, 10)
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.constant_(model.fc.bias, 0)
    
    model = model.to(device)

    # Optimizer with higher initial learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate schedule - cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    loss_fn = nn.CrossEntropyLoss()

    # Training setup
    best_robust_score = 0.0
    best_clean_acc = 0.0
    best_fgsm_acc = 0.0
    best_pgd_acc = 0.0
    best_epoch = 0
    os.makedirs('out/models', exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_clean_acc': [],
        'val_fgsm_acc': [],
        'val_pgd_acc': [],
        'robust_score': []
    }

    print("Starting improved robust training...")
    
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        
        # Training
        use_trades = epoch > 5  # Start TRADES after epoch 5
        train_loss, train_acc = train_epoch_improved(
            model, device, train_loader, optimizer, epoch, epochs,
            epsilon, alpha, pgd_iters, use_trades=use_trades
        )
        
        scheduler.step()
        
        # Log training progress
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Evaluate on validation set
        if epoch % 5 == 0 or epoch == epochs:
            print("Evaluating on validation set...")
            clean_acc, fgsm_acc, pgd_acc = evaluate_model(
                model, device, val_loader, loss_fn, epsilon, alpha, pgd_iters
            )
            
            # Score that prioritizes FGSM and PGD accuracy as requested
            robust_score = 0.7 * fgsm_acc + 0.3 * pgd_acc
            
            history['val_clean_acc'].append(clean_acc)
            history['val_fgsm_acc'].append(fgsm_acc)
            history['val_pgd_acc'].append(pgd_acc)
            history['robust_score'].append(robust_score)
            
            print(f"Val Clean Acc: {clean_acc:.4f}")
            print(f"Val FGSM Acc:  {fgsm_acc:.4f}")
            print(f"Val PGD Acc:   {pgd_acc:.4f}")
            print(f"Robust Score:  {robust_score:.4f}")
            
            # Save best model based on criteria: high FGSM/PGD accuracy while maintaining decent clean accuracy
            if robust_score > best_robust_score and clean_acc > 0.5:
                best_robust_score = robust_score
                best_clean_acc = clean_acc
                best_fgsm_acc = fgsm_acc
                best_pgd_acc = pgd_acc
                best_epoch = epoch
                torch.save(model.state_dict(), f"out/models/{model_name}_best.pt")
                print(f"🎯 New best model saved! Robust Score: {robust_score:.4f}")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    clean_acc, fgsm_acc, pgd_acc = evaluate_model(
        model, device, val_loader, loss_fn, epsilon, alpha, pgd_iters
    )
    final_robust_score = 0.7 * fgsm_acc + 0.3 * pgd_acc
    
    print(f"Final Clean Accuracy: {clean_acc:.4f}")
    print(f"Final FGSM Accuracy:  {fgsm_acc:.4f}")
    print(f"Final PGD Accuracy:   {pgd_acc:.4f}")
    print(f"Final Robust Score:   {final_robust_score:.4f}")
    
    print(f"\nBest Model (Epoch {best_epoch}):")
    print(f"Clean Accuracy: {best_clean_acc:.4f}")
    print(f"FGSM Accuracy:  {best_fgsm_acc:.4f}")
    print(f"PGD Accuracy:   {best_pgd_acc:.4f}")
    print(f"Robust Score:   {best_robust_score:.4f}")
    
    # Save final model and history
    torch.save(model.state_dict(), f"out/models/{model_name}_final.pt")
    with open(f"out/models/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best model saved as: {model_name}_best.pt")
    print("Submit this model for evaluation.")