# TML25_A3_7 - Adversarial Robustness Assignment

## Overview
This project implements adversarial training for building robust deep learning classifiers that can withstand adversarial attacks while maintaining good clean accuracy. The goal is to train a model that performs well on both clean data and adversarial examples generated using Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).

## Final Results
- **Clean Accuracy**: 64.87%
- **FGSM Accuracy**: 39.93%
- **PGD Accuracy**: 0.33%

## Repository Structure
```
├── tml-task-3.ipynb          # Main training notebook
├── README.md                 # This file
├── report.pdf                # Detailed technical report
├── out/models/               # Directory for saved models
│   ├── resnet34_best.pt      # Best model checkpoint
│   ├── resnet34_final.pt     # Final model checkpoint
└── └── training_history.json # Training metrics history
```

## Key Files and Code Components

### Main Training Script: `tml-task-3.ipynb`
The notebook contains the complete adversarial training pipeline with the following key components:

#### 1. Data Loading and Preprocessing (`TaskDataset` class)
- Flexible dataset loader that handles different data formats
- Conservative data augmentation to maintain clean accuracy
- Train/validation split (90/10)

#### 2. Adversarial Attack Implementations
- **FGSM Attack** (`fgsm_attack` function): Single-step gradient-based attack
- **PGD Attack** (`pgd_attack` function): Multi-step iterative attack with projection

#### 3. Conservative Training Strategy (`train_epoch` function)
- **Curriculum Learning**: Gradual introduction of adversarial examples
  - Epochs 1-20: Clean data only (establish baseline)
  - Epochs 21-50: Gradually introduce weak adversarial examples
  - Epochs 51-80: Increase adversarial strength
  - Epochs 81-100: Full strength with 75% clean data ratio
- **Mixed Training**: Combines clean and adversarial examples
- **Gradient Clipping**: Prevents gradient explosion

#### 4. Model Architecture
- **Base Model**: ResNet-34 with ImageNet pre-trained weights
- **Output Layer**: Modified for 10-class classification
- **Initialization**: Xavier uniform initialization for stability

#### 5. Training Configuration
- **Optimizer**: SGD with momentum (0.9) and weight decay (1e-4)
- **Learning Rate**: 0.01 with MultiStepLR scheduler
- **Batch Size**: 128
- **Epochs**: 100
- **Attack Parameters**: ε=8/255, α=2/255, PGD iterations=10

#### 6. Evaluation (`evaluate_model` function)
- Comprehensive evaluation on clean, FGSM, and PGD examples
- Combined scoring with heavy weighting toward clean accuracy (70% clean, 20% FGSM, 10% PGD)

## Running the Code

### Prerequisites
```bash
pip install torch torchvision numpy tqdm requests
```

### Training
1. Place the training data (`Train.pt`) in the appropriate directory
2. Run the Jupyter notebook `tml-task-3.ipynb`
3. The best model will be saved as `out/models/resnet34_best.pt`

### Model Submission
The code includes a `submit_model` function for automatic submission to the evaluation server.

## Key Design Decisions

### 1. Conservative Approach
- Heavy emphasis on maintaining clean accuracy (75% clean data even in final epochs)
- Gradual introduction of adversarial examples
- More conservative data augmentation

### 2. Curriculum Learning
- Start with clean data only to establish a strong baseline
- Gradually introduce adversarial examples with increasing strength
- This prevents the model from overfitting to adversarial examples early

### 3. Mixed Training Strategy
- Combines clean and adversarial examples in each batch
- Uses both FGSM and PGD attacks for diverse adversarial training
- Reduced PGD iterations to prevent overly aggressive training

### 4. Stability Measures
- Gradient clipping to prevent gradient explosion
- Xavier initialization for better convergence
- Conservative learning rate scheduling

## Model Performance Analysis

The trained model achieved:
- **Clean Accuracy**: 64.87% - Good baseline performance
- **FGSM Accuracy**: 39.93% - Moderate robustness against single-step attacks
- **PGD Accuracy**: 0.33% - Limited robustness against strong multi-step attacks

The results demonstrate the classic robustness-accuracy trade-off in adversarial training. The model maintains reasonable clean accuracy while gaining some robustness against weaker attacks, but struggles with stronger PGD attacks.

## Future Improvements

1. **Advanced Training Techniques**:
   - Adversarial Weight Perturbation (AWP)
   - TRADES loss function
   - Certified adversarial training

2. **Architecture Modifications**:
   - Wider ResNet architectures
   - Ensemble methods
   - Attention mechanisms

3. **Training Strategies**:
   - Progressive adversarial training
   - Adversarial training with unlabeled data
   - Multi-step curriculum learning

## Contact
For questions or issues, please refer to the detailed technical report (`report.pdf`) or contact the team members.