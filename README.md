# Aerodynamic Coefficient Prediction via Multi-Input CNN

A surrogate deep learning model utilizing a multi-input Convolutional Neural Network (CNN) to predict NACA 4-digit airfoil lift ($C_L$) and drag ($C_D$) coefficients directly from geometry images and flow operating conditions.

## Overview
Evaluating aerodynamic performance via Computational Fluid Dynamics (CFD) or wind tunnel testing is computationally expensive during early-stage conceptual aircraft design. This project implements a surrogate neural network capable of predicting aerodynamic forces in milliseconds. 

The architecture accepts two distinct modalities: a 2D spatial rasterization of the airfoil geometry ($64 \times 64$ binary image) and continuous flow variables (Angle of Attack and Reynolds Number).

## Key Features
* **First-Principles Data Generation**: Generates 1,500 NACA 4-digit parametric geometries (varying camber $m$, camber position $p$, and maximum thickness $t$) combined with analytical thin-foil theory and skin friction approximations.
* **Spatial Rasterization**: Converts boundary coordinates $(x, y)$ into a $64 \times 64$ binary occupancy grid using matplotlib path algorithms.
* **Dual-Branch Architecture**: Features a 2D CNN branch to extract geometric feature maps, fused with a dense branch encoding flow conditions ($\alpha$ and $Re$).
* **Surrogate Performance**: Reaches high-fidelity evaluation accuracy ($R^2 = 0.972$ for $C_L$, $R^2 = 0.854$ for $C_D$) on unseen test sets.

## Model Architecture

```
Geometry Input (64x64x1) --------> [ Conv2D(32) -> MaxPool ] 
                                 -> [ Conv2D(64) -> MaxPool ] 
                                 -> [ Conv2D(128) -> MaxPool ] -> Flatten (8192) --\
                                                                                   +--> [ Concatenate ] -> Dense(64) -> Dense(32) -> Output (CL, CD)
Operating Conditions (AoA, Re) -> [ Input Tensor (2) ] ---------------------------/
```

## Metrics & Performance
| Target Variable | $R^2$ Score | Mean Absolute Error (MAE) |
| :--- | :--- | :--- |
| **Lift Coefficient ($C_L$)** | 0.972 | 0.084 |
| **Drag Coefficient ($C_D$)** | 0.854 | 0.014 |

*Note: The lower drag accuracy reflects the known non-linear complexity of viscous skin friction and boundary layer interactions.*

## Tech Stack
* **Language**: Python 3.x
* **Framework**: TensorFlow / Keras
* **Geometric & Numerical Libraries**: NumPy, Pandas, PIL, Matplotlib
* **Evaluation**: Scikit-Learn

## Repository Structure
```
├── train_cnn.py          # Main script for dataset generation, model compilation, training, and testing
├── requirements.txt      # Dependencies
└── README.md
```

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Airfoil-CNN-Aerodynamics.git](https://github.com/YOUR_USERNAME/Airfoil-CNN-Aerodynamics.git)
   cd Airfoil-CNN-Aerodynamics
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib pillow tensorflow scikit-learn
   ```

3. **Run training & evaluation**:
   ```bash
   python train_cnn.py
   ```
