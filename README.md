# Softplus-Power Activation

Softplus-Power is a **biologically inspired activation function** designed as a smooth, flexible alternative to ReLU, SiLU/Swish, and Softplus.  
It is motivated by real neuronal contrast-response functions and can emulate different neuron classes via parameter presets.

---

## 📌 Overview
 Most deep learning models rely on simple activation functions like **ReLU** or **SiLU**. While effective, these functions are far removed from the graded, saturating, and adaptive responses of real neurons in the brain.

 **Softplus-Power** is a  generalization of the classic **Softplus** activation:

 $$
 f(x) = \alpha \cdot \big( \ln(1 + e^{\beta(x - \theta)}) \big)^{p} - \delta
 $$

 where:
 - **α** = output gain  
 - **β** = slope (input scaling)  
 - **θ** = threshold shift  
 - **p** = curvature (exponent)  
 - **δ** = output offset  

👉 This parametrization allows Softplus-Power to smoothly interpolate between **ReLU-like**, **sigmoid-like**, and **biologically plausible saturating** responses.

---

## ✨ Features

- 🚀 **Cross-framework support**: PyTorch, Keras/TensorFlow, JAX, and NumPy reference implementations.  
- 🔒 **Calibrated variants**: guarantees α, β > 0 and p ≥ 0.5 for stable training and well-behaved gradients.  
- 🧪 **Numerically stable**: careful `log1p`, `exp`, and `abs` usage to avoid overflow/underflow.  
- 🧩 **Drop-in replacement**: works as a substitute for `ReLU`, `Softplus`, or `SiLU` in most models.  
- 📊 **Gradient-friendly**: fully differentiable and backprop-compatible across frameworks.  
- 📦 **MIT License**: free for academic and commercial use.  

### 🔬 Biologically Inspired Extensions
- 🧠 **Presets for neuron types**: built-in settings (`retinal_on`, `retinal_off`, `lgn_opponent`, `v1_simple`, `v1_complex`, `inhibitory`) inspired by retinal, LGN, and V1 cell responses.  
- 🎲 **Jitter option**: introduces controlled random variation in parameters to simulate biological heterogeneity and diversify unit responses.  
- 🔄 **Flexible parameterization**: smoothly interpolates between ReLU-like, Softplus-like, and Naka–Rushton-like behaviors by tuning α, β, θ, p, δ.  
- 📉 **Derivative-friendly**: exposes both activation and derivative (gain curve) for analysis and visualization.  
- 🖼️ **Visualization tools**: included plotting scripts to compare presets, derivatives, and jittered populations against standard activations.  
- 📖 **Educational value**: doubles as a teaching tool to illustrate how activation shapes influence learning and how they map to biological neurons.  

---

### Biological Presets
    Softplus-Power provides **presets** that approximate real neural response types:

 - `generic` – neutral baseline (default)  
 - `retinal_on` – early threshold, moderate slope  
 - `retinal_off` – mirrored ON response with offset  
 - `lgn_opponent` – steeper contrast gain control  
 - `v1_simple` – clear thresholding, steeper slope  
 - `v1_complex` – more compressive/saturating  
 - `inhibitory` – elevated threshold and offset (sparse response)  

---

## Jitter (Heterogeneity)

 - Real neurons are not identical. To mimic population diversity, Softplus-Power allows a jitter option that introduces small random variations in initialization:
 - Prevents identical unit responses.
 - Encourages diverse feature learning.
 - Mimics biological heterogeneity.
---

## 📦 Installation

 ```bash
 git clone https://github.com/daa2bg/softplus-power.git
 cd softplus-power
 pip install -e .
 ```

##  Examples
**PyTorch examples:**

 ```python
 from softplus_power import CalibratedSoftplusPowerTorch
 act = CalibratedSoftplusPowerTorch(preset="v1_simple")
 ```
 ```python
 from softplus_power import CalibratedSoftplusPowerTorch
 act = CalibratedSoftplusPowerTorch(preset="v1_simple", jitter=0.05)
 ```
**Keras examples:**

  ```python
 from softplus_power import CalibratedSoftplusPowerKeras
 act = CalibratedSoftplusPowerKeras(preset="retinal_on")
 ```
 ```python
 from softplus_power import CalibratedSoftplusPowerKeras
 act = CalibratedSoftplusPowerKeras(preset="lgn_opponent", jitter=0.1)
 ```
---

###  Visualization
 A plotting script is included to visualize presets, derivatives, and jitter effects:

 python scripts/plot_activations.py --preset v1_simple --jitter 0.1 --save figs



### 🔹 Citation

If you use this activation in academic work, you can cite:
@misc{softpluspower2025,
  title   = {Softplus-Power: A Biologically Inspired Activation Function},
  author  = {Daniel Angelov},
  year    = {2025},
  url     = {https://github.com/daa2bg/softplus-power.git}
}
