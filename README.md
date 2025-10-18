# **Blade Cascade Flow Solver (3D Vortex Panel Method)**

A Python implementation of the **two-dimensional potential-flow solver** for **axial and ducted fan cascades**, based on the *vortex panel and element method* formulations described in:

> **Lewis, R.I. (1991). _Vortex Element Methods for Fluid Dynamic Analysis of Engineering Systems_. Cambridge University Press.**

This solver reproduces the *inviscid, irrotational flow solution* for a periodic cascade of airfoils (rotor or stator rows) using **linear vortex panel methods**, coupled with cascade influence kernels and wake alignment through iterative correction of the trailing-edge flow angle.

---

## 🧩 **Overview**

- **Purpose:**  
  To compute aerodynamic performance (lift, circulation, flow angles, pressure distribution) of blade cascades under potential-flow assumptions.

- **Scope:**  
  Implements the **inviscid vortex element method** up to **Chapter 7** of *Lewis (1991)* — i.e. without the viscous or boundary-layer corrections introduced in later chapters.

- **Applications:**  
  - Preliminary design of fan and compressor blade rows  
  - Low-order modeling for CFD validation  
  - Fast iterations of designs

---

## ⚙️ **Methodology**

### 1. **Geometry and Discretization**
Each airfoil section is defined by a set of coordinates (from **TE → LE → TE**) and resampled using a cosine spacing with finite trailing-edge correction.  
The surface is divided into *m* linear panels, each carrying a **bound vortex element** of unknown strength γₙ.

### 2. **Boundary Conditions**
The impermeability condition is enforced at the control point of each panel:

$U_\infty \cos(\phi_i) + (V_\infty - \Omega r)\sin(\phi_i) + \sum_{j=1}^{m} (L_{ij}C_{ij} \gamma_j) = 0$

where $\(L_{ij}C_{ij}\)$ is the influence coefficient from panel *j* to control point *i*, assembled via **Martensen’s periodic kernel** for cascades.

### 3. **Coupling Matrices**
Two integral operators are used:
- $\(C\)$ — the **bound vortex coupling** (panel influence)
- $\(L\)$ — the **wake influence** (trailing vortex sheet coupling)  
constructed using chordwise integration operators \(Cds\) and \(Lds\) that distinguish **upper** and **lower** surface influences.

### 4. **Wake Iteration**
The **trailing-edge wake direction** is iteratively corrected until the exit flow angle \(\beta_{\text{exit}}\) matches the wake direction \(\beta_{\text{wake}}\) within a given tolerance.  
This ensures a physically consistent, periodic downstream flow.

### 5. **Performance Calculation**
After solving for the circulation distribution $\(\gamma(s)\)$, the solver computes:
- Surface velocity and pressure coefficient $\(C_p\)$
- Sectional lift and moment
- Flow exit angles and turning
- Total blade row forces, torque, and power (for rotating cascades)

---

## 📚 **Reference**

> **Lewis, R.I. (1991).**  
> *Vortex Element Methods for Fluid Dynamic Analysis of Engineering Systems.*  
> Cambridge University Press, Chapters 1–7.

---

## 🧠 **Implemented Equations**

- Linear vortex element panel method (Ch. 3–5)  
- Cascade influence matrices using Martensen kernels (Ch. 6)  
- Periodic wake treatment and coupling coefficients $\(K_{mn}\)$, $\(L_{mn}\)$ (Ch. 7)  
- Iterative wake alignment for cascade consistency  

*(Viscous layer, separation, and loss models from later chapters are not yet included.)*

