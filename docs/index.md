# MHRQI: Multi-scale Hierarchical Representation of Quantum Images

MHRQI is a Python-based framework for representing and denoising medical images using hierarchical quantum circuits.

## Key Features

- **Hierarchical Encoding**: Efficiently maps image data to quantum states using a hierarchical structure.
- **Quantum Denoising**: Implements a novel quantum denoising algorithm based on sibling smoothing and hierarchical consistency.
- **Standardized Benchmarks**: Built-in comparison suite against classical state-of-the-art denoisers (BM3D, NL-Means, SRAD).
- **Medical Specialization**: Optimized for medical imaging data like OCT scans.

## Quick Start

### Installation

```bash
pip install .
```

### Basic Usage

Run the denoising pipeline with default settings:

```bash
mhrqi --denoise
```

### Options

- `-n`, `--size`: Set image size (e.g., `-n 64`).
- `--denoise`: Enable the quantum denoising circuit.
- `--statevector`: Use exact statevector simulation.
- `--img`: Path to a specific input image.
