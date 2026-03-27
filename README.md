# MHRQI: Multi-scale Hierarchical Representation of Quantum Images

[![CI](https://github.com/Keno-00/MHRQI/actions/workflows/ci.yml/badge.svg)](https://github.com/Keno-00/MHRQI/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

MHRQI is a Python-based framework for representing images using hierarchical quantum circuits. It provides a multi-scale quad-tree decomposition to improve the robustness and efficiency of quantum image representations.

---

## 🚀 Key Features

- **MHRQI Core**: A robust Object-Oriented framework for hierarchical quantum image encoding.
- **MHRQIResult**: A powerful results object for automated decoding, reconstruction, and metric calculation.
- **Multi-scale Encoding**: Maps image data to quantum states using a hierarchical qudit structure.
- **Denoising Application**: A functional extension for quantum denoising based on sibling smoothing and hierarchical consistency.
- **Standardized Benchmarks**: Built-in `BenchmarkSuite` for comparison against classical denoisers (BM3D, NL-Means, SRAD).

- **Medical Specialization**: optimized and tested on medical imaging datasets (e.g., OCT scans).

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Keno-00/MHRQI.git
cd MHRQI

# Install dependencies and package in editable mode
pip install -e .
```

## 💻 Usage

### Command Line Interface
MHRQI provides a comprehensive CLI for running the denoising pipeline.

```bash
# Run denoising with default settings
mhrqi --denoise

# Specify image size and number of shots
mhrqi --size 16 --shots 100000 --denoise

# Use a specific input image
mhrqi --img resources/drusen1.jpeg --denoise
```

Use `mhrqi --help` to see all available options.

### Python API (Library Usage)
You can use MHRQI as a dependency in your own projects (e.g., a "Quantum Vision Suite").

```python
import cv2
import numpy as np
from mhrqi import MHRQI
from mhrqi.utils import general as utils

# 1. Prepare image
img = cv2.imread("image.jpg", cv2.GRAYSCALE)
img_resized = cv2.resize(img, (16, 16))
img_normalized = img_resized.astype(np.float64) / 255.0

# 2. Initialize MHRQI model
depth = 4 # for 16x16 image (log2(16))
model = MHRQI(depth=depth)

# 3. Generate hierarchical coordinate matrix and upload data
hierarchical_coord_matrix = utils.generate_hierarchical_coord_matrix(16, 2)
model.upload(hierarchical_coord_matrix, img_normalized)

# 4. Apply denoising (optional)
model.apply_denoising()

# 5. Simulate
result = model.simulate(shots=10000)

# 6. Reconstruct image using the powerful Result object
img_recon = result.reconstruct()

# 7. Compute quality metrics (MSE, SSIM, NIQE, etc.)
metrics = result.compute_metrics(reference_image=img_normalized)
print(f"Metrics: {metrics}")

# 8. Plot the result
result.plot(title="Denoised OCT Scan")
```

## 📂 Project Structure

```
MHRQI/
├── mhrqi/                  # Main package
│   ├── core/               # Quantum circuit implementations
│   ├── utils/              # General & visualization utilities
│   ├── benchmarks/         # Scaling and comparison suites
│   └── cli/                # Command-line interface logic
├── docs/                   # Documentation source (MkDocs)
├── tests/                  # Unit and integration tests
├── pyproject.toml          # Project metadata and configuration
└── .github/                # CI/CD workflows and templates
```

## 📚 Documentation

For detailed guides and API documentation, see the `docs/` folder or visit the documentation site:

```bash
pip install -e .[docs]
mkdocs serve
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and development workflows.

## 📝 Citation

If you use this work in your research, please cite:

```bitex
@software{MHRQI_2026,
  author = {Keno S. Jose},
  title = {Multi-scale Hierarchical Representation of Quantum Images (MHRQI)},
  url = {https://github.com/Keno-00/MHRQI},
  version = {0.1.0},
  year = {2026}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.
