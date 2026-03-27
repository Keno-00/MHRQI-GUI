# User Guide

## Hierarchical Image Encoding

MHRQI uses a hierarchical qudit representation (Multi-scale Hierarchical Representation) to encode image intensity data. 

### The MHRQI Algorithm

1. **Subdivision**: The image is recursively subdivided into $d \times d$ quadrants.
2. **Hierarchy**: Each level of subdivision corresponds to a pair of qudits ($q_y, q_x$).
3. **Encoding**: Pixel intensities are mapped to rotation angles or basis states within the quantum circuit.

## Quantum Denoising

The denoising component utilizes hierarchical consistency checks. If a pixel's intensity differs significantly from its parent region's average (computed in superposition), it is marked on an outcome qubit.

### Usage

```bash
mhrqi --denoise --shots 100000 --size 16
```
