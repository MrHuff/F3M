# F³M (Faster - Fast and Free Memory Method)

## Requirements

To run anything with **F³M**, you will need:

1. A version of **LibTorch ≥ 2.0**, compiled with **CUDA 10.2** or later
2. A **NVIDIA `nvcc` compiler** that matches the CUDA version used in your LibTorch installation

## Installation

To install the Python bindings, simply run:

```bash
uv pip install F3M_src/. --no-build-isolation
```

> **Note:** You may need to modify the `CMakeLists.txt` file in `F3M_src/` to point to the correct LibTorch path.

## Running Experiments

### 3D Experiments

```bash
python experiments.py --idx={experiment_number}
```

### 4D/5D Experiments

```bash
python experiments678.py --idx={experiment_number}
```

## FALKON Experiments

1. Generate the dataset:
    ```bash
    python generate_KRR_data.py
    ```
2. Run the experiment:
    ```bash
    python experiments_2.py --idx={experiment_number}
    ```

### FALKON Experiments (Appendix)

```bash
python experiments_3.py --idx=1 --penalty_in=0 --seed=0
```