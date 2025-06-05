# Explainable Trajectory Corrections from Language Inputs

This repository contains a quick implementation of ExTraCT

## 📄 Paper

**Title**: Explainable Trajectory Corrections from Language Inputs using Textual Description of Features  
**Link**: [https://arxiv.org/abs/2401.03701](https://arxiv.org/abs/2401.03701)

---

## 🚀 Installation Instructions

This codebase uses `conda` for environment management.

### 1. Create Environment

```bash
conda env create -f environment.yaml
```

### 2. Activate Environment
```bash
conda activate extract
```

### 3. Install in Editable Mode
```bash
pip install -e .
```

## Usage
### Run on All Trajectories

```bash
python scripts/collect_results_extract.py
```

### Run on a Single Trajectory File
```bash
python -m extract.extract --trajectory_path data/latte_0.json
```
### Run with a Custom Instruction
```bash
python -m extract.extract --trajectory_path data/latte_0.json --instruction "<your instruction>"
```



