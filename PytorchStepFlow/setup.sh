#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="step-perf"
STEP_TL_DIR="$SCRIPT_DIR/step_tl"

echo "=== PytorchStepFlow Environment Setup ==="

# ── Step 1: Install Miniconda ──
if [ -x "$MINICONDA_DIR/bin/conda" ]; then
    echo "[skip] Miniconda already installed at $MINICONDA_DIR"
else
    echo "[install] Downloading and installing Miniconda..."
    INSTALLER="/tmp/miniconda_installer.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$MINICONDA_DIR"
    rm -f "$INSTALLER"
    echo "[done] Miniconda installed to $MINICONDA_DIR"
fi

# Make conda available in this script
eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"

# ── Step 2: Create conda environment ──
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "[skip] Conda env '$CONDA_ENV_NAME' already exists"
else
    echo "[create] Creating conda env '$CONDA_ENV_NAME' from step_tl/environment.yml..."
    conda env create -f "$STEP_TL_DIR/environment.yml" -n "$CONDA_ENV_NAME"
    echo "[done] Conda env '$CONDA_ENV_NAME' created"
fi

conda activate "$CONDA_ENV_NAME"
# Unset VIRTUAL_ENV to avoid conflict with CONDA_PREFIX (maturin refuses both)
unset VIRTUAL_ENV
echo "[info] Active env: $CONDA_DEFAULT_ENV (Python: $(python --version))"

# ── Step 2b: Install PyTorch CPU ──
if python -c "import torch" 2>/dev/null; then
    echo "[skip] PyTorch already installed"
else
    echo "[install] Installing PyTorch CPU..."
    pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    echo "[done] PyTorch CPU installed"
fi

# ── Step 3: Compile protobufs ──
PROTO_SRC_DIR="$STEP_TL_DIR/step_perf_ir/proto"
PROTO_OUT_DIR="$STEP_TL_DIR/src/proto"

if [ -f "$PROTO_OUT_DIR/graph_pb2.py" ] && \
   [ -f "$PROTO_OUT_DIR/ops_pb2.py" ] && \
   [ -f "$PROTO_OUT_DIR/datatype_pb2.py" ] && \
   [ -f "$PROTO_OUT_DIR/func_pb2.py" ]; then
    echo "[skip] Protobuf Python files already exist in $PROTO_OUT_DIR"
else
    echo "[build] Compiling .proto files..."
    mkdir -p "$PROTO_OUT_DIR"
    protoc --experimental_allow_proto3_optional \
        --proto_path="$PROTO_SRC_DIR" --python_out="$PROTO_OUT_DIR" \
        "$PROTO_SRC_DIR/graph.proto" \
        "$PROTO_SRC_DIR/ops.proto" \
        "$PROTO_SRC_DIR/datatype.proto" \
        "$PROTO_SRC_DIR/func.proto"
    touch "$PROTO_OUT_DIR/__init__.py"
    echo "[done] Protobuf files compiled to $PROTO_OUT_DIR"
fi

# ── Step 4: Build step-perf (Rust simulator) ──
if python -c "import step_perf" 2>/dev/null; then
    echo "[skip] step_perf already importable"
else
    echo "[build] Building step-perf..."
    pushd "$STEP_TL_DIR/step-perf" > /dev/null
    cargo build
    maturin develop
    popd > /dev/null
    echo "[done] step-perf built and installed"
fi

# ── Step 5: Build step_tl Rust extension ──
if python -c "import step_tl" 2>/dev/null; then
    echo "[skip] step_tl Rust extension already importable"
else
    echo "[build] Building step_tl Rust extension..."
    pushd "$STEP_TL_DIR" > /dev/null
    maturin develop
    popd > /dev/null
    echo "[done] step_tl Rust extension built and installed"
fi

# ── Step 6: Install agent packages ──
if pip show openai-agents > /dev/null 2>&1 && pip show openai > /dev/null 2>&1; then
    echo "[skip] Agent packages already installed"
else
    echo "[install] Installing agent packages from requirements.txt..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
    echo "[done] Agent packages installed"
fi

# ── Done ──
echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment:"
echo "  conda activate $CONDA_ENV_NAME"
echo "  export PYTHONPATH=$STEP_TL_DIR/src:$STEP_TL_DIR/src/step_py:$STEP_TL_DIR/src/sim:$STEP_TL_DIR/src/proto:$SCRIPT_DIR"
echo ""
echo "Or source this one-liner:"
echo "  conda activate $CONDA_ENV_NAME && export PYTHONPATH=$STEP_TL_DIR/src:$STEP_TL_DIR/src/step_py:$STEP_TL_DIR/src/sim:$STEP_TL_DIR/src/proto:$SCRIPT_DIR"
