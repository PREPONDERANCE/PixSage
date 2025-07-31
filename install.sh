pip install uv

uv venv

source .venv/bin/activate

uv pip install torch==2.1.0
if python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    uv pip install flash-attn==2.3.6 --no-build-isolation
else
    echo "No GPU detected. Skipping flash-attn installation."
fi

uv sync

if [[ "$OSTYPE" == "darwin"* ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SHELL_RC="$HOME/.bashrc"
elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
    SHELL_RC="$HOME/.bash_profile"
else
    SHELL_RC="$HOME/.profile"
fi

echo 'export HF_ENDPOINT=https://hf-mirror.com' >> "$SHELL_RC"