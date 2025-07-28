pip install uv

uv venv

source .venv/bin/activate

uv pip install torch==2.0.1

uv sync --no-build-isolation

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