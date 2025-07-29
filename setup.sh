uv venv --python 3.11.5
source .venv/bin/activate
uv pip install numpy scipy ipykernel pandas scikit-learn
uv pip install torch
uv pip install git+https://github.com/huggingface/transformers.git
uv pip install matplotlib seaborn accelerate sentencepiece evaluate einops rouge-score gputil bitsandbytes