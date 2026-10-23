
from huggingface_hub import snapshot_download

model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

path = snapshot_download(
    repo_id=model_id,
    local_dir="./models/smollm2-360m"
)

print("Model downloaded to:", path)
