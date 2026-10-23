
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/smollm2-360m"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto"
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"Model loaded on: {device}")
print("Type 'exit' to stop.")

while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    messages = [{"role": "user", "content": question}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\nSmolLM2:", answer)
