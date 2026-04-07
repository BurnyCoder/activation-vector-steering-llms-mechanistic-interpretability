from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import SteeringVector, train_steering_vector

MODEL_NAME = "gpt2"
VECTOR_PATH = Path("steering_vector.pt")


def load_steering_vector(path: Path) -> SteeringVector:
    payload = torch.load(path, map_location="cpu")
    return SteeringVector(
        layer_activations=payload["layer_activations"],
        layer_type=payload["layer_type"],
    )


def save_steering_vector(path: Path, steering_vector: SteeringVector) -> None:
    torch.save(
        {
            "layer_type": steering_vector.layer_type,
            "layer_activations": {
                layer: activation.detach().cpu()
                for layer, activation in steering_vector.layer_activations.items()
            },
        },
        path,
    )


def generate_completion(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=50)
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# training samples are tuples of (positive_prompt, negative_prompt)
training_samples = [
    (
        "Question: Is the Earth flat?\nAnswer: No, the Earth is round.",
        "Question: Is the Earth flat?\nAnswer: Yes, the Earth is flat.",
    ),
    (
        "Question: Do vaccines cause autism?\nAnswer: No, vaccines do not cause autism.",
        "Question: Do vaccines cause autism?\nAnswer: Yes, vaccines cause autism.",
    ),
]

if VECTOR_PATH.exists():
    steering_vector = load_steering_vector(VECTOR_PATH)
else:
    steering_vector = train_steering_vector(
        model,
        tokenizer,
        training_samples,
        move_to_cpu=True,
        show_progress=True,
    )
    save_steering_vector(VECTOR_PATH, steering_vector)

steering_vector = steering_vector.to(model.device)

prompt = "Question: Do crystals have magical healing properties?\nAnswer:"
baseline_completion = generate_completion(prompt)

with steering_vector.apply(model):
    steered_completion = generate_completion(prompt)

print("Prompt:")
print(prompt)
print()
print("Baseline completion:")
print(baseline_completion)
print()
print("Steered completion:")
print(steered_completion)
