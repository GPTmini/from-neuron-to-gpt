import pickle
from itertools import pairwise

import numpy as np

from src.core import WarmupCosineScheduler, AdamWOptimizer, Tensor
from src.dpo.dpo_dataset import DPODataset
from src.dpo.dpo_loss import DPOLoss
from src.dpo.dpo_trainer import DPOTrainer
from src.gpt import CharDataset, GPT, GPTTrainer
from src.gpt.gpt import DATA_DIR, CONTEXT_SIZE, MODEL_FILE, EMBEDDING_SIZE, HEADS, BLOCKS, BATCH_SIZE, DATA_FILE
from src.sft.sft import SFT_MODEL, extract_turns

np.random.seed(42)

DPO_SAMPLES = str(DATA_DIR / "dpo-samples.pkl")
DPO_MODEL = str(DATA_DIR / "tinyshakespeare-dpo.npz")

DPO_LR = 0.000001


def generate_rejected(model, prompt, length, temperature=0.8, top_k=20):
    tokens = list(prompt)

    with Tensor.no_grad():
        for _ in range(length):
            logits = model(Tensor([tokens[-CONTEXT_SIZE:]]))
            last_logits = logits.data[0, -1] / temperature

            k = min(top_k, last_logits.size)
            threshold = np.partition(last_logits, -k)[-k]
            last_logits = np.where(last_logits < threshold, -np.inf, last_logits)
            probs = np.exp(last_logits - np.max(last_logits))
            probs /= np.sum(probs)
            tokens.append(int(np.random.choice(len(probs), p=probs)))

    return tokens[len(prompt):]


def dpo_sample():
    dataset = CharDataset(DATA_FILE, BATCH_SIZE, CONTEXT_SIZE)

    model = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    trainer = GPTTrainer(model, None, None)
    trainer.load(MODEL_FILE)
    model.eval()

    pairs = list(pairwise(extract_turns(DATA_FILE)))
    np.random.shuffle(pairs)
    pairs = pairs[:512]

    samples = []
    for i, ((speaker_a, content_a), (speaker_b, content_b)) in enumerate(pairs):
        prompt = dataset.encode(f"{speaker_a}:\n{content_a}\n\n")
        chosen = dataset.encode(f"{speaker_b}:\n{content_b}\n\n")
        rejected = generate_rejected(model, prompt, len(chosen))

        if rejected != chosen:
            samples.append((prompt, chosen, rejected))

        if (i + 1) % 64 == 0:
            print(f"generated {len(samples)}/{len(pairs)} DPO samples")

    with open(DPO_SAMPLES, "wb") as f:
        pickle.dump(samples, f)
    print(f"saved {len(samples)} DPO samples to {DPO_SAMPLES}")

    sample = samples[0]
    print(f"{dataset.decode(sample[0])}\n{dataset.decode(sample[1])}\n{dataset.decode(sample[2])}")


def dpo_train():
    dataset = CharDataset(DATA_FILE, BATCH_SIZE, CONTEXT_SIZE)
    model = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    reference = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    loss_fn = DPOLoss()
    optimizer = AdamWOptimizer(model.parameters, lr=DPO_LR)
    trainer = DPOTrainer(model, loss_fn, optimizer, reference)
    trainer.load(SFT_MODEL)
    trainer.load_reference(SFT_MODEL)

    dpo_dataset = DPODataset(DPO_SAMPLES, CONTEXT_SIZE)
    scheduler = WarmupCosineScheduler(DPO_LR, len(dpo_dataset), 50, DPO_LR / 10)
    trainer.train(dpo_dataset, 1, scheduler, DPO_MODEL)
    return trainer, dataset


def dpo_generate(model, dataset):
    print(model.generate(dataset, prompt="ROMEO:"))


if __name__ == "__main__":
    dpo_sample()
    trainer, dataset = dpo_train()
    dpo_generate(trainer, dataset)
