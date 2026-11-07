import pickle
import re
from itertools import pairwise

import numpy as np

from src.core import AdamWOptimizer, WarmupCosineScheduler
from src.gpt import CharDataset, GPT
from src.gpt.gpt import DATA_DIR, DATA_FILE, CONTEXT_SIZE, BATCH_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS, MODEL_FILE
from src.sft.sft_dataset import SFTDataset
from src.sft.sft_loss import SFTLoss
from src.sft.sft_trainer import SFTTrainer

np.random.seed(42)

SFT_SAMPLES = str(DATA_DIR / "sft-samples.pkl")
SFT_MODEL = str(DATA_DIR / "tinyshakespeare-sft.npz")

SFT_LR = 0.0001


def extract_turns(filename, split=0.9, max_turn_chars=100):
    with open(filename, encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(r"\n\s*\n", text.strip())
    turns = []

    split = int(len(blocks) * split)
    for block in blocks[split:]:
        lines = block.strip("\n").split("\n")
        if not lines:
            continue

        m = re.compile(r"^([A-Z][A-Za-z' ]{0,30}):\s*$").match(lines[0].strip())
        if not m:
            continue

        speaker = m.group(1).strip()
        content_lines = []
        length = 0
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            if length + len(line) + 1 > max_turn_chars:
                break
            content_lines.append(line)
            length += len(line) + 1
        if not content_lines:
            continue

        turns.append((speaker, "\n".join(content_lines)))

    return turns


def sft_sample():
    dataset = CharDataset(DATA_FILE, 1, CONTEXT_SIZE)

    pairs = list(pairwise(extract_turns(DATA_FILE)))
    np.random.shuffle(pairs)
    pairs = pairs[:512]

    samples = []
    for (speaker_a, content_a), (speaker_b, content_b) in pairs:
        prompt = f"{speaker_a}:\n{content_a}\n\n"
        response = f"{speaker_b}:\n{content_b}\n\n"
        samples.append((dataset.encode(prompt), dataset.encode(response)))

    with open(SFT_SAMPLES, "wb") as f:
        pickle.dump(samples, f)
    print(f"saved {len(samples)} SFT samples to {SFT_SAMPLES}")

    sample = samples[0]
    print(f"{dataset.decode(sample[0])}\n{dataset.decode(sample[1])}")


def sft_train():
    dataset = CharDataset(DATA_FILE, BATCH_SIZE, CONTEXT_SIZE)
    model = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    loss_fn = SFTLoss()
    optimizer = AdamWOptimizer(model.parameters, lr=SFT_LR)
    trainer = SFTTrainer(model, loss_fn, optimizer)
    trainer.load(MODEL_FILE)

    sft_dataset = SFTDataset(SFT_SAMPLES, CONTEXT_SIZE)
    scheduler = WarmupCosineScheduler(SFT_LR, len(sft_dataset), 50, SFT_LR / 10)
    trainer.train(sft_dataset, 1, scheduler, SFT_MODEL)
    return trainer, dataset


def sft_generate(model, dataset):
    print(model.generate(dataset, prompt="ROMEO:"))


if __name__ == "__main__":
    sft_sample()
    trainer, dataset = sft_train()
    sft_generate(trainer, dataset)
