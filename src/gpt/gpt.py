from pathlib import Path

import numpy as np

from src.core import AdamWOptimizer, WarmupCosineScheduler
from src.gpt import CharDataset, GPT, GPTTrainer, GPTLoss

np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_FILE = str(DATA_DIR / "tinyshakespeare.txt")
MODEL_FILE = str(DATA_DIR / "tinyshakespeare-gpt.npz")

BATCH_SIZE = 16
CONTEXT_SIZE = 256
EMBEDDING_SIZE = 384
HEADS = 6
BLOCKS = 6
EPOCHS = 10

GPT_LR = 0.001


def download_file():
    import requests

    file = Path(DATA_FILE)
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        response.raise_for_status()
        file.write_text(response.text)


def gpt_train():
    dataset = CharDataset(DATA_FILE, BATCH_SIZE, CONTEXT_SIZE)
    model = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    loss_fn = GPTLoss()
    optimizer = AdamWOptimizer(model.parameters, lr=GPT_LR)
    trainer = GPTTrainer(model, loss_fn, optimizer)

    scheduler = WarmupCosineScheduler(GPT_LR, EPOCHS * len(dataset), 100, GPT_LR / 10)
    trainer.train(dataset, EPOCHS, scheduler, MODEL_FILE)
    return trainer, dataset


def gpt_test(model, dataset):
    prediction, loss = model.test(dataset)
    print(f'prediction:\t{len(prediction)} steps, each {prediction[0].data.shape}\nloss:\t{loss}')


def gpt_generate(model, dataset):
    print(model.generate(dataset, prompt="ROMEO:"))


if __name__ == "__main__":
    download_file()
    trainer, dataset = gpt_train()
    gpt_test(trainer, dataset)
    gpt_generate(trainer, dataset)
