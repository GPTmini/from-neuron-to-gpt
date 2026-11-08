import os

import numpy as np

from src.core import Layer, Tensor, DTYPE, Composite, Embedding, Linear, Softmax, Trainer


class GELU(Layer):

    def __init__(self):
        super().__init__()
        self.c = np.sqrt(2.0 / np.pi)

    def forward(self, x: Tensor):
        tanh = np.tanh(self.c * (x.data + 0.044715 * x.data ** 3))
        a = Tensor(0.5 * x.data * (1.0 + tanh))

        def gradient_fn():
            grad = 0.5 * (1.0 + tanh) + 0.5 * x.data * (1.0 - tanh ** 2) * self.c * (1.0 + 3.0 * 0.044715 * x.data ** 2)
            x.grad += a.grad * grad

        return a.attach(gradient_fn, parents=[x])


class Tril(Layer):

    def __init__(self, value=-1e9):
        super().__init__()
        self.value = value

    def forward(self, x: Tensor):
        keep = np.tril(np.ones(x.shape[-2:]))
        p = Tensor(np.where(keep, x.data, self.value))

        def gradient_fn():
            x.grad += p.grad * keep

        return p.attach(gradient_fn, [x])


class LayerNorm(Layer):

    def __init__(self, normalized_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Tensor(np.ones(normalized_size, dtype=DTYPE))
        self.bias = Tensor(np.zeros(normalized_size, dtype=DTYPE))

    def forward(self, x: Tensor):
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True, ddof=0)
        norm = (x.data - mean) / np.sqrt(var + self.eps)
        p = Tensor(self.weight.data * norm + self.bias.data)

        def gradient_fn():
            axis = tuple(range(p.grad.ndim - 1)) if p.grad.ndim > 1 else None
            self.weight.grad += np.sum(p.grad * norm, axis=axis)
            self.bias.grad += np.sum(p.grad, axis=axis)
            grad = p.grad * self.weight.data
            grad_mean = np.mean(grad, axis=-1, keepdims=True)
            norm_mean = np.mean(grad * norm, axis=-1, keepdims=True)
            x.grad += (grad - grad_mean - norm * norm_mean) / np.sqrt(var + self.eps)

        return p.attach(gradient_fn, [x])

    @property
    def parameters(self):
        return [self.weight, self.bias]


class Dropout(Layer):

    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob

    def forward(self, x: Tensor):
        if not self.training or self.prob == 0:
            return x

        keep_prob = 1.0 - self.prob
        mask = (np.random.rand(*x.shape) < keep_prob).astype(DTYPE) / keep_prob
        p = Tensor(x.data * mask)

        def gradient_fn():
            x.grad += p.grad * mask

        return p.attach(gradient_fn, [x])


class GPTEmbedding(Composite):

    def __init__(self, vocab_size, context_size, embedding_size):
        self.embedding = Embedding(vocab_size, embedding_size)
        self.positional_embedding = Embedding(context_size, embedding_size)
        self.dropout = Dropout()

        super().__init__([self.embedding,
                          self.positional_embedding,
                          self.dropout])

    def forward(self, x: Tensor):
        token = self.embedding(x)
        position = self.positional_embedding(Tensor(range(x.shape[1])))
        return self.dropout(token + position)


class GPTAttention(Composite):

    def __init__(self, embedding_size, heads=1):
        self.embedding_size = embedding_size
        self.heads = heads

        self.normalize = LayerNorm(embedding_size)
        self.query = Linear(embedding_size, embedding_size)
        self.key = Linear(embedding_size, embedding_size)
        self.value = Linear(embedding_size, embedding_size)
        self.mask = Tril()
        self.softmax = Softmax()
        self.output = Linear(embedding_size, embedding_size)
        self.dropout = Dropout()

        super().__init__([self.normalize,
                          self.query,
                          self.key,
                          self.value,
                          self.mask,
                          self.softmax,
                          self.output,
                          self.dropout])

    def forward(self, x: Tensor):
        norm = self.normalize(x)
        head_embedding_size = self.embedding_size // self.heads
        multi_head_shape = (-1, x.shape[1], self.heads, head_embedding_size)
        shape = (-1, x.shape[1], self.embedding_size)

        query = self.query(norm).reshape(multi_head_shape).transpose((0, 2, 1, 3))
        key = self.key(norm).reshape(multi_head_shape).transpose((0, 2, 3, 1))
        value = self.value(norm).reshape(multi_head_shape).transpose((0, 2, 1, 3))
        scale = Tensor(np.array(1.0 / np.sqrt(head_embedding_size)))
        scores = query @ key * scale
        weights = self.softmax(self.mask(scores))
        return x + self.dropout(self.output((weights @ value).transpose((0, 2, 1, 3)).reshape(shape)))


class GPTFeedForward(Composite):

    def __init__(self, embedding_size):
        self.normalize = LayerNorm(embedding_size)
        self.input = Linear(embedding_size, embedding_size * 4)
        self.gelu = GELU()
        self.output = Linear(embedding_size * 4, embedding_size)
        self.dropout = Dropout()

        super().__init__([self.normalize,
                          self.input,
                          self.gelu,
                          self.output,
                          self.dropout])

    def forward(self, x: Tensor):
        norm = self.normalize(x)
        h = self.gelu(self.input(norm))
        return x + self.dropout(self.output(h))


class GPTTransformer(Composite):

    def __init__(self, embedding_size, heads):
        self.attention = GPTAttention(embedding_size, heads)
        self.feed_forward = GPTFeedForward(embedding_size)

        super().__init__([self.attention,
                          self.feed_forward])

    def forward(self, x: Tensor):
        x = self.attention(x)
        return self.feed_forward(x)


class GPTOutput(Composite):

    def __init__(self, embedding_size, vocab_size):
        self.normalize = LayerNorm(embedding_size)
        self.output = Linear(embedding_size, vocab_size)

        super().__init__([self.normalize,
                          self.output])

    def forward(self, x: Tensor):
        norm = self.normalize(x)
        return self.output(norm)


class GPT(Composite):

    def __init__(self, vocab_size, context_size, embedding_size, heads, blocks):
        self.embedding = GPTEmbedding(vocab_size, context_size, embedding_size)
        self.transformers = [GPTTransformer(embedding_size, heads) for _ in range(blocks)]
        self.output = GPTOutput(embedding_size, vocab_size)

        super().__init__([self.embedding] + self.transformers + [self.output])

    def forward(self, x: Tensor):
        x = self.embedding(x)
        for layer in self.transformers:
            x = layer(x)
        return self.output(x)


class GPTTrainer(Trainer):

    def train(self, dataset, epochs, scheduler=None, filename=None):
        dataset.train()
        self.model.train()

        steps = 0
        best_losses = float('inf')
        for epoch in range(epochs):
            order = list(range(len(dataset)))
            np.random.shuffle(order)

            total_loss = 0.0
            for step, i in enumerate(order):
                if scheduler is not None:
                    self.optimizer.lr = scheduler.step(steps)

                feature, label = dataset[i]
                prediction = self.model(feature)
                loss = self.loss_fn(prediction, label)

                self.optimizer.zero_grad()
                loss.backward()
                total_loss += float(loss.data)
                self.optimizer.clip_grad_norm()
                self.optimizer.step()
                steps += 1

                if (step + 1) % 1000 == 0:
                    lr = f" lr {self.optimizer.lr:.6f}" if scheduler is not None else ""
                    print(f"epoch {epoch + 1} step {step + 1}/{len(dataset)} loss {(total_loss / 1000):.4f}{lr}")
                    total_loss = 0.0

            _, total_loss = self.test(dataset)
            print(f"epoch {epoch + 1} done, test loss {total_loss:.4f}")
            if total_loss < best_losses and filename is not None:
                best_losses = total_loss
                self.save(filename)
                print(f"epoch {epoch + 1} saved model to {filename}")

        print("GPT-training completed")

    def test(self, dataset):
        dataset.eval()
        self.model.eval()

        predictions = []
        total_loss = 0.0
        with Tensor.no_grad():
            for i in range(len(dataset)):
                feature, label = dataset[i]
                prediction = self.model(feature)
                loss = self.loss_fn(prediction, label)
                predictions.append(prediction)
                total_loss += float(loss.data)

        loss = total_loss / len(dataset)
        dataset.train()
        self.model.train()
        return predictions, loss

    def generate(self, dataset, prompt, steps=256):
        self.model.eval()
        tokens = dataset.encode(prompt)

        with Tensor.no_grad():
            for _ in range(steps):
                feature = Tensor([tokens[-dataset.context_size:]])
                logits = self.model(feature)

                last_logits = logits.data[0, -1]
                exp = np.exp(last_logits - np.max(last_logits))
                probs = exp / np.sum(exp)
                token = np.random.choice(len(probs), p=probs)
                tokens.append(token)

        return dataset.decode(tokens)

    def save(self, filename):
        params = {f"param_{i}": p.data for i, p in enumerate(self.model.parameters)}
        np.savez(filename, **params)

    def load(self, filename):
        if os.path.isfile(filename):
            data = np.load(filename, allow_pickle=False)

            for i, p in enumerate(self.model.parameters):
                p.data = data[f"param_{i}"]
                p.grad = np.zeros_like(p.data)
