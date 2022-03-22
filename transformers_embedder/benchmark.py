import itertools
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from torch.types import Device
from torch.utils.data import DataLoader

import transformers_embedder as tre

torch.set_grad_enabled(False)

seed: int = 42
transformer_name: str = "bert-base-cased"
device: Device = "cuda"

# seed_everything by PyTorch Lightning
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


tokenizer = tre.Tokenizer(transformer_name)

dataset = load_dataset("Babelscape/wikineural", split="test_en")
batch_size = 160
n_batches = 50

dataloader = DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=batch_size,
    pin_memory=True,
    collate_fn=lambda samples: tokenizer(
        [sample["tokens"] for sample in samples],
        padding=True,
        return_tensors=True,
        is_split_into_words=True,
        compute_bpe_info=True,
    ),
)
iter_dataloader = iter(dataloader)

batches = [next(iter_dataloader).to(device) for _ in range(n_batches)]

pooling2model = {
    pooling_strategy: tre.TransformersEmbedder(
        transformer_name,
        return_words=True,
        layer_pooling_strategy="mean",
        subword_pooling_strategy=pooling_strategy,
    )
    .to(device)
    .eval()
    for pooling_strategy in ("scatter", "sparse", "inefficient")
}

pooling2output = {pooling: model(**batches[0]) for pooling, model in pooling2model.items()}
atol = 1e-7
for (pool1, out1), (pool2, out2) in itertools.combinations(pooling2output.items(), r=2):
    all_close = torch.allclose(out1.word_embeddings, out2.word_embeddings, atol=atol)
    print(f"{pool1} == {pool2} (allclose with {atol=}): {all_close}")

for pooling, model in pooling2model.items():
    start = time.time()
    for batch in batches:
        model(**batch)
    end = time.time()
    print(pooling, (end - start))
