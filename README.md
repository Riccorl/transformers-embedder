<div align="center">

# Transformers Embedder

[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://github.dev/Riccorl/transformers-embedder)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/4.16-🤗%20Transformers-6670ff)](https://huggingface.co/transformers/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

[![Upload to PyPi](https://github.com/Riccorl/transformers-embedder/actions/workflows/python-publish-pypi.yml/badge.svg)](https://github.com/Riccorl/transformers-embedder/actions/workflows/python-publish-pypi.yml)
[![Upload to PyPi](https://github.com/Riccorl/transformers-embedder/actions/workflows/python-publish-conda.yml/badge.svg)](https://github.com/Riccorl/transformers-embedder/actions/workflows/python-publish-conda.yml)
[![PyPi Version](https://img.shields.io/github/v/release/Riccorl/transformers-embedder)](https://github.com/Riccorl/transformers-embedder/releases)
[![Anaconda-Server Badge](https://anaconda.org/riccorl/transformers-embedder/badges/version.svg)](https://anaconda.org/riccorl/transformers-embedder)
[![DeepSource](https://deepsource.io/gh/Riccorl/transformers-embedder.svg/?label=active+issues)](https://deepsource.io/gh/Riccorl/transformers-embedder/?ref=repository-badge)

</div>

A Word Level Transformer layer based on PyTorch and 🤗 Transformers.

## How to use

Install the library from [PyPI](https://pypi.org/project/transformers-embedder):

```bash
pip install transformers-embedder
```

or from [Conda](https://anaconda.org/riccorl/transformers-embedder):

```bash
conda install -c riccorl transformers-embedder
```

It offers a PyTorch layer and a tokenizer that support almost every pretrained model from Huggingface [🤗Transformers](https://huggingface.co/transformers/) library. Here is a quick example:

```python
import transformers_embedder as tre

tokenizer = tre.Tokenizer("bert-base-cased")
model = tre.TransformersEmbedder("bert-base-cased", return_words=True, output_layer="sum")

example = "This is a sample sentence"
inputs = tokenizer(example, return_tensors=True)
```

```text
{
   'input_ids': tensor([[ 101, 1188, 1110,  170, 6876, 5650,  102]]),
   'attention_mask': tensor([[True, True, True, True, True, True, True]]),
   'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0]])
   'offsets': tensor([[0, 1, 2, 3, 4, 5, 6]]),
   'sentence_length': 7  # with special tokens included
}
```

```python
outputs = model(**inputs)
```

```text
# outputs.shape[1:-1]       # remove [CLS] and [SEP]
torch.Size([1, 5, 768])
# len(example)
5
```

## Info

One of the annoyance of using transfomer-based models is that it is not trivial to compute word embeddings from the sub-token embeddings they output. With this API it's as easy as using 🤗Transformers to get word-level embeddings from theoretically every transformer model it supports.

### Model

The `TransformersEmbedder` offer 2 ways to retrieve the embeddings:

- `return_words=True`: computes the mean of the embeddings of the sub-tokens of each word
- `return_words=False`: returns the raw output of the transformer model without sub-token pooling

There are also multiple type of outputs you can get using `output_layer` parameter:

- `last`: returns the last hidden state of the transformer model
- `concat`: returns the concatenation of the last four hidden states of the transformer model
- `sum`: returns the sum of the last four hidden states of the transformer model
- `pooled`: returns the output of the pooling layer

If you also want all the outputs from the HuggingFace model, you can set `return_all=True` to get them.

```python
class TransformersEmbedder(torch.nn.Module):
    def __init__(
        self,
        model: Union[str, tr.PreTrainedModel],
        return_words: bool = True,
        output_layer: str = "last",
        fine_tune: bool = True,
        return_all: bool = False,
    )
```

### Tokenizer

The `Tokenizer` class provides the `tokenize` method to preprocess the input for the `TransformersEmbedder` layer. You
can pass raw sentences, pre-tokenized sentences and sentences in batch. It will preprocess them returning a dictionary
with the inputs for the model. By passing `return_tensors=True` it will return the inputs as `torch.Tensor`.

By default, if you pass text (or batch) as strings, it splits them on spaces

```python
text = "This is a sample sentence"
tokenizer(text)

text = ["This is a sample sentence", "This is another sample sentence"]
tokenizer(text)
```

You can also use SpaCy to pre-tokenize the inputs into words first, using `use_spacy=True`

```python
text = "This is a sample sentence"
tokenizer(text, use_spacy=True)

text = ["This is a sample sentence", "This is another sample sentence"]
tokenizer(text, use_spacy=True)
```

or you can pass an pre-tokenized sentence (or batch of sentences) by setting `is_split_into_words=True`

```python
text = ["This", "is", "a", "sample", "sentence"]
tokenizer(text, is_split_into_words=True)

text = [
    ["This", "is", "a", "sample", "sentence", "1"],
    ["This", "is", "sample", "sentence", "2"],
]
tokenizer(text, is_split_into_words=True) # here is_split_into_words is redundant
```

#### Examples

First, initialize the tokenizer

```python
import transformers_embedder as tre

tokenizer = tre.Tokenizer("bert-base-cased")
```

- You can pass a single sentence as a string:

```python
text = "This is a sample sentence"
tokenizer(text)
```

```text
{
  'input_ids': [101, 1188, 1110, 170, 6876, 5650, 102],
  'offsets': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)],
  'attention_mask': [True, True, True, True, True, True, True],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0],
  'sentence_length': 7
}
```

- A sentence pair

```python
text = "This is a sample sentence A"
text_pair = "This is a sample sentence B"
tokenizer(text, text_pair)
```

```text
{
  'input_ids': [101, 1188, 1110, 170, 6876, 5650, 138, 102, 1188, 1110, 170, 6876, 5650, 139, 102],
  'attention_mask': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'offsets': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
  'sentence_length': 15
}
```

- A batch of sentences or sentence pairs. Using `padding=True` and `return_tensors=True`, the tokenizer returns the text ready for the model

```python
batch = [
    ["This", "is", "a", "sample", "sentence", "1"],
    ["This", "is", "sample", "sentence", "2"],
    ["This", "is", "a", "sample", "sentence", "3"],
    # ...
    ["This", "is", "a", "sample", "sentence", "n", "for", "batch"],
]
tokenizer(batch, padding=True, return_tensors=True)

batch_pair = [
    ["This", "is", "a", "sample", "sentence", "pair", "1"],
    ["This", "is", "sample", "sentence", "pair", "2"],
    ["This", "is", "a", "sample", "sentence", "pair", "3"],
    # ...
    ["This", "is", "a", "sample", "sentence", "pair", "n", "for", "batch"],
]
tokenizer(batch, batch_pair, padding=True, return_tensors=True)
```

#### Custom fields

It is possible to add custom fields to the model input and tell the `tokenizer` how to pad them using `add_padding_ops`.
Start by simply tokenizing the input (without padding or tensor mapping)

```python
import transformers_embedder as tre

tokenizer = tre.Tokenizer("bert-base-cased")

text = [
    ["This", "is", "a", "sample", "sentence"],
    ["This", "is", "another", "example", "sentence", "just", "make", "it", "longer"]
]
inputs = tokenizer(text)
```

Then add the custom fileds to the result

```python
custom_fields = {
  "custom_filed_1": [
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  ]
}

inputs.update(custom_fields)
```

Now we can add the padding logic for our custom field `custom_filed_1`. `add_padding_ops` method takes in input

- `key`: name of the field in the tokenzer input
- `value`: value to use for padding
- `length`: length to pad. It can be an `int`, or two string value, `subtoken` in which the element is padded to the batch max length relative to the sub-tokens length, and `word` where the element is padded to the batch max length relative to the original word length

```python
tokenizer.add_padding_ops("custom_filed_1", 0, "word")
```

Finally, pad the input and convert it to a tensor:

```python
# manual processing
inputs = tokenizer.pad_batch(inputs)
inputs = tokenizer.to_tensor(inputs)
```

The inputs are ready for the model, including the custom filed.

```text
>>> inputs

{
   "input_ids": tensor(
       [
           [101, 1188, 1110, 170, 6876, 5650, 102, 0, 0, 0, 0],
           [101, 1188, 1110, 1330, 1859, 5650, 1198, 1294, 1122, 2039, 102],
       ]
   ),
   "attention_mask": tensor(
       [
           [True, True, True, True, True, True, True, False, False, False, False],
           [True, True, True, True, True, True, True, True, True, True, True],
       ]
   ),
   "word_mask": tensor(
       [
           [True, True, True, True, True, True, True, False, False, False, False],
           [True, True, True, True, True, True, True, True, True, True, True],
       ]
   ),
   "token_type_ids": tensor(
       [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
   ),
   "offsets": tensor(
       [
           [0, 1, 2, 3, 4, 5, 6, 7, 10, 10, 10],
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
       ]
   ),
   "sentence_length": tensor([7, 11]),
   "custom_filed_1": tensor(
       [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]
   ),
}
```

#### SpaCy Tokenizer

By default, it uses the [multilingual model](https://spacy.io/models/xx#xx_sent_ud_sm) `xx_sent_ud_sm`. You can change
it with the `language` parameter during the tokenizer initialization. For example, if you prefer an English tokenizer:

```python
tokenizer = tre.Tokenizer("bert-base-cased", language="en_core_web_sm")
```

For a complete list of languages and models, you can go [here](https://spacy.io/models).

## To-Do

Future developments

- [X] Add an optional word tokenizer, maybe using SpaCy
- [X] Add `add_special_tokens` wrapper
- [X] Make `pad_batch` function more general
- [X] Add logic (like how to pad, etc) for custom fields
  - [X] Documentation
- [X] Include all model outputs
  - [X] Documentation
- [ ] A TensorFlow version (improbable)

[comment]: <> (- [ ] Include more &#40;maybe all&#41; tokenizer outputs)

## Acknowledgements

Some of the code in the `TransformersEmbedder` class is taken from the [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter/)
library. The pretrained models and the core of the tokenizer is from [🤗 Transformers](https://huggingface.co/transformers/).
