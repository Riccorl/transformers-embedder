from transformers import (
    RobertaTokenizerFast,
    RobertaTokenizer,
    XLMRobertaTokenizerFast,
    XLMRobertaTokenizer,
    BertTokenizerFast,
    BertTokenizer,
    DistilBertTokenizerFast,
    DistilBertTokenizer,
    MobileBertTokenizerFast,
    MobileBertTokenizer,
    BertweetTokenizer,
    CamembertTokenizerFast,
    CamembertTokenizer,
    DebertaTokenizer,
    XLMTokenizer,
)

MODELS_WITH_STARTING_TOKEN = (
    BertTokenizer,
    BertTokenizerFast,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    MobileBertTokenizer,
    MobileBertTokenizerFast,
    BertweetTokenizer,
    CamembertTokenizer,
    CamembertTokenizerFast,
    DebertaTokenizer,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    XLMTokenizer,
)

MODELS_WITH_DOUBLE_SEP = (
    CamembertTokenizer,
    CamembertTokenizerFast,
    BertweetTokenizer,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
)

from transformer_embedder.embedder import TransformerEmbedder
from transformer_embedder.tokenizer import Tokenizer
