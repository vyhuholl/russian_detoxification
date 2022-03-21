from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    RobertaForMaskedLM,
    RobertaTokenizer,
)

MODEL_CLASSES = {
    "bert": (
        BertForMaskedLM,
        BertTokenizer,
        "ruBERT-large",
    ),
    "roberta": (
        RobertaForMaskedLM,
        RobertaTokenizer,
        "sberbank-ai/ruRoberta-large",
    ),
}
