from transformers import RobertaForMaskedLM, RobertaTokenizer

MODEL_CLASSES = {
    "roberta": (
        RobertaForMaskedLM,
        RobertaTokenizer,
        "sberbank-ai/ruRoberta-large",
    )
}
