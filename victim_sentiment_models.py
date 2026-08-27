from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging


def initialize_model(model_code):
    victim_initializers = {
        'DR': initialize_distilroberta,
        'RB': initialize_robertabase,
        'FH': initialize_finberthuang,
        'FA': initialize_finbertaraci,
    }

    initializer = victim_initializers.get(model_code)

    if initializer:
        return initializer()
    else:
        raise ValueError(f"Model code '{model_code}' not recognized.")


def initialize_distilroberta():
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    return prepare_model_components_roberta(model, tokenizer)


def initialize_robertabase():
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    return prepare_model_components_roberta(model, tokenizer)


def initialize_finbertaraci():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return prepare_model_components_bert(model, tokenizer)


def initialize_finberthuang():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return prepare_model_components_bert(model, tokenizer)


def prepare_model_components_roberta(model, tokenizer):
    from OpenAttack import classifiers
    from textattack import models

    wrapper = models.wrappers.HuggingFaceModelWrapper(model=model, tokenizer=tokenizer)
    victim = classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)

    return victim, wrapper, model, tokenizer


def prepare_model_components_bert(model, tokenizer):
    from OpenAttack import classifiers
    from textattack import models

    wrapper = models.wrappers.HuggingFaceModelWrapper(model=model, tokenizer=tokenizer)
    victim = classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    return victim, wrapper, model, tokenizer
