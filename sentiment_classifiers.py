from transformers import pipeline
import warnings
import logging


def initialize_classifier(model_code):
    if model_code == "DR":
        classifier = DR_Classifier()
    elif model_code == "RB":
        classifier = RB_Classifier()
    elif model_code == "FH":
        classifier = FH_Classifier()
    elif model_code == "FA":
        classifier = FA_Classifier()
    else:
        print('Classifier not found')
        return

    return classifier


class BaseClassifier:
    def __init__(self, model):
        self.classifier = pipeline('text-classification', model=model, truncation=True, max_length=512)

    def classify(self, dataset, target_column):
        if target_column in dataset.column_names:
            result = dataset.map(lambda batch: self.process_batch(batch, target_column), batched=True, batch_size=16)
            return result
        else:
            warnings.warn(
                f"Dataset does not contain the specified column '{target_column}'. Classification cannot be performed.")
            return dataset

    def classify_multiple_columns(self, dataset, columns):
        counter = 1
        for column in columns:
            print(f'\t {counter}-Classifying column:', column)
            dataset = self.classify(dataset, column)
            print(f'\t {counter}-Completed column:', column)
            counter += 1
        return dataset

    def process_batch(self, batch, target_column):
        predictions = self.classifier(batch[target_column])
        return {
            f'{target_column}_Sentiment': [self.label_to_numeric(prediction['label']) for prediction in predictions],
        }

    def label_to_numeric(self, label):
        raise NotImplementedError("This method should be implemented by subclasses.")


class DR_Classifier(BaseClassifier):
    def __init__(self):
        super().__init__("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    def label_to_numeric(self, label):
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        return label_map[label]


class RB_Classifier(BaseClassifier):
    def __init__(self):
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        super().__init__("cardiffnlp/twitter-roberta-base-sentiment-latest")

    def label_to_numeric(self, label):
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        return label_map[label]


class FH_Classifier(BaseClassifier):
    def __init__(self):
        super().__init__("yiyanghkust/finbert-tone")

    def label_to_numeric(self, label):
        label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        return label_map[label]


class FA_Classifier(BaseClassifier):
    def __init__(self):
        super().__init__("ProsusAI/finbert")

    def label_to_numeric(self, label):
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        return label_map[label]
