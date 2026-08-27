import json
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from datasets import load_from_disk


def load_financial_tweets_sentiment():

    # FINANCIAL TWEETS SENTIMENT - 45.6% Positive, 32% Neutral 22.4% Negative
    ft_sentiment = load_dataset("TimKoornstra/financial-tweets-sentiment")
    fts_pandas = ft_sentiment['train'].to_pandas()
    fts_pandas['Source'] = 'Financial Tweets Sentiment'
    fts_pandas.rename(columns={'sentiment': 'Labels', 'tweet': 'Sentences'}, inplace=True)
    fts_pandas['Labels'] = fts_pandas['Labels'].replace({0: 1, 1: 2, 2: 0})
    fts_pandas = fts_pandas[['Source', 'Sentences', 'Labels', 'url']]
    ft_sentiment = Dataset.from_pandas(fts_pandas)

    return ft_sentiment


def load_financial_phrasebank():

    # FINANCIAL PHRASEBANK  28.1% Positive, 59,4% Neutral , 12.5% Negative
    phrasebank = load_dataset("financial_phrasebank", 'sentences_50agree', split='train', trust_remote_code=True)
    phrasebank_pandas = phrasebank.to_pandas()
    phrasebank_pandas['Source'] = 'Financial Phrasebank'
    phrasebank_pandas.rename(columns={'sentence': 'Sentence', 'label': 'Label'}, inplace=True)
    # reorder columns - Source first
    phrasebank_pandas = phrasebank_pandas[['Source', 'Sentence', 'Label']]
    phrasebank = Dataset.from_pandas(phrasebank_pandas)

    return phrasebank


def load_sentfin():

    def has_single_sentiment(row):
        try:
            decision_dict = json.loads(row)
            return len(decision_dict) == 1
        except json.JSONDecodeError:

            return False

    def parse_decisions(row):
        decision_dict = json.loads(row)
        # Assuming each decision has only one key-value pair
        entity, sentiment = list(decision_dict.items())[0]

        return pd.Series([entity, sentiment])

    sentfin_pandas = pd.read_csv('SEntFiN-v1.1.csv')
    sentfin_pandas = sentfin_pandas[sentfin_pandas['Decisions'].apply(has_single_sentiment)]
    sentfin_pandas[['Entity', 'Sentiment']] = sentfin_pandas['Decisions'].apply(parse_decisions)
    sentfin_pandas.reset_index(drop=True, inplace=True)
    sentfin_pandas.drop(columns=['Decisions'], inplace=True)
    sentfin_pandas['Source'] = 'SentFIN'
    sentfin_pandas.rename(
        columns={'S No.': 'SentFIN_ID', 'Title': 'Sentences', 'Entity': 'Entities', 'Words': 'Word Count',
                 'Sentiment': 'Labels'}, inplace=True)
    sentfin_pandas['Labels'] = sentfin_pandas['Labels'].replace({'negative': 0, 'neutral': 1, 'positive': 2})
    sentfin_pandas = sentfin_pandas[['Source', 'Sentences', 'Labels', 'Entities', 'SentFIN_ID', 'Word Count']]
    sentfin = Dataset.from_pandas(sentfin_pandas)

    return sentfin


def load_dataset_from_disk_to_pd(data_code):

    dataset = load_from_disk(data_code)
    pd_dataset = dataset.to_pandas()

    return pd_dataset


def load_attack_results_from_json(file_name):

    pd_results = pd.read_json(file_name, orient='index')
    pd_results = pd_results.reset_index().rename(
        columns={'index': 'Sentence_ID', 'success': 'Attack_Success', 'result': 'Adverserial_Example'})

    return pd_results
