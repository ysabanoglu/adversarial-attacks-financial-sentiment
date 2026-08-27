from os import path
from json import load as js_load

from datasets import Dataset
from pandas import DataFrame
from pandas import concat
from pandas import merge


def left_join_dataset_with_result(original_dataset, attack_result):

    # take single attack result and merge it with original dataset
    dataset_with_result = merge(original_dataset, attack_result, on='Sentence_ID', how='left')

    return dataset_with_result


def merge_phrasebank_sentfin(pd_phrasebank, pd_sentfin):

    merged_df = concat([pd_phrasebank, pd_sentfin], ignore_index=True, sort=False)
    # if sentence and label same keep first sample, drop others
    merged_df.drop_duplicates(subset=['Sentence', 'Label'], inplace=True)
    # if sentences are same but labels are different remove them completely to calculate correct metrics.
    merged_df.drop_duplicates(subset=['Sentence'], keep=False, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    id_column = range(1, len(merged_df) + 1)
    merged_df.insert(0, 'Sentence_ID', id_column)

    merged_df.to_csv('phrasebank_and_sentfin.csv', index=False)
    final_dataset = Dataset.from_pandas(merged_df, preserve_index=False)
    final_dataset.save_to_disk('phrasebank_and_sentfin')

    return merged_df


def merge_attack_results_with_dataset(original_dataset, json_files):

    df = original_dataset.to_pandas()

    # Ensure the 'Sentence_ID' column is of a consistent type, e.g., str, to avoid merge issues.
    df = df[['Sentence_ID', 'Source', 'Sentence', 'Label']].astype({'Sentence_ID': str})

    for json_file in json_files:
        # Extract the attack code from the filename
        attack_code = path.basename(json_file).split('_vs_')[0]

        # Load JSON file into a DataFrame
        with open(json_file, 'r') as file:
            json_data = js_load(file)
        pd_attack_result = DataFrame.from_dict(json_data, orient='index').reset_index().rename(
            columns={'index': 'Sentence_ID'})
        pd_attack_result['Sentence_ID'] = pd_attack_result['Sentence_ID'].astype(
            str)  # Ensure this is the same type as in `df`

        # Dynamically rename all columns except 'Sentence_ID' to include the attack code as a prefix
        pd_attack_result.columns = ['Sentence_ID'] + [f"{attack_code}_{col}" for col in pd_attack_result.columns if
                                                      col != 'Sentence_ID']

        df = merge(df, pd_attack_result, on='Sentence_ID', how='left')

    return df