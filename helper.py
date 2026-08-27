import datasets
import os
import shutil
from datasets import load_from_disk
from difflib import ndiff
import pandas as pd

def save_dataset_to_disk (data_code,dataset):
    if not os.path.exists(data_code):
        dataset.save_to_disk(data_code)
        print(f'{data_code} file saved successfully.')
    else:
        print(f'{data_code} already exists in the directory. No action taken.')

def setup_directory(data_code):
    """
    Set up a directory named 'directory_<data_code>' in the current working directory.
    If the directory does not exist, it will be created.
    The function then changes the current working directory to this new directory and prints the path.

    Parameters:
    data_code (str): A string to append to the directory name.
    """
    dir_path = os.getcwd()  # Get the current working directory
    new_dir = os.path.join(dir_path, f'directory_{data_code}')  # Define the new directory path

    os.makedirs(new_dir, exist_ok=True)  # Create the subdirectory if it doesn't exist
    os.chdir(new_dir)  # Change to the subdirectory
    dir_path = os.getcwd()  # Update the current working directory path

    print(dir_path)  # Print the new current working directory path


def create_file_names(model_list, data_code, after_code):
    attack_result_list = []
    for model in model_list:
        attack_result_list.append(f'{model}_{data_code}_{after_code}')

    return attack_result_list


def bring_4models_list():
    model_list = ['RB', 'DR', 'FA', 'FH']

    return model_list


def bring_datasets_list(search_string, dir_path):
    matching_directories = []

    # Walk through the directory
    for root, dirs, files in os.walk(dir_path):
        # Check each directory in the current root directory
        for dir in dirs:
            # If the search string is in the directory name, add it to the list
            print(dir)
            if search_string in dir:
                matching_directories.append(os.path.join(root, dir))

    return matching_directories


def extract_model_names_from_paths(dataset_paths):
    model_names = []
    suffix = "_phrasebank_1020"

    for path in dataset_paths:
        # Get the last part of the path which is the directory name
        dir_name = os.path.basename(path)
        # Find the position where the model name suffix starts
        index = dir_name.find(suffix)
        if index != -1:
            # Extract the model name based on the known suffix position
            model_name = dir_name[:index]
            model_names.append(model_name)

    return model_names


def extract_model_code(model_name):
    model_code = model_name.split('_')[0]

    return model_code


def standardize_col_names(dataset, model_code):
    df = dataset.to_pandas()
    df = df.rename(columns={
        'Labels': 'Ground_Truth',
        'Label': 'Ground_Truth',
        'Sentences': 'Sentence_Original',
        'Sentence': 'Sentence_Original',
        'Sentences_Sentiment': f'Sentiment_{model_code}',
        'Sentence_Sentiment': f'Sentiment_{model_code}',
        # Attacks
        'SCPN_result': 'Sentence_SCPN',
        'SCPN_result_Sentiment': 'Sentiment_SCPN',
        'SCPN_success': 'Bool_Success_SCPN',
        'SCPN_metrics': 'Metrics_SCPN',
        'TF_result': 'Sentence_TF',
        'TF_result_Sentiment': 'Sentiment_TF',
        'TF_success': 'Bool_Success_TF',
        'TF_metrics': 'Metrics_TF',
        'UAT_result': 'Sentence_UAT',
        'UAT_result_Sentiment': 'Sentiment_UAT',
        'UAT_success': 'Bool_Success_UAT',
        'UAT_metrics': 'Metrics_UAT',
        'TBG_result': 'Sentence_TBG',
        'TBG_result_Sentiment': 'Sentiment_TBG',
        'TBG_success': 'Bool_Success_TBG',
        'TBG_metrics': 'Metrics_TBG',
        'DWB_result': 'Sentence_DWB',
        'DWB_result_Sentiment': 'Sentiment_DWB',
        'DWB_success': 'Bool_Success_DWB',
        'DWB_metrics': 'Metrics_DWB',
        'PWWS_result': 'Sentence_PWWS',
        'PWWS_result_Sentiment': 'Sentiment_PWWS',
        'PWWS_success': 'Bool_Success_PWWS',
        'PWWS_metrics': 'Metrics_PWWS',
        'PSO_result': 'Sentence_PSO',
        'PSO_result_Sentiment': 'Sentiment_PSO',
        'PSO_success': 'Bool_Success_PSO',
        'PSO_metrics': 'Metrics_PSO'
    })

    fixed_columns = ['Sentence_ID', 'Source', 'Sentence_Original', 'Ground_Truth', f'Sentiment_{model_code}']
    sorted_remaining_columns = sorted(col for col in df.columns if col not in fixed_columns)
    df = df[fixed_columns + sorted_remaining_columns]

    dataset = datasets.Dataset.from_pandas(df)

    return dataset


def clear_datasets_cache():
    datasets_cache_dir = os.getenv('HF_DATASETS_CACHE')
    # Delete the datasets cache directory to clear the cache
    if os.path.exists(datasets_cache_dir):
        shutil.rmtree(datasets_cache_dir)
        print("Datasets cache cleared.")
    else:
        print("No datasets cache directory found to clear.")


def highlight_differences(original, adversarial):
    # Use ndiff to get differences between the original and adversarial sentences
    differences = list(ndiff(original.split(), adversarial.split()))

    # Prepare the highlighted output
    highlighted = []
    for diff in differences:
        if diff.startswith('+ '):
            # Additions in green
            highlighted.append(f'<span style="background-color: green;">{diff[2:]}</span>')
        elif diff.startswith('- '):
            # Deletions in red
            highlighted.append(f'<span style="background-color: red;">{diff[2:]}</span>')
        else:
            # Unchanged parts
            highlighted.append(diff[2:])

    return ' '.join(highlighted)


def bring_adversarial_samples(model_code, data_code):
    # Assuming load_from_disk and standardize_col_names are defined elsewhere and work as intended
    data_path = f'{model_code}_{data_code}_complete_sentiment'
    dataset = load_from_disk(data_path)
    dataset = standardize_col_names(dataset, model_code)
    df = dataset.to_pandas()

    # Extract all unique attack codes
    attack_codes = [col.split('_')[2] for col in df.columns if col.startswith('Bool_Success_')]

    results = []

    for attack_code in attack_codes:
        success_col = f'Bool_Success_{attack_code}'
        filtered_df = df[df[success_col] == True]

        if not filtered_df.empty:
            sample = filtered_df.sample(1)
            sentence_id = sample['Sentence_ID'].iloc[0]
            original_sentence = sample['Sentence_Original'].iloc[0]
            adv_sample = sample[f'Sentence_{attack_code}'].iloc[0]
            highlighted_sample = highlight_differences(original_sentence, adv_sample)

            results.append({
                'sentence_id': sentence_id,
                'attack_code': attack_code,
                'original_sentence': original_sentence,
                'adversarial_sample': adv_sample,
                'highlighted_sample': highlighted_sample
            })
    return results, df


def bring_one_adversarial_sample(model_code, data_code):
    # Load the dataset
    data_path = f'{model_code}_{data_code}_complete_sentiment'  # Adjust path as needed
    dataset = load_from_disk(data_path)  # This line may need to be changed depending on your data loading method
    dataset = standardize_col_names(dataset, model_code)
    df = dataset.to_pandas()

    # Randomly select one sample from the entire dataset first
    sample = df.sample(1).iloc[0]

    # Assuming 'Sentence_ID' is the column name for sentence identifiers
    sentence_id = sample['Sentence_ID']

    # Extract all unique attack codes
    attack_codes = [col.split('_')[2] for col in df.columns if col.startswith('Bool_Success_')]

    results = []

    # Check each attack code to see if the attack was successful for the randomly selected sample
    for attack_code in attack_codes:
        success_col = f'Bool_Success_{attack_code}'

        # Check if the attack was successful
        if sample[success_col]:
            original_sentence = sample['Sentence_Original']
            adversarial_sample = sample[f'Sentence_{attack_code}']
            highlighted_sample = highlight_differences(original_sentence, adversarial_sample)

            results.append({
                'sentence_id': sentence_id,  # Include sentence_id in the result
                'attack_code': attack_code,
                'original_sentence': original_sentence,
                'adversarial_sample': adversarial_sample,
                'highlighted_sample': highlighted_sample
            })

    return results, df


def bring_sentence_id(df, id):
    selected_sentence = df[df['Sentence_ID'] == id]

    return selected_sentence
