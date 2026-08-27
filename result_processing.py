import datasets
from data_merge import merge_attack_results_with_dataset
from datasets import Dataset
from fnmatch import fnmatch
from os import walk, path
import os


def print_friendly_from_list(list_of_results):
    friendly_names = []

    for path in list_of_results:
        # Extract the filename from the path
        filename = path.split('/')[-1]
        # Assume the part before the first date is what you want to keep
        # This splits the filename by underscores and removes the last 3 segments (date and time)
        name_parts = filename.split('_')[:-3]
        friendly_name = '_'.join(name_parts)
        # Add a tab before the friendly name
        friendly_names.append('\t' + friendly_name)

    # Return a string with each name on a new line
    return '\n'.join(friendly_names)


def create_results_dataset_for_models(list_of_models, original_dataset, data_code, dir_path):
    # Check if the directory 'datasets_with_attack_results' exists, if not, create it
    results_dir = os.path.join(dir_path, 'datasets_with_attack_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for model in list_of_models:
        list_of_results = bring_results_for_model(model, data_code, dir_path)
        number_of_files = len(list_of_results)
        dataframe_with_results = merge_attack_results_with_dataset(original_dataset, list_of_results)
        dataset_with_results = Dataset.from_pandas(dataframe_with_results)

        print('List of attacks found and merged with original dataset:')
        print(print_friendly_from_list(list_of_results))

        print(f'Saving {model} dataset with attack results')
        save_path = os.path.join(results_dir, f"{model}_{data_code}_with_attack_results")
        dataset_with_results.save_to_disk(save_path)

    return


def bring_results_for_model(model_code, data_code, dir_path):
    # Adjust the pattern to include a wildcard after the data_code
    pattern = f"*_vs_{model_code}_{data_code}*.json"
    matching_files = []

    for root, dirs, files in walk(dir_path):
        for filename in files:
            if fnmatch(filename, pattern):
                matching_files.append(path.join(root, filename))

    return matching_files


def filter_adv_sample_columns(dataset):
    all_columns = dataset.column_names

    result_columns = [column for column in all_columns if '_result' in column]

    return result_columns


def fill_none_for_failed_attacks(dataset, adv_sample_columns):
    df = dataset.to_pandas()
    # Fill NA values in specified columns with values from the 'Sentence' column
    for column in adv_sample_columns:
        df[column] = df[column].fillna(df['Sentence'])
    dataset = Dataset.from_pandas(df)

    return dataset


def bring_dataset_with_attack_results(model_code, data_code, dir_path):
    file_name = f'{model_code}_{data_code}_with_attack_results'
    dataset_path = os.path.join(dir_path, 'datasets_with_attack_results', file_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Directory {dataset_path} not found")

    return datasets.load_from_disk(dataset_path)