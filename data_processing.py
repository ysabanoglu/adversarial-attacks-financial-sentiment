from datasets import Dataset
from sklearn.utils import resample
import pandas as pd


def standardize_labels(label_column):

    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2, 'Negative': 0, 'Neutral': 1, 'Positive': 2,
                     'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2}
    label_column = label_column.replace(label_mapping).astype(int)

    return label_column


def output_to_integer(output_column):

    output_column = output_column.astype(int)
    return output_column


def downsample_to_equal_classes(dataset, class_sample_size):

    df = dataset.to_pandas()
    df_class0 = df[df['Label'] == 0]
    df_class1 = df[df['Label'] == 1]
    df_class2 = df[df['Label'] == 2]

    df_class0_downsampled = resample(df_class0, replace=False, n_samples=class_sample_size, random_state=123)
    df_class1_downsampled = resample(df_class1, replace=False, n_samples=class_sample_size, random_state=123)
    df_class2_downsampled = resample(df_class2, replace=False, n_samples=class_sample_size, random_state=123)

    df_downsampled = pd.concat([df_class0_downsampled, df_class1_downsampled, df_class2_downsampled])
    df_downsampled = df_downsampled.sample(frac=1, random_state=123).reset_index(drop=True)

    dataset_downsampled = Dataset.from_pandas(df_downsampled, preserve_index=False)

    return dataset_downsampled


def assign_sentence_id(dataset):

    df = dataset.to_pandas()
    id_column = range(1, len(df) + 1)
    df.insert(0, 'Sentence_ID', id_column)
    dataset = Dataset.from_pandas(df)

    return dataset


def remove_broken_sentences(dataset, broken_sentence_ids):

    df = dataset.to_pandas()

    # Filter the DataFrame to exclude rows with Sentence_IDs found in list_broken_sentence_ids
    filtered_df = df[~df['Sentence_ID'].isin(broken_sentence_ids)]

    dataset = Dataset.from_pandas(filtered_df, preserve_index=False)
    return dataset
