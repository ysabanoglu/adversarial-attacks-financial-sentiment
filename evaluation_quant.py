from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd


def calculate_metrics_for_model(dataset, model_code, ground_truth='Ground_Truth'):

    df = pd.DataFrame(dataset)
    true_labels = df[ground_truth]

    model_predictions = df[f'Sentiment_{model_code}']

    # Calculating metrics
    metrics_for_model = {
        f'Accuracy': accuracy_score(true_labels, model_predictions),
        f'Precision': precision_score(true_labels, model_predictions, average='macro', zero_division=0),
        f'Recall': recall_score(true_labels, model_predictions, average='macro', zero_division=0),
        f'F1-Score': f1_score(true_labels, model_predictions, average='macro', zero_division=0)
    }

    metrics_df = pd.DataFrame(metrics_for_model, index=[model_code])
    metrics_df = metrics_df.round(3)

    return metrics_df


def calculate_metrics_after_attacks(dataset, model_code, average='macro'):

    accuracy_df = calculate_accuracy_after_attacks(dataset)
    precision_df = calculate_precision_after_attacks(dataset, average)
    recall_df = calculate_recall_after_attacks(dataset, average)
    f1_df = calculate_f1_after_attacks(dataset, average)
    success_rate_df = calculate_attack_success_rates(dataset)

    # Ensure all DataFrames use the same index naming convention
    accuracy_df.index = accuracy_df.index.map(lambda x: f'{x} vs {model_code}')
    precision_df.index = precision_df.index.map(lambda x: f'{x} vs {model_code}')
    recall_df.index = recall_df.index.map(lambda x: f'{x} vs {model_code}')
    f1_df.index = f1_df.index.map(lambda x: f'{x} vs {model_code}')
    success_rate_df.index = success_rate_df.index.map(lambda x: f'{x} vs {model_code}')

    # Merge all DataFrames on their index (attack codes)
    merged_df = pd.concat([accuracy_df, precision_df, recall_df, f1_df, success_rate_df], axis=1)

    # Remove any rows where the attack code and model code are the same
    self_comparison_key = f'{model_code} vs {model_code}'
    if self_comparison_key in merged_df.index:
        merged_df = merged_df.drop(self_comparison_key)

    # Sort the DataFrame by 'Accuracy'
    result_df = merged_df.sort_values(by='Accuracy')

    # Rename the index to indicate it represents 'Attack vs Model'
    result_df.index.name = 'Attack vs Model'

    return result_df


def calculate_metric_changes(dataset, model_code, ground_truth='Ground_Truth', average='macro'):

    # Calculate metrics before attacks
    before_metrics_df = calculate_metrics_for_model(dataset, model_code, ground_truth)

    # Calculate metrics after attacks for each attack type
    after_metrics_df = calculate_metrics_after_attacks(dataset, model_code, average)

    asr_col = f'Attack Success Rate'
    if asr_col in after_metrics_df.columns:
        after_metrics_df = after_metrics_df.drop([asr_col], axis=1)

    before_metrics_7_rows = pd.concat([before_metrics_df] * len(after_metrics_df), ignore_index=True)
    before_metrics_7_rows.index = after_metrics_df.index

    # Calculate changes for each metric
    changes_df = after_metrics_df.subtract(before_metrics_7_rows)
    changes_df = changes_df.rename(columns=lambda x: x.replace(model_code, f'Change in {model_code}'))

    changes_df = changes_df.rename(columns=lambda x: f'{x} (Change)')

    changes_df = changes_df.round(3)

    return changes_df


def calculate_attack_success_rates(dataset):

    df = pd.DataFrame(dataset)
    success_rates = {}
    for col in df.columns:
        if col.startswith('Bool_Success_'):
            attack_code = col[len('Bool_Success_'):]
            total_count = len(df[col])
            if total_count == 0:
                success_rate = 0
            else:
                true_count = df[col].sum()
                success_rate = true_count / total_count
                success_rate = round(success_rate, 3)
            success_rates[attack_code] = success_rate

    result_df = pd.DataFrame(list(success_rates.values()), index=success_rates.keys(),
                             columns=['Attack Success Rate'])

    return result_df.sort_values(by='Attack Success Rate', ascending=False)


def calculate_accuracy_after_attacks(dataset):

    df = pd.DataFrame(dataset)
    if 'Ground_Truth' not in df.columns:
        raise ValueError("Dataset must include a 'Ground_Truth' column.")

    accuracies = {}
    for col in df.columns:
        if col.startswith('Sentiment_'):
            attack_code = col[len('Sentiment_'):]
            accuracy = accuracy_score(df['Ground_Truth'], df[col])
            accuracies[attack_code] = round(accuracy, 3)

    result_df = pd.DataFrame(list(accuracies.values()), index=accuracies.keys(),
                             columns=['Accuracy'])

    return result_df.sort_values(by='Accuracy')


def calculate_precision_after_attacks(dataset, average='macro'):

    df = pd.DataFrame(dataset)
    if 'Ground_Truth' not in df.columns:
        raise ValueError("Dataset must include a 'Ground_Truth' column.")

    precisions = {}
    for col in df.columns:
        if col.startswith('Sentiment_'):
            attack_code = col[len('Sentiment_'):]
            precision = precision_score(df['Ground_Truth'], df[col], zero_division=0, average=average)
            precisions[attack_code] = round(precision, 3)

    result_df = pd.DataFrame(list(precisions.values()), index=precisions.keys(),
                             columns=['Precision'])

    return result_df.sort_values(by='Precision')


def calculate_recall_after_attacks(dataset, average='macro'):

    df = pd.DataFrame(dataset)
    if 'Ground_Truth' not in df.columns:
        raise ValueError("Dataset must include a 'Ground_Truth' column.")

    recalls = {}
    for col in df.columns:
        if col.startswith('Sentiment_'):
            attack_code = col[len('Sentiment_'):]
            recall = recall_score(df['Ground_Truth'], df[col], zero_division=0, average=average)
            recalls[attack_code] = round(recall, 3)

    result_df = pd.DataFrame(list(recalls.values()), index=recalls.keys(),
                             columns=['Recall'])

    return result_df.sort_values(by='Recall')


def calculate_f1_after_attacks(dataset, average='macro'):

    df = pd.DataFrame(dataset)
    if 'Ground_Truth' not in df.columns:
        raise ValueError("Dataset must include a 'Ground_Truth' column.")

    f1_scores = {}
    for col in df.columns:
        if col.startswith('Sentiment_'):
            attack_code = col[len('Sentiment_'):]
            f1 = f1_score(df['Ground_Truth'], df[col], zero_division=0, average=average)
            f1_scores[attack_code] = round(f1, 3)

    result_df = pd.DataFrame(list(f1_scores.values()), index=f1_scores.keys(),
                             columns=['F1-Score'])

    return result_df.sort_values(by='F1-Score')


def calculate_accuracy(dataset, model_pred, ground_truth='Ground_Truth'):

    df = dataset.to_pandas()
    truth = df[ground_truth]
    pred = df[model_pred]

    accuracy = accuracy_score(y_true=truth, y_pred=pred)
    accuracy = round(accuracy, 3)

    return accuracy


def decrease_in_accuracy(dataset, model_code, model_pred):

    # Calculate initial accuracy before attacks
    accuracy_before = calculate_accuracy(dataset, 'Ground_Truth', model_pred)

    # Calculate accuracy after attacks based on the model code
    df = calculate_accuracy_after_attacks(dataset, model_code)

    # Assign initial accuracy to all entries

    accuracy_before_attacks = f'Accuracy ({model_code}) before Attack'
    accuracy_after_attacks = f'Accuracy ({model_code})'
    decrease_in_accuracy = f'Accuracy Decrease ({model_code})'

    df[accuracy_before_attacks] = accuracy_before

    # Calculate the decrease in accuracy
    df[decrease_in_accuracy] = df[accuracy_before_attacks] - df[accuracy_after_attacks]

    df_reordered = df[[accuracy_after_attacks, decrease_in_accuracy, accuracy_before_attacks]]

    df_reordered = df_reordered.sort_values(by=accuracy_after_attacks, ascending=True)

    return df_reordered
