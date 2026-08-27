import pandas as pd
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util


def calculate_quality_metrics(dataset, model_code):
    df = dataset.to_pandas()

    # Extract all unique attack codes from the Bool_Success_ columns
    attack_codes = [col.split('_')[2] for col in df.columns if col.startswith('Bool_Success_')]

    # Initialize an empty list to hold the results for each attack code
    results = []

    # Loop over each attack code to calculate metrics
    for attack_code in attack_codes:
        use_similarity = avg_USE_similarity(df, attack_code)
        fluency = avg_fluency(df, attack_code)
        grammatical_errors = avg_grammatical_errors(df, attack_code)
        word_modification_rate = avg_word_modification_rate(df, attack_code)
        # Calculate METEOR score
        meteor_score = avg_METEOR_score(df, attack_code)
        #Calculate MiniLM Score average
        minilm_score = avg_MiniLM_score(df,attack_code)

        # Append a dictionary with the results for the current attack code to the results list
        results.append({
            'Attack vs Model (Adv. Samples)': f'{attack_code} vs {model_code}',
            'USE Similarity': use_similarity,
            'MiniLM Similarity': minilm_score,
            'METEOR Score': meteor_score,
            'Fluency (GPT-2 perplexity)': fluency,
            'Grammatical Errors': grammatical_errors,
            'Word Modification Rate': word_modification_rate,
        })

    # Convert the list of dictionaries into a pandas DataFrame and set the index
    results_df = pd.DataFrame(results)
    results_df.set_index('Attack vs Model (Adv. Samples)', inplace=True)
    results_df = results_df.sort_values(by='USE Similarity', ascending=False)

    return results_df


def avg_USE_similarity(df, attack_code):
    success_filter = df[f'Bool_Success_{attack_code}'] == True
    filtered_df = df[success_filter]
    similarity_values = filtered_df[f'Metrics_{attack_code}'].apply(
        lambda metrics: extract_and_check_metric(metrics, 'Semantic Similarity', attack_code))

    average_similarity = similarity_values.mean()
    return round(average_similarity, 3)


def avg_fluency(df, attack_code):
    success_filter = df[f'Bool_Success_{attack_code}'] == True
    filtered_df = df[success_filter]
    fluency_values = filtered_df[f'Metrics_{attack_code}'].apply(
        lambda metrics: extract_and_check_metric(metrics, 'Fluency (ppl)', attack_code))
    average_fluency = fluency_values.mean()

    return int(round(average_fluency, 0))


def avg_grammatical_errors(df, attack_code):
    success_filter = df[f'Bool_Success_{attack_code}'] == True
    filtered_df = df[success_filter]
    error_values = filtered_df[f'Metrics_{attack_code}'].apply(
        lambda metrics: extract_and_check_metric(metrics, 'Grammatical Errors', attack_code))
    average_errors = error_values.mean()

    return round(average_errors, 1)


def avg_word_modification_rate(df, attack_code):
    success_filter = df[f'Bool_Success_{attack_code}'] == True
    filtered_df = df[success_filter]
    mod_rate_values = filtered_df[f'Metrics_{attack_code}'].apply(
        lambda metrics: extract_and_check_metric(metrics, 'Word Modif. Rate', attack_code))
    average_mod_rate = mod_rate_values.mean()

    return round(average_mod_rate, 2)


def extract_and_check_metric(metrics, key, attack_code):
    # Try to retrieve the value for the specified metric key from the dictionary
    value = metrics.get(key)
    # If the value is None (indicating the key was not found), print a warning and return 0
    if value is None:
        print(f"Warning: Missing '{key}' value detected for attack code {attack_code}. Using default value 0.")

        return 0
    else:
        return value


def avg_METEOR_score(df, attack_code):
    # Apply success filter
    success_filter = df[f'Bool_Success_{attack_code}'] == True
    filtered_df = df[success_filter]

    # Find the column corresponding to the attack code
    column = f'Sentence_{attack_code}'

    # Extract the original and modified sentences
    original_sentences = filtered_df['Sentence_Original']
    modified_sentences = filtered_df[column]

    # Calculate METEOR scores
    meteor_scores = []
    for original, modified in zip(original_sentences, modified_sentences):
        # Preprocess the hypothesis (modified sentence) and reference (original sentence) to tokenize them properly
        original_tokens = original.split()  # Assuming sentences are space-separated
        modified_tokens = modified.split()
        score = meteor_score([original_tokens], modified_tokens)
        meteor_scores.append(score)

    # Calculate the average METEOR score
    if len(meteor_scores) > 0:
        avg_meteor_score = sum(meteor_scores) / len(meteor_scores)

        return round(avg_meteor_score, 3)
    else:

        return  # Return , if there are no successful attacks for this attack code


def avg_MiniLM_score(df, attack_code):
    # Apply success filter
    success_filter = df[f'Bool_Success_{attack_code}'] == True
    filtered_df = df[success_filter]

    # Find the column corresponding to the MiniLM score for the attack code
    score_column = f'MiniLM_Score_{attack_code}'

    # Extract MiniLM scores
    minilm_scores = filtered_df[score_column].tolist()

    # Calculate the average MiniLM score
    if minilm_scores:
        avg_score = sum(minilm_scores) / len(minilm_scores)
        return round(avg_score, 3)
    else:
        # Return None if there are no successful attacks for this attack code
        return None


def calculate_MiniLM_Similarity(dataset):
    # Initialize the model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Function to calculate cosine similarity
    def compute_similarity(original, modified, success):
        if success:
            emb1 = model.encode(original, convert_to_tensor=True)
            emb2 = model.encode(modified, convert_to_tensor=True)
            return util.pytorch_cos_sim(emb1, emb2).item()
        return 1.0  # Return 1.0 if the attack was not successful

    # Iterate over each possible attack code
    attack_codes = [col.split('_')[2] for col in dataset.column_names if col.startswith('Bool_Success_')]
    for attack_code in attack_codes:
        # Define a function that will be applied to each row
        dataset = dataset.map(
            lambda example: {
                f'MiniLM_Score_{attack_code}': compute_similarity(
                    example['Sentence_Original'],
                    example[f'Sentence_{attack_code}'],
                    example[f'Bool_Success_{attack_code}']
                )
            },
            batched=False  # Process row-by-row; set to True for batch processing if your environment supports it
        )

    return dataset
