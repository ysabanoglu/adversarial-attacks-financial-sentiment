import json
import numpy as np
import openattack_recipes
import victim_sentiment_models
from OpenAttack import metric
from OpenAttack import AttackEval
import os
import time
from datetime import datetime


class JSONEncoderWithFloat32(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)

        return json.JSONEncoder.default(self, obj)




def wrap_attack(model_code, recipe_code):
    victim, wrapper, model, tokenizer = victim_sentiment_models.initialize_model(model_code)
    attacker = openattack_recipes.initialize_recipe(recipe_code)

    quality_metrics = [metric.SemanticSimilarity(), metric.Fluency(),
                       metric.GrammaticalErrors(),
                       metric.ModificationRate()]

    attack_wrapper = AttackEval(attacker, victim, metrics=quality_metrics)

    return attack_wrapper

def run_attacks(attacks, models, data_code, dir_attack_results):
    for attack in attacks:
        for model in models:
            result_code = f"{attack}_vs_{model}_{data_code}"
            file_exists = any(filename.startswith(result_code) and filename.endswith('.json') for filename in os.listdir(dir_attack_results))

            if file_exists:
                print(f"A results file for {attack} against {model} already exists in attack results directory. Skipping this attack.")
                continue  # Skip to the next iteration of the inner loop

            print(f"{attack} starts against {model}")
            attack_iterator = create_attack_iterator(model, attack, data_code)
            run_attack_on_iterator(attack_iterator, model, attack, data_code, dir_attack_results)

def create_attack_iterator(model_code, attack_code, data_code):
    attack_wrapper = wrap_attack(model_code, attack_code)
    dataset = openattack_recipes.dataset_for_recipes(data_code)
    attack_iterator = attack_wrapper.ieval(dataset)

    return attack_iterator

def run_attack_on_iterator(attack_iterator, model_code, attack_code, data_code, dir_attack_results, total_samples=1020):
    # Combine the codes to form the filename result code part
    result_code = f"{attack_code}_vs_{model_code}_{data_code}"

    # Check if a file with the result code already exists
    for filename in os.listdir(dir_attack_results):
        if filename.startswith(result_code) and filename.endswith('.json'):
            print(f"Results file starting with {result_code} already exists in {dir_attack_results}. Skipping processing.")
            return

    results_dict = {}
    processed_samples = 0
    start_time = time.time()

    for sample in attack_iterator:
        sentence_id = sample['data']['Sentence_ID']
        # Extract success, result, and metrics
        success = sample['success']
        result = sample['result']
        metrics = sample['metrics']
        # Store these in a nested dictionary under the Sentence_ID key
        results_dict[sentence_id] = {'success': success, 'result': result, 'metrics': metrics}

        processed_samples += 1

        if processed_samples % 10 == 0:
            # Calculate the total elapsed time so far in seconds
            total_elapsed_time = time.time() - start_time
            # Estimate the average time per sample in seconds
            average_time_per_sample = total_elapsed_time / processed_samples
            # Estimate the remaining time in seconds
            estimated_remaining_time = average_time_per_sample * (total_samples - processed_samples)
            # Convert times to minutes and hours for readability
            total_time_in_minutes = (total_elapsed_time + estimated_remaining_time) / 60
            remaining_time_in_minutes = estimated_remaining_time / 60
            total_time_in_hours = total_time_in_minutes / 60
            remaining_time_in_hours = remaining_time_in_minutes / 60

            print(
                f"Step {processed_samples}/{total_samples}. Estimated Total Time: {total_time_in_minutes:.2f} minutes ({total_time_in_hours:.2f} hours), Estimated Remaining Time: {remaining_time_in_minutes:.2f} minutes ({remaining_time_in_hours:.2f} hours)")

    # Format the current date and time for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    filename = f"{result_code}_{current_time}.json"
    filepath = os.path.join(dir_attack_results, filename)

    # Save the results_dict to a JSON file using the custom encoder
    with open(filepath, 'w') as file:
        json.dump(results_dict, file, cls=JSONEncoderWithFloat32, indent=4)

    print(f"Results saved to {filepath}")

