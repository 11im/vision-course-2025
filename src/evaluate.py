from collections import defaultdict
import time
import json
from Levenshtein import distance

def calculate_anls(prediction: str, ground_truth: str) -> float:
    """
    Calculates the Average Normalized Levenshtein Similarity (ANLS).
    This is the primary metric for the DocVQA dataset.
    """
    if not ground_truth:
        return 1.0 if not prediction else 0.0
    
    lev_dist = distance(prediction.lower(), ground_truth.lower())
    max_len = max(len(prediction), len(ground_truth))
    
    if max_len == 0:
        return 1.0 # Both strings are empty

    anls = 1.0 - (lev_dist / max_len)
    return max(0.0, anls)

def evaluate_method(method_fn, test_samples: list) -> dict:
    """
    Evaluates a given inference method on a set of test samples.

    Args:
        method_fn: The inference function to evaluate (e.g., `inference_baseline`).
        test_samples: A list of dictionaries, where each dict contains 
                      'image_path', 'question', and 'answers' (a list of strings).

    Returns:
        A dictionary containing the evaluation results.
    """
    results = defaultdict(list)
    
    print(f"Evaluating method: {method_fn.__name__} on {len(test_samples)} samples...")

    for i, sample in enumerate(test_samples):
        image_path = sample['image_path']
        question = sample['question']
        ground_truths = sample['answers']

        try:
            # Measure inference time
            start_time = time.time()
            prediction = method_fn(image_path, question)
            inference_time = time.time() - start_time
            
            # Calculate ANLS against all possible ground truths and take the max
            accuracies = [calculate_anls(prediction, gt) for gt in ground_truths]
            max_accuracy = max(accuracies)
        
        except RuntimeError as e:
            print(f"  Sample {i+1}/{len(test_samples)}: Encountered a RuntimeError. Skipping.")
            print(f"  Error: {e}")
            prediction = "[ERROR]"
            inference_time = 0
            max_accuracy = 0.0
            # Log the problematic image path to a file
            with open("error_log.txt", "a") as f:
                f.write(image_path + "\n")

        results['predictions'].append(prediction)
        results['ground_truths'].append(ground_truths)
        results['anls_scores'].append(max_accuracy)
        results['inference_times'].append(inference_time)

        if prediction != "[ERROR]":
            print(f"  Sample {i+1}/{len(test_samples)}: ANLS={max_accuracy:.4f}, Time={inference_time:.2f}s")
        
    # Calculate and add aggregate stats
    avg_anls = sum(results['anls_scores']) / len(results['anls_scores']) if results['anls_scores'] else 0
    avg_time = sum(results['inference_times']) / len(results['inference_times']) if results['inference_times'] else 0
    results['average_anls'] = avg_anls
    results['average_inference_time'] = avg_time

    print(f"  Average ANLS: {avg_anls:.4f}")
    print(f"  Average Time: {avg_time:.2f}s")
    
    return dict(results)

def save_results(results: dict, filepath: str):
    """Saves evaluation results to a JSON file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {filepath}")

# This import should be here to avoid circular dependency if you run this file directly for testing
import os