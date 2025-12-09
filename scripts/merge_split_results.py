"""
Merges two JSON result files from split dataset runs into a single file.
"""
import json
import argparse
import os

def merge_results(file1, file2, output_file):
    """
    Merges two JSON result files.

    Args:
        file1 (str): Path to the first JSON result file.
        file2 (str): Path to the second JSON result file.
        output_file (str): Path to save the merged JSON result file.
    """
    print(f"Merging {file1} and {file2} into {output_file}")

    # Load the two result files
    try:
        with open(file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from input file - {e}")
        return

    # Combine the lists from both files
    merged_data = {
        'predictions': data1.get('predictions', []) + data2.get('predictions', []),
        'ground_truths': data1.get('ground_truths', []) + data2.get('ground_truths', []),
        'anls_scores': data1.get('anls_scores', []) + data2.get('anls_scores', []),
        'inference_times': data1.get('inference_times', []) + data2.get('inference_times', [])
    }

    # Recalculate averages and totals
    total_samples = len(merged_data['predictions'])
    average_anls = sum(merged_data['anls_scores']) / total_samples if total_samples > 0 else 0
    average_inference_time = sum(merged_data['inference_times']) / total_samples if total_samples > 0 else 0

    merged_data['total_samples'] = total_samples
    merged_data['average_anls'] = average_anls
    merged_data['average_inference_time'] = average_inference_time

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the merged data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully merged results. Total samples: {total_samples}, Avg ANLS: {average_anls:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge two donut-ocr-assisted result JSON files.")
    parser.add_argument('file1', type=str, help='Path to the first input JSON file.')
    parser.add_argument('file2', type=str, help='Path to the second input JSON file.')
    parser.add_argument('output_file', type=str, help='Path for the merged output JSON file.')
    args = parser.parse_args()

    merge_results(args.file1, args.file2, args.output_file)