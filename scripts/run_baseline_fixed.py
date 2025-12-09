import argparse
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import only what we need (avoid importing full src package)
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Import evaluation utilities
from src.evaluate import calculate_anls

# Initialize model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

def inference_baseline(image_path, question, device):
    image = Image.open(image_path).convert("RGB")
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    # Decode with skip_special_tokens=True gives: "question answer"
    clean_sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract answer after the question
    if question in clean_sequence:
        # Split by question and take everything after
        answer = clean_sequence.split(question, 1)[-1].strip()
    else:
        # Fallback: use the whole sequence
        answer = clean_sequence.strip()

    return answer

def docvqa_json_to_samples(json_path, image_dir, start_idx=0, end_idx=None):
    """Convert DocVQA JSON to samples"""
    with open(json_path, 'r', encoding='utf-8') as f:
        docvqa_data = json.load(f)

    all_samples = []
    dataset = docvqa_data.get('data', [])

    for item in dataset:
        # item['image'] is like 'documents/pybv0228_81.png'
        # Actual path is data/docvqa/images/val/pybv0228_81.png
        image_name = os.path.basename(item['image'])  # Get filename only
        image_path = os.path.join(image_dir, 'images', 'val', image_name)

        all_samples.append({
            "image_path": image_path,
            "question": item['question'],
            "answers": item['answers']
        })

    if end_idx is None:
        end_idx = len(all_samples)

    samples = all_samples[start_idx:end_idx]

    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Processing samples from index {start_idx} to {end_idx}")
    print(f"Samples to evaluate in this split: {len(samples)}")

    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Using device: {device}")
    print(f"Loading samples from {args.json_path}")

    samples = docvqa_json_to_samples(args.json_path, args.image_dir, args.start_idx, args.end_idx)

    results = {
        'predictions': [],
        'ground_truths': [],
        'anls_scores': [],
        'inference_times': []
    }

    import time

    for i, sample in enumerate(samples):
        image_path = sample['image_path']
        question = sample['question']
        ground_truths = sample['answers']

        try:
            start_time = time.time()
            prediction = inference_baseline(image_path, question, device)
            inference_time = time.time() - start_time

            accuracies = [calculate_anls(prediction, gt) for gt in ground_truths]
            max_accuracy = max(accuracies)

        except Exception as e:
            print(f"  Sample {i+1}/{len(samples)}: Error - {e}")
            prediction = "[ERROR]"
            inference_time = 0
            max_accuracy = 0.0

        results['predictions'].append(prediction)
        results['ground_truths'].append(ground_truths)
        results['anls_scores'].append(max_accuracy)
        results['inference_times'].append(inference_time)

        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{len(samples)}: ANLS={max_accuracy:.4f}, Time={inference_time:.2f}s")

    # Calculate averages
    avg_anls = sum(results['anls_scores']) / len(results['anls_scores'])
    avg_time = sum(results['inference_times']) / len(results['inference_times'])
    results['average_anls'] = avg_anls
    results['average_inference_time'] = avg_time

    print(f"\n{'='*60}")
    print(f"Average ANLS: {avg_anls:.4f}")
    print(f"Average Time: {avg_time:.2f}s")
    print(f"{'='*60}")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
