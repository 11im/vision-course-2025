"""
RAG-based Prompt-Augmented Donut evaluation on English DocVQA dataset
Uses PaddleOCR for GPU-accelerated OCR with device control
Uses embedding-based retrieval to select most relevant OCR segments
"""
import argparse
import json
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR

# Import evaluation utilities
from src.evaluate import calculate_anls

# Initialize model and processor (will be moved to GPU later via device parameter)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# PaddleOCR and embedding model will be initialized in main() with specified GPU


def extract_ocr_text_segments_paddle(image_path):
    """Extract OCR text as individual segments using PaddleOCR"""
    global ocr_reader
    try:
        result = ocr_reader.ocr(image_path, cls=True)

        segments = []
        if result and result[0]:
            for line in result[0]:
                # PaddleOCR format: [bbox, (text, confidence)]
                bbox, (text, conf) = line

                if conf > 0.5 and len(text.strip()) > 0:  # Filter low confidence
                    segments.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': conf
                    })

        return segments
    except Exception as e:
        print(f"Error during PaddleOCR processing: {e}")
        # Reinitialize OCR reader on error to recover from CUDA corruption
        print("Reinitializing PaddleOCR due to error...")
        try:
            ocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=True,
                gpu_mem=500,
                show_log=False,
                use_space_char=True,
                det_model_dir=None,
                rec_model_dir=None,
            )
        except Exception as reinit_error:
            print(f"Failed to reinitialize PaddleOCR: {reinit_error}")
        return []


def retrieve_relevant_ocr_segments(question, ocr_segments, top_k=5):
    """
    RAG-style retrieval: embed question and OCR segments,
    return top-k most similar segments
    """
    if not ocr_segments:
        return ""

    # Embed question
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)

    # Embed all OCR segments
    ocr_texts = [seg['text'] for seg in ocr_segments]
    ocr_embeddings = embedding_model.encode(ocr_texts, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = torch.nn.functional.cosine_similarity(
        question_embedding.unsqueeze(0),
        ocr_embeddings,
        dim=1
    )

    # Get top-k indices
    top_k = min(top_k, len(ocr_segments))
    top_indices = torch.topk(similarities, k=top_k).indices.cpu().numpy()

    # Collect top-k segments
    relevant_texts = [ocr_segments[idx]['text'] for idx in top_indices]

    return " ".join(relevant_texts)


def clean_ocr_text(text):
    """Remove characters that might cause tokenizer issues"""
    import re
    # Keep only alphanumeric, basic punctuation, and spaces
    # This is more aggressive to prevent any tokenizer issues
    cleaned = re.sub(r'[^a-zA-Z0-9\s\.,!?\-\'\"]', ' ', text)
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remove special XML/HTML-like tags that might confuse tokenizer
    cleaned = re.sub(r'<[^>]*>', ' ', cleaned)
    return cleaned.strip()


def inference_with_rag_prompt(image_path, question, device, top_k=5, max_context_length=100):
    """Donut inference with RAG-retrieved OCR context in prompt"""
    image = Image.open(image_path).convert("RGB")

    # Stage 1: OCR
    print("--- Starting OCR Stage ---")
    ocr_segments = extract_ocr_text_segments_paddle(image_path)
    print("--- Finished OCR Stage ---")

    # Stage 2: Embedding and Retrieval
    print("--- Starting Embedding/Retrieval Stage ---")
    relevant_context = retrieve_relevant_ocr_segments(question, ocr_segments, top_k=top_k)
    print("--- Finished Embedding/Retrieval Stage ---")

    # Clean OCR text to prevent tokenizer issues
    relevant_context = clean_ocr_text(relevant_context)

    # Truncate context to max_context_length
    relevant_context = relevant_context[:max_context_length].strip()

    # Create prompt with OCR context if available
    if relevant_context:
        prompt = f"<s_docvqa><s_context>{relevant_context}</s_context><s_question>{question}</s_question><s_answer>"
    else:
        # Fallback to baseline if no OCR
        prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

    # Tokenize and generate
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # Print final prompt used for generation
    print(f"\n--- PROMPT ---")
    print(f"{prompt}")
    print(f"--- END PROMPT ---\n")

    # Stage 3: Generation
    print("--- Starting Generation Stage ---")
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        bad_words_ids=[[processor.tokenizer.unk_token_id]]
    )
    print("--- Finished Generation Stage ---")

    # Decode output
    clean_sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract answer after the question
    if question in clean_sequence:
        answer = clean_sequence.split(question, 1)[-1].strip()
    else:
        answer = clean_sequence.strip()

    return answer


def docvqa_json_to_samples(json_path, image_dir, start_idx=0, end_idx=None):
    """Convert DocVQA JSON to samples"""
    with open(json_path, 'r', encoding='utf-8') as f:
        docvqa_data = json.load(f)

    all_samples = []
    dataset = docvqa_data.get('data', [])

    for item in dataset:
        image_name = os.path.basename(item['image'])
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
    parser.add_argument("--json_path", required=True, help="Path to DocVQA JSON file")
    parser.add_argument("--image_dir", required=True, help="Path to images directory")
    parser.add_argument("--output_file", required=True, help="Path to output JSON file")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for split")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for split")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID for Donut model")
    parser.add_argument("--ocr_gpu", type=int, default=1, help="GPU device ID for PaddleOCR")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top OCR segments to retrieve")
    parser.add_argument("--max_context_length", type=int, default=200, help="Max context length for prompt")
    args = parser.parse_args()

    # Initialize PaddleOCR with GPU support
    global ocr_reader, embedding_model

    print(f"Initializing PaddleOCR on GPU {args.ocr_gpu}...")

    # Set CUDA_VISIBLE_DEVICES for PaddleOCR
    # This must be done before initializing PaddleOCR
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.ocr_gpu)

    ocr_reader = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=True,
        gpu_mem=500,  # 500MB GPU memory
        show_log=False,
        use_space_char=True,
        det_model_dir=None,  # Use default detection model
        rec_model_dir=None,  # Use default recognition model
    )

    # Restore original CUDA_VISIBLE_DEVICES
    if original_cuda_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

    # Initialize embedding model (CPU is fine for this)
    print("Initializing embedding model (SentenceTransformer)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Set Donut model to specified GPU
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Using device for Donut: {device}")
    print(f"Using GPU {args.ocr_gpu} for PaddleOCR")
    print(f"Top-k OCR segments: {args.top_k}")
    print(f"Max context length: {args.max_context_length}")
    print(f"Loading samples from {args.json_path}")

    samples = docvqa_json_to_samples(args.json_path, args.image_dir, args.start_idx, args.end_idx)

    results = {
        'predictions': [],
        'ground_truths': [],
        'anls_scores': [],
        'inference_times': []
    }

    # Track CUDA errors to prevent cascading failures
    cuda_error_occurred = False

    for i, sample in enumerate(samples):
        image_path = sample['image_path']
        question = sample['question']
        ground_truths = sample['answers']

        # Skip remaining samples if CUDA error occurred (device unusable)
        if cuda_error_occurred:
            print(f"  Sample {i+1}/{len(samples)}: Skipped (CUDA device corrupted)")
            prediction = "[SKIPPED]"
            inference_time = 0
            anls_score = 0.0
        else:
            try:
                start_time = time.time()
                prediction = inference_with_rag_prompt(
                    image_path,
                    question,
                    device,
                    top_k=args.top_k,
                    max_context_length=args.max_context_length
                )
                inference_time = time.time() - start_time

                # Calculate ANLS against all possible ground truths and take the average
                anls_scores = [calculate_anls(prediction, gt) for gt in ground_truths]
                anls_score = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0

            except Exception as e:
                error_msg = str(e)
                print(f"  Sample {i+1}/{len(samples)}: Error - {error_msg}")

                # Check if it's a CUDA error that corrupts the device
                if "CUDA error" in error_msg or "device-side assert" in error_msg:
                    print(f"  !!! CUDA device corrupted at sample {i+1}. Marking all remaining samples as skipped.")
                    cuda_error_occurred = True

                prediction = "[ERROR]"
                inference_time = 0
                anls_score = 0.0

        results['predictions'].append(prediction)
        results['ground_truths'].append(ground_truths)
        results['anls_scores'].append(anls_score)
        results['inference_times'].append(inference_time)

        if (i + 1) % 100 == 0:
            # Calculate running average ANLS
            running_avg_anls = sum(results['anls_scores']) / len(results['anls_scores'])
            print(f"  Processed {i+1}/{len(samples)} samples")
            print(f"    Last ANLS: {anls_score:.4f}, Running Avg ANLS: {running_avg_anls:.4f}, Time: {inference_time:.2f}s")

            # Periodic GPU memory cleanup to prevent accumulation
            torch.cuda.empty_cache()
            print(f"    [GPU memory cache cleared]")

    # Calculate averages
    avg_anls = sum(results['anls_scores']) / len(results['anls_scores'])
    avg_time = sum(results['inference_times']) / len(results['inference_times'])
    results['average_anls'] = avg_anls
    results['average_inference_time'] = avg_time
    results['total_samples'] = len(samples)

    print(f"\n{'='*60}")
    print(f"RAG-based (PaddleOCR) Average ANLS: {avg_anls:.4f}")
    print(f"Average Inference Time: {avg_time:.2f}s")
    print(f"Total Samples: {len(samples)}")
    print(f"{'='*60}")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
