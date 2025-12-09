import torch
from PIL import Image
from difflib import get_close_matches

# Import the shared model and processor from the baseline script
from src.baseline import model, processor, device, inference_baseline
from src.ocr_utils import get_ocr_text

def inference_baseline_with_confidence(image_path, question):
    """
    Baseline inference that also returns a confidence score.
    """
    image = Image.open(image_path).convert("RGB")
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
      
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    
    all_token_probs = []
    if outputs.scores is not None:
        for score_tensor in outputs.scores:
            probs = torch.softmax(score_tensor, dim=-1)
            max_prob = torch.max(probs, dim=-1).values
            all_token_probs.append(max_prob)
    
    if all_token_probs:
        confidence = torch.cat(all_token_probs).mean().item()
    else:
        confidence = 0.0

    answer = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    return answer, confidence

def inference_with_ocr_prompt(image_path, question):
    """
    Generates an answer using OCR text prepended to the prompt.
    """
    image = Image.open(image_path).convert("RGB")
    ocr_text, _ = get_ocr_text(image_path)
    
    prompt = f"<s_docvqa><s_ocr>{ocr_text}</s_ocr><s_question>{question}</s_question><s_answer>"
    
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Let the tokenizer handle truncation safely
    max_decoder_length = model.config.decoder.max_position_embeddings - 3
    decoder_input_ids = processor.tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=max_decoder_length,
        truncation=True
    ).input_ids.to(device)

    # Final safeguard against any invalid token IDs
    vocab_size = model.config.decoder.vocab_size
    clamped_decoder_input_ids = decoder_input_ids.clamp(max=vocab_size - 1)

    try:
        # Explicitly check for invalid token IDs before model.generate
        for i, token_id in enumerate(clamped_decoder_input_ids[0]):
            if token_id >= vocab_size:
                print(f"!!! FATAL (PRE-GENERATE): Invalid token ID found at index {i} !!!")
                print(f"Token ID: {token_id}, Vocab Size: {vocab_size}")
                # Try to decode them
                context_window = 10
                start = max(0, i - context_window)
                end = min(len(clamped_decoder_input_ids[0]), i + context_window + 1)
                print(f"Surrounding token IDs: {clamped_decoder_input_ids[0][start:end]}")
                try:
                    decoded_context = processor.tokenizer.decode(clamped_decoder_input_ids[0][start:end])
                    print(f"Decoded context: {decoded_context}")
                except Exception as decode_e:
                    print(f"Could not decode surrounding tokens: {decode_e}")
                raise ValueError(f"Invalid token_id {token_id} detected before generate. Vocab size is {vocab_size}.")

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=clamped_decoder_input_ids,
            max_length=1024,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    except (RuntimeError, ValueError) as e:
        print(f"Error caught during model.generate: {e}")
        print("Debugging information:")
        print(f"  Image path: {image_path}")
        print(f"  Question: {question}")
        print(f"  Decoder input IDs shape: {clamped_decoder_input_ids.shape}")
        print(f"  Problematic input IDs sample: {clamped_decoder_input_ids[0, :min(20, clamped_decoder_input_ids.shape[1])]}")
        raise e
      
    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

def inference_with_confidence_routing(image_path, question, threshold=0.7):
    """
    Routes to OCR-assisted method if baseline confidence is low.
    """
    answer, confidence = inference_baseline_with_confidence(image_path, question)
    
    if confidence < threshold:
        print(f"Confidence ({confidence:.4f}) is below threshold ({threshold}). Switching to OCR-assisted method.")
        answer = inference_with_ocr_prompt(image_path, question)
        
    return answer

def inference_with_post_correction(image_path, question):
    """
    Corrects the baseline prediction using OCR text.
    """
    donut_answer = inference_baseline(image_path, question)
    ocr_text, _ = get_ocr_text(image_path)
    
    corrected_answer = correct_with_ocr(donut_answer, ocr_text)
    return corrected_answer
  
def correct_with_ocr(answer, ocr_text):
    """
    A simple spell-check like correction based on OCR words.
    """
    ocr_words = ocr_text.split()
    corrected_words = []
      
    for word in answer.split():
        matches = get_close_matches(word, ocr_words, n=1, cutoff=0.6)
        if matches:
            corrected_words.append(matches[0])
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)