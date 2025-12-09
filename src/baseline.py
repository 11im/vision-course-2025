import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Initialize model and processor once and share them
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def inference_baseline(image_path, question):
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
    )

    # Decode the full sequence
    full_sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract only the answer part after </s_answer> tag
    # The model generates: <s_docvqa><s_question>Q</s_question><s_answer>A</s_answer>
    if "</s_answer>" in full_sequence:
        answer = full_sequence.split("</s_answer>")[0].split("<s_answer>")[-1].strip()
    else:
        # Fallback: try to extract answer after the question
        if question in full_sequence:
            answer = full_sequence.split(question)[-1].strip()
        else:
            answer = full_sequence

    return answer
