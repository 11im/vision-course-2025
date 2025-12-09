"""
OCR utilities using PaddleOCR for GPU-accelerated text extraction
"""
from paddleocr import PaddleOCR
import os

# Initialize PaddleOCR reader
# This can be overridden by setting GPU via environment variables before import
reader = None

def init_paddle_ocr(gpu_id=None):
    """
    Initialize PaddleOCR with optional GPU specification

    Args:
        gpu_id: GPU device ID (e.g., 0, 1). If None, uses GPU by default.
    """
    global reader

    if gpu_id is not None:
        # Set CUDA_VISIBLE_DEVICES for specific GPU
        original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    reader = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=True,
        gpu_mem=500,  # 500MB GPU memory
        show_log=False,
        use_space_char=True
    )

    if gpu_id is not None:
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
        else:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']


def get_ocr_text(image_path: str):
    """
    Extracts text from an image using PaddleOCR.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple containing:
        - A single string of all extracted text, joined by spaces.
        - The raw results from PaddleOCR, which include bounding boxes and confidence scores.
    """
    global reader

    # Initialize reader if not already done
    if reader is None:
        init_paddle_ocr()

    try:
        result = reader.ocr(image_path, cls=True)

        texts = []
        raw_results = []

        if result and result[0]:
            for line in result[0]:
                # PaddleOCR format: [bbox, (text, confidence)]
                bbox, (text, conf) = line
                texts.append(text)
                raw_results.append((bbox, text, conf))

        text = " ".join(texts)
        return text, raw_results

    except Exception as e:
        print(f"Error during PaddleOCR processing: {e}")
        return "", []


def get_ocr_words_with_boxes(ocr_results):
    """
    Processes the raw output from PaddleOCR to get a list of words and their bounding boxes.

    Args:
        ocr_results: Raw results from PaddleOCR [(bbox, text, conf), ...]

    Returns:
        List of dicts with 'text', 'confidence', and 'box' keys
    """
    word_data = []

    for (bbox, text, conf) in ocr_results:
        # bbox is a list of 4 points (top-left, top-right, bottom-right, bottom-left)
        # For simplicity, we can convert it to a simpler x, y, w, h format
        top_left = bbox[0]
        bottom_right = bbox[2]
        x = top_left[0]
        y = top_left[1]
        w = bottom_right[0] - top_left[0]
        h = bottom_right[1] - top_left[1]

        word_data.append({
            'text': text,
            'confidence': conf,
            'box': [x, y, w, h]
        })

    return word_data
