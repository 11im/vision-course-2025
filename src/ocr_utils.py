import easyocr

# Initialize the reader. This will download the models for the specified languages.
# It's recommended to do this once and reuse the reader object.
reader = easyocr.Reader(['en'], gpu=True) 

def get_ocr_text(image_path: str):
    """
    Extracts text from an image using easyocr.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple containing:
        - A single string of all extracted text, joined by spaces.
        - The raw results from easyocr, which include bounding boxes and confidence scores.
    """
    try:
        results = reader.readtext(image_path)
        text = " ".join([res[1] for res in results])
        return text, results
    except Exception as e:
        print(f"Error during EasyOCR processing: {e}")
        return "", []

# Example of how you might get bounding boxes and other info from the raw results
def get_ocr_words_with_boxes(ocr_results):
    """
    Processes the raw output from easyocr to get a list of words and their bounding boxes.
    """
    word_data = []
    for (bbox, text, conf) in ocr_results:
        # bbox is a list of 4 points (top-left, top-right, bottom-right, bottom-left)
        # For simplicity, we can convert it to a simpler x, y, w, h format if needed.
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