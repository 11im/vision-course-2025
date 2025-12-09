# OCR-Assisted Donut for Document VQA

## Overview
This project evaluates whether adding OCR information improves the pre-trained Donut model on document question answering. We test if Donut's "OCR-free" approach is sufficient or if OCR provides meaningful benefits for certain document types.

## Motivation
Donut claims to avoid OCR entirely, eliminating computational costs and error propagation from traditional OCR pipelines. However, the original paper hints at limitations with small fonts, low resolution, and complex layouts. We empirically test when OCR helps and whether the performance gains justify the added cost.

## Methods
We test three OCR-assisted approaches against baseline Donut on DocVQA validation set (5,349 samples):

1. **Post-Correction**: Run baseline Donut, then correct the answer using OCR text (spell-check style matching)
2. **RAG (Retrieval-Augmented)**: Select top-K most relevant OCR segments using semantic similarity (SentenceTransformer `all-MiniLM-L6-v2`), add to prompt
3. **Random Sampling**: Randomly select OCR segments and add to prompt (control for RAG)

We use PaddleOCR for text extraction and measure performance with ANLS (Average Normalized Levenshtein Similarity) and inference time.

## Data Setup

Download the DocVQA dataset and organize it as follows:

```
vision/
├── data/docvqa/
│   ├── annotations/val_v1.0_withQT.json
│   └── images/val/                    # 5,349 validation images
└── donut-ocr-assisted/
    ├── logs/                          # Experiment logs
    └── results/                       # JSON results by method
        ├── baseline_fixed/
        ├── post_correction_paddle/
        ├── prompt_augmented_rag_paddle/
        └── prompt_augmented_random_paddle/
```

The dataset is available from the official DocVQA challenge or Hugging Face.

## Running Experiments

### Docker Setup (Recommended)

Build the Docker image:
```bash
cd ~/vision
docker build -t donut_experiment -f donut-ocr-assisted/Dockerfile .
```
### Run Individual Experiments

Each experiment processes the dataset in 50-sample chunks (107 chunks total):

```bash
# Baseline on GPU 0
docker run --gpus '"device=0"' --rm -v /home/jhlim/vision:/work -w /work donut_experiment \
  bash donut-ocr-assisted/run_baseline.sh

# Post-Correction on GPU 0
docker run --gpus '"device=0"' --rm -v /home/jhlim/vision:/work -w /work donut_experiment \
  bash donut-ocr-assisted/run_post_correction.sh

# RAG on GPU 1
docker run --gpus '"device=1"' --rm -v /home/jhlim/vision:/work -w /work donut_experiment \
  bash donut-ocr-assisted/run_rag.sh

# Random Sampling on GPU 1
docker run --gpus '"device=1"' --rm -v /home/jhlim/vision:/work -w /work donut_experiment \
  bash donut-ocr-assisted/run_random.sh
```

The resume scripts automatically detect completed chunks and only process remaining samples. They also clean up corrupted chunk files from CUDA errors.

Results include prompts, predictions, and ANLS scores for all four methods side-by-side.

## Key Implementation Details

- **Model**: `naver-clova-ix/donut-base-finetuned-docvqa` (pre-trained)
- **OCR Engine**: PaddleOCR with GPU acceleration
- **Embedding Model**: SentenceTransformer `all-MiniLM-L6-v2` for RAG
- **Chunked Processing**: 50 samples per chunk to handle CUDA memory issues
- **Error Handling**: Automatic detection of CUDA errors and corrupted chunks
- **Resume Capability**: All experiments can resume from where they left off

### Prompt Format

```
<s_docvqa><s_context>{ocr_text}</s_context><s_question>{question}</s_question><s_answer>
```

For baseline Donut (no OCR), the `<s_context>` section is omitted.

## Citation

If you use this code, please cite the original Donut paper:

```bibtex
@inproceedings{kim2022donut,
    title={{Don't} Overlook the {Obvious}: {OCR-free} Document Understanding Transformer},
    author={Kim, Geewook and Lee, Teakgyu and Park, Bado and Kim, Jinu and Kim, Daehyun and
            Hwang, Young-Bin and Yoo, Inkwon and Kim, Jinbeom and Kim, Jung-Woo and
            Park, Jaejun and Lee, Nayeon and Park, Seungbeom},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```