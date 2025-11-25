# TransformerOCR

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.8%2B-blue)](#)

Lightweight Transformer-based OCR example implemented as Jupyter notebooks. The project contains code to prepare a dataset, train/evaluate a Transformer OCR model, and a pretrained Keras checkpoint for inference.

Table of Contents
- [Files](#files)
- [Dataset](#dataset)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Pretrained model](#pretrained-model)
- [Requirements](#requirements)
- [Notes & caveats](#notes--caveats)
- [Contributing](#contributing)
- [License](#license)

Files

- `transformerMainScript.ipynb` — main notebook with model definition, training loop, evaluation and inference examples.
- `corpuscreator.ipynb` — notebook to build and preprocess the training corpus (image/text pairs and tokenization).
- `pretrained model/` — directory containing a saved Keras model:
  - `best_transformerocrceer1.h5` — pretrained checkpoint produced by the training notebook.

Dataset

- This project uses the IAM Handwriting Database (IAM) word-level dataset as the primary training and evaluation corpus. The IAM words dataset contains on the order of 100,000+ labeled word images (roughly 115,000 word samples in the public IAM words split), making it suitable for training sequence-to-sequence OCR models.

- The notebooks reference the typical IAM `words.txt` annotation file and expect the dataset folder layout used by the IAM dataset (see `transformerMainScript.ipynb` where `WORDS_TXT` and `DATA_PATH` are defined). If you're using a different dataset or a custom layout, update these path constants in the notebook before running.

Quickstart

1. Clone the repository and open a terminal in the project root.

2. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies. If you don't have a `requirements.txt` yet, install common packages used by the notebooks:

```powershell
pip install --upgrade pip
pip install tensorflow numpy pandas pillow opencv-python scikit-image matplotlib h5py tqdm
```

4. Start Jupyter and open the notebooks:

```powershell
jupyter notebook
```

Usage

- Run `corpuscreator.ipynb` first to prepare or inspect the dataset and create tokenizers/encoders used by the model.
- Then run `transformerMainScript.ipynb`. Typical cells:
  - model building (encoder/decoder Transformer blocks)
  - training/evaluation (ModelCheckpoint may write .h5 files)
  - inference examples showing how to convert model outputs to text.

transformerMainScript Explanation

The `transformerMainScript.ipynb` notebook is the heart of the project. Below is a breakdown of its major sections and components:

**1. Data Loading & Preprocessing (Cell 1)**
- Loads the IAM dataset from `words.txt` annotation file and associated word image files.
- Applies image preprocessing:
  - Resizes images to **128 × 32** pixels (standard for word-level OCR).
  - Converts to grayscale and inverts (text becomes foreground=1, background=0).
  - Normalizes pixel values to [0, 1] range.
- Splits data into training (90%) and validation (10%) sets.
- Creates label encodings using a predefined charset (printable ASCII + digits + punctuation).
- Pads labels to a maximum sequence length for CTC training.

**2. Model Architecture (Cell 2)**
The model follows a CNN-Transformer-CTC pipeline:

- **CNN Backbone** (6 Conv2D layers with BatchNorm and MaxPooling):
  - Progressively reduces spatial dimensions (128×32 → ~31×1 time steps).
  - Extracts visual features from the image.
  - Output: tensor of shape `[batch, ~31 timesteps, 256 channels]`.

- **Positional Encoding Layer**:
  - Adds absolute positional information to each time step (sine/cosine encoding).
  - Allows the Transformer to learn sequence order.

- **Transformer Encoder** (configurable, typically 2 layers):
  - Each layer has Multi-Head Self-Attention (8 heads, 256 dims).
  - Feed-Forward Network (FFN) with 512 hidden units and dropout (0.2).
  - LayerNorm applied after each sub-layer (residual connections).
  - Attends over the entire sequence to capture global context.

- **CTC Decoder Output**:
  - Dense layer projecting to `[batch, timesteps, num_chars+1]`.
  - Softmax activation over the character vocabulary.
  - CTC loss handles variable-length alignment between input and text.

**3. Training Setup (Cell 2 continued)**
- **Loss Function**: CTC (Connectionist Temporal Classification) via a custom lambda layer.
- **Optimizer**: Adam with learning rate 1e-4.
- **Callbacks**:
  - `ModelCheckpoint`: saves best model by validation CER (Character Error Rate).
  - `EarlyStopping`: stops training if CER doesn't improve for 6 epochs.
  - `ReduceLROnPlateau`: reduces learning rate by 0.5× if CER plateaus (patience=5).
  - `EvaluateCER_WER`: custom callback that decodes predictions and computes per-epoch CER/WER using pyctcdecode.

**4. Evaluation Metrics (Cell 5)**
- **Character Error Rate (CER)**: fraction of character-level edits (Levenshtein distance).
- **Word Error Rate (WER)**: fraction of word-level edits.
- **Exact Match Accuracy**: fraction of images where prediction matches ground truth exactly.
- Uses the pyctcdecode library with a language model (from `corpus.txt`) to improve decoding.

**5. Inference & Decoding (Cells 6–8)**
- Loads the trained or pretrained model.
- Runs predictions on validation images.
- Decodes CTC logits to text using beam search (width=10) and optional LM.
- Displays sample predictions, error analysis (character confusion matrix), and overall metrics.

**Key Hyperparameters**
- Image size: 128×32 pixels
- Charset size: ~76 characters (printable ASCII)
- Transformer layers: 2 (configurable via `NUM_TRANSFORMER_LAYERS`)
- Model dimension: 256 (`D_MODEL`)
- Attention heads: 8 (`NUM_HEADS`)
- Feed-forward dimension: 512 (`DFF`)
- Dropout: 0.2
- Batch size: 16 (training)
- Learning rate: 1e-4 (Adam)
- Max training epochs: 100 (with early stopping)

**Data Augmentation**
- Random rotation (factor=0.02, fill with 0) is applied during training via `RandomRotation` layer (active only in training mode).

**Output Files**
- `best_transformerocrceer1.h5` — best model checkpoint (monitored by CER).
- `modelc.h5` — final model checkpoint (optional, Cell 3).
- `cer_wer_historycer.pkl` — pickle file with CER/WER history across epochs (Cell 4).

Pretrained model

To load the included checkpoint in Python (adjust if the notebook defines custom layers):

```python
from tensorflow import keras

# point to the file in the repo root
model = keras.models.load_model(r"pretrained model\best_transformerocrceer1.h5", compile=False)

# prepare your image(s) using the exact preprocessing in `corpuscreator.ipynb`
# then run model.predict() and decode token indices to text using the notebook's tokenizer
```

Requirements

- Python 3.8 or later
- TensorFlow 2.x (or tensorflow-gpu if you have compatible GPU/drivers)
- numpy, pandas, pillow, opencv-python, scikit-image, matplotlib, h5py, tqdm

Notes & caveats

- The notebooks in this repository are not executed in the repo snapshot; open and run the first cells to confirm exact imports, tokenizers, input image shapes, and any custom objects required to load the model.
- If the model was saved with custom layers or functions, pass a matching `custom_objects` dict to `keras.models.load_model` when loading the `.h5` file.
- Confirm the filename `best_transformerocrceer1.h5` in `pretrained model/` matches what the training notebook writes.

Contributing

- Issues and PRs are welcome. For small changes (README, minor fixes), push a branch and open a PR describing the change.


