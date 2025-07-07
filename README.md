# Depression Detection from Audio

This project uses deep learning to detect signs of depression from audio recordings. It extracts MFCCs, delta, and delta-delta features using Librosa, then classifies them using a fine-tuned BiLSTM model built with TensorFlow.

## Features

- 📁 Processes `.wav` files
- 🧠 Extracts 78 audio features (MFCC + delta + delta-delta)
- 🔄 StandardScaler normalization
- 🎙️ Optionally supports speaker diarization for multi-speaker files
- 🔮 Predicts depression using a trained BiLSTM model
- 🌐 Gradio interface for easy browser-based testing

## Setup

```bash
pip install -r requirements.txt
## Usage

- Prepare your audio file (WAV format, mono preferred).
- Run the notebook or script to test prediction.
- Use the Gradio interface to upload and classify audio.

## Model Info

- Fine-tuned on the DAIC-WOZ dataset.
- Supports testing, training, and fine-tuning.
- Input shape: `(None, 216, 78)`

## Notes

- CREMA-D can be used for pretraining.
- Ensure `scaler.pkl` matches feature shape used during training.

