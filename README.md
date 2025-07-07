# Depression Detection System

This project predicts depression using both audio signals and facial expressions from video input. It combines a BiLSTM-based model for audio (MFCC + delta + delta-delta features) and a DenseNet201-based model for facial emotion detection from video frames.

## Features
- 🎙️ Audio analysis using MFCC + delta features (78-dim total)
- 🎥 Facial emotion detection via DenseNet201
- 🤝 Late fusion of audio and video models for better prediction
- 🧠 Supports fine-tuning and ensemble learning
- 📦 Gradio interface available for quick testing

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt


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

