# Multimodal_FineTuning_SLM
Multimodal Fine-Tuning using Small Language Models
Image Captioning on Mobile UI Screens
Overview

This project explores multimodal fine-tuning using Small Language Models (SLMs). The goal is to train a lightweight vision-language model to generate natural language descriptions for mobile application screenshots.

The model takes an image of a mobile UI screen as input and produces a textual caption describing the screen content.

The system was trained on the RICO-Screen2Words dataset and fine-tuned using parameter-efficient LoRA adapters to ensure the training process can run efficiently on a T4 GPU.

Dataset
RICO-Screen2Words

This project uses the RICO-Screen2Words, a dataset of mobile application UI screenshots paired with natural language descriptions.

Each sample contains:

image → screenshot of a mobile application UI

captions → textual description of the screen

Example:

Image	Caption
Mobile app UI screenshot	"app showing the weight tracking page"
Dataset Size
Split	Samples
Train	15,743
Validation	2,364
Test	4,310

This dataset is well-suited for multimodal learning, as it aligns visual UI layouts with semantic descriptions.

Model Architecture

The base model used is BLIP Image Captioning Base.

BLIP (Bootstrapped Language Image Pretraining) is a vision-language model designed for tasks such as:

Image captioning

Visual question answering

Vision-language understanding

Why BLIP?

BLIP was selected because:

It is lightweight and suitable for small GPU environments

It provides strong performance for image captioning tasks

It integrates seamlessly with the Transformers ecosystem

It supports efficient multimodal processing (image + text)

Parameter Efficient Fine-Tuning

Instead of fine-tuning the entire model, this project uses LoRA adapters through PEFT.

What is LoRA?

LoRA (Low Rank Adaptation) injects small trainable matrices into the attention layers of a transformer model.

Advantages:

Reduces the number of trainable parameters

Requires significantly less GPU memory

Enables training large models on limited hardware

Faster training and experimentation

In this project, LoRA layers were applied to the query and value projection layers of the attention mechanism.

Data Preprocessing

Data preprocessing is performed using the BLIP processor.

Steps involved:

Images are resized and normalized

Captions are tokenized

Inputs are converted into tensors compatible with the model

Labels are aligned with tokenized captions for training

The preprocessing pipeline ensures that both visual and textual modalities are correctly formatted for the model.

Training Setup

Training was performed using the **Transformers Trainer API.

Hardware

Training was executed on:

GPU: NVIDIA T4

Platform: Google Colab / Kaggle

Training Configuration
Parameter	Value
Batch Size	8
Learning Rate	2e-5
Epochs	2
Precision	FP16
Evaluation Strategy	Step-based evaluation

These hyperparameters were selected to balance training stability and GPU memory constraints.

Training Results

The model was successfully fine-tuned on the RICO dataset.

Example training log:

Step	Training Loss	Validation Loss
500	8.16	8.11
1000	7.93	7.92

The steady decrease in both training and validation loss indicates that the model learned meaningful multimodal representations.

Adapter Merging

After training, the LoRA adapters were merged into the base model weights.

This step is important because:

It produces a standalone model

The model can run inference without requiring external adapters

It simplifies deployment

The merged model produces outputs that are semantically equivalent to the adapter-based model.

## Model

The fine-tuned model is available on Hugging Face:

https://huggingface.co/PMN23/rico-blip-lora

Inference

The model generates captions for input screenshots.

Example code to load and run the model:

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained(
    "PMN23/rico-blip-lora"
)

url = "IMAGE_URL"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(image, return_tensors="pt")

outputs = model.generate(**inputs)

caption = processor.decode(outputs[0], skip_special_tokens=True)

print(caption)
Qualitative Results

Below are examples comparing ground truth captions with model predictions.

Example:

Reference Caption

app showing the weight tracking page

Model Prediction

display of weight tracking app

The predictions show that the model captures high-level semantic information about the UI screens, even when the wording differs slightly from the ground truth.


Reproducibility

To reproduce the results:

Install dependencies

pip install transformers datasets peft accelerate

Run the training notebook

Load the model from Hugging Face Hub

Conclusion

This project demonstrates that multimodal fine-tuning with small language models is feasible on limited compute resources.

Using LoRA adapters and the BLIP architecture, we successfully trained a vision-language model capable of generating meaningful captions for mobile UI screenshots.

The project highlights the effectiveness of parameter-efficient training for multimodal tasks.
