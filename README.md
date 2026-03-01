# CPUT Calendar Q&A System

A complete Question-Answering (Q&A) system built with Rust and the Burn deep learning framework.

This project implements a full machine learning pipeline that:

- Loads and processes Word (`.docx`) calendar documents  
- Builds a tokenizer and dataset  
- Trains a Transformer-based neural network  
- Saves model checkpoints  
- Answers natural language questions via a command-line interface  

---

## 📌 Project Overview

The system is designed to answer questions about CPUT institutional calendar documents for the years 2024–2026.

Example questions:

- *When does Term 1 start in 2026?*  
- *How many times did the HDC hold their meetings in 2024?*  
- *What is the month and date will the 2026 End of Year Graduation Ceremony be held?*

The model learns from document content and supports retrieval-augmented inference.

---

## 🏗️ System Architecture

The pipeline consists of the following components:

### 1. Data Pipeline
- Word document loading using docx-rs
- Text extraction (paragraphs + tables)
- Automatic Q&A dataset generation
- Tokenization and vocabulary building
- Train/validation split

### 2. Model Architecture
- 6-layer Transformer Encoder
- d_model = 128
- num_heads = 4
- Feed-forward layers
- Cross-entropy loss

### 3. Training Pipeline
- Adam optimizer
- Accuracy tracking
- Perplexity calculation
- Checkpoint saving

### 4. Inference System
- Loads checkpoint
- Accepts CLI question input
- Retrieval-augmented answer generation



---

## 🚀 Quick Start

### 1️⃣ Train the Model

```bash
cargo run -- train ./data --epochs 10 --lr 0.0001 --batch-size 4



## ASK QESTIONS

cargo run -- ask model_checkpoint.json "When does Term 1 start in 2026?"

 **Note:** The project report can be found in the `docs/` folder (`docs/REPORT.md`).

---
[View the Report](docs/REPORT.md)