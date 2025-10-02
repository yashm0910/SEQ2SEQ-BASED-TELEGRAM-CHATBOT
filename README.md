# Contextual Dialogue Chatbot with T5 Seq2Seq Fine-Tuning

---

## Overview

This project develops a lightweight, context-aware Telegram chatbot for engaging, multi-turn conversations. By fine-tuning the **T5-small** sequence-to-sequence model on a curated dialogue corpus, the bot generates responsive replies that maintain context, pose follow-up questions, and simulate natural interaction. While not matching the fluency of large-scale LLMs, it excels in targeted, efficient deployment—ideal for educational or casual chat applications. Trained on 30K records in ~25 minutes on GPU, it prioritizes accessibility over exhaustive scale.

---

## Dataset

- **Source**: Combined multi-turn dialogue datasets (e.g., DailyDialog + custom conversational logs; total ~120K raw exchanges)
- **Size**: 30K filtered records for training (train/validation split: 80/20; stratified for dialogue length and topic diversity)
- **Preprocessing**:
  - NLP techniques: Tokenization (Hugging Face tokenizer), lowercasing, punctuation normalization, stopword removal, and duplicate elimination
  - Context encoding: Prompted as "context: [history] question: [user input]" for seq2seq input
  - Augmentation: None (focused on quality over quantity to manage compute constraints)

---

## Methodology

### 1. Data Preparation
- Merged datasets into a unified JSONL format with input-output pairs
- Applied cleaning pipeline using NLTK and spaCy for noise reduction and coherence checks
- Ensured balanced representation of casual topics (e.g., hobbies, books, daily life) via topic modeling (LDA)

### 2. Model Architecture
- **Base Model**: T5-small (60M parameters; pretrained on C4 for text-to-text tasks)
- **Task Formulation**: Seq2seq for dialogue generation—inputs as "dialogue: [context + query]" → outputs as generated responses
- **Customization**: No additional layers; leveraged T5's encoder-decoder for end-to-end fine-tuning

### 3. Training Strategy
- **Framework**: Hugging Face Transformers (PyTorch backend)
- **Key Hyperparameters**:
  - Learning Rate: 5e-5
  - Batch Size: 8 (train/eval)
  - Epochs: 3
  - Optimizer: AdamW with 0.01 weight decay
  - Mixed Precision: FP16 (GPU acceleration)
- **Monitoring**: Epoch-wise evaluation with BLEU/ROUGE scores; early stopping on validation perplexity
- **Compute**: ~25 minutes on single T5 GPU for 30K samples

---

## Deployment

- **Platform**: Telegram Bot API integration via `python-telegram-bot` library
- **Local Execution**: Run `app.py` for instant deployment—handles incoming messages, generates responses via the fine-tuned model, and maintains session context
- **Sample Interaction**: See attached screenshot for a demo conversation (e.g., discussing transformers, books like Harry Potter, and exam stress). The bot responds contextually ("Haha, I agree! Do you have any hobbies?") and probes further ("What's your favourite book?"), fostering engaging back-and-forth.

*Note*: Currently local-only; not publicly hosted. Inference latency: ~1-2 seconds per response on CPU.

---

## Results

| Metric          | Train Set | Validation Set |
|-----------------|-----------|----------------|
| **BLEU Score**  | 0.45     | 0.42          |
| **ROUGE-1**     | 0.52     | 0.49          |
| **ROUGE-L**     | 0.48     | 0.45          |
| **Perplexity**  | 12.3     | 14.1          |

- **Qualitative**: Generates coherent, question-asking replies (e.g., transitions from tech jokes to personal queries); handles casual topics with ~80% relevance in manual evals
- **Limitations**: Occasional repetition or brevity due to small model size; outperforms rule-based bots in context retention

---

## Key Highlights

- **Efficiency**: T5-small enables quick fine-tuning and low-latency inference, deployable on consumer hardware without cloud dependency
- **Contextual Depth**: Maintains dialogue history for natural flow, e.g., referencing prior topics like "Harry Potter" in follow-ups
- **Extensibility**: Modular design supports easy dataset expansion or model swaps (e.g., to T5-base for enhanced fluency)

---

## Future Enhancements

- Scale training to 100K+ records with distributed GPU for improved fluency
- Integrate retrieval-augmented generation (RAG) for fact-checked responses
- Public deployment via Heroku/Render with rate limiting
- Add multi-modal support (e.g., image queries) using CLIP integration

---

## Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: Hugging Face Transformers, PyTorch
- **NLP Tools**: NLTK, spaCy, Datasets library
- **Deployment**: python-telegram-bot, Streamlit (for optional local UI)
- **Environment**: Jupyter Notebook / Google Colab (GPU for training)

---

## Impact

This project bridges accessible NLP with practical chat applications, empowering developers to build responsive bots without massive resources. By fine-tuning open models on dialogue data, it advances ethical AI for education and mental health support—e.g., stress-relief chats. Repository includes full code, model weights, and dataset samples for seamless replication and collaboration.
