# OCR Prompt Optimizer for Worksheet Grading

![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-enabled-green.svg?style=flat-square)
![Gemini](https://img.shields.io/badge/Google%20Gemini-supported-orange.svg?style=flat-square)

> A powerful tool to iteratively optimize OCR prompts using LLMs. Built for education with 🍑 [peach.study](https://peach.study).

## 📖 About

This project is an open-source tool designed to help developers and students master prompt engineering for OCR tasks. It uses an iterative "teacher-student" loop where a stronger model (like Gemini-3-pro) critiques and improves the prompts used by a faster/cheaper model (like Gemini Flash).

## Features

- **LLM-Driven Optimization:** Automatically improves prompts over multiple iterations based on ground truth.
- **Feedback Loop:** Uses a "teacher" model to analyze errors and suggest specific fixes.
- **Local Vector Scoring:** Evaluates output quality using cosine similarity with [fastembed](https://github.com/qdrant/fastembed) (no API costs!).
- **Multi-Provider Support:** Seamlessly switch between OpenAI and Google Gemini models.
- **Multi-Sample Support:** Process multiple samples automatically - just add folders to the Dataset directory.

## Installation

1.  **Clone the repository**

    ```bash
    git clone https://github.com/learnoAI/ocr-prompt-optimization.git
    cd ocr-prompt-optimization
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_openai_key
    GEMINI_API_KEY=your_gemini_key
    ```

## Usage

Run the optimizer via the CLI. The tool uses two models:

1.  **Student Model (Test Model):** The smaller/cheaper model you want to optimize the prompt for.
2.  **Teacher Model (Improve Model):** A stronger model that analyzes errors and writes better prompts.

```bash
python main.py --iterations 5 --test-model gpt-4o-mini --improve-model gpt-5.2-2025-12-11

python main.py --iterations 5 --test-model gemini-2.0-flash --improve-model gemini-3-pro-preview
```

| Argument          | Description                        | Default              |
| :---------------- | :--------------------------------- | :------------------- |
| `--dataset`       | Path to dataset directory          | `Dataset`            |
| `--iterations`    | Number of optimization loops       | `10`                 |
| `--test-model`    | **Student:** Model to test prompts | `gpt-4o-mini`        |
| `--improve-model` | **Teacher:** Model for feedback    | `gpt-5.2-2025-12-11` |

## Dataset Structure

Create a folder for each sample you want to optimize. The tool auto-discovers all sample folders:

```
Dataset/
├── sample1/                  # Sample 1
│   ├── images/               # Input images (jpg/png/jpeg)
│   │   ├── page1.jpg
│   │   └── page2.jpg
│   ├── prompts/              # Initial prompt
│   │   └── prompt.txt
│   └── outputs/              # Target JSON
│       └── expected.json
│
├── sample2/                  # Sample 2
│   ├── images/
│   ├── prompts/
│   └── outputs/
```

**Results** are automatically saved by sample name:

```
Results/
├── sample1/
│   ├── optimized_prompt.txt
│   └── best_output.json
├── sample2/
│   └── optimized_prompt.txt
│   └── best_output.json
```

<p align="center">
  Built with ❤️ for education at <a href="https://peach.study">Peach.study</a>
</p>
