
# OCR Prompt Optimizer for Worksheet Grading

![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-enabled-green.svg?style=flat-square)
![Gemini](https://img.shields.io/badge/Google%20Gemini-supported-orange.svg?style=flat-square)

> A powerful tool to iteratively optimize OCR prompts using LLMs. Built for education with 🍑 [peach.study](https://peach.study).

---

## 📖 About

This project is an open-source tool designed to help developers and students master prompt engineering for OCR tasks. It uses an iterative "teacher-student" loop where a stronger model (like GPT-4o) critiques and improves the prompts used by a faster/cheaper model (like GPT-4o-mini or Gemini Flash).

**Author:** [ayam04](https://github.com/ayam04)

## ✨ Features

-   **🤖 LLM-Driven Optimization:** Automatically improves prompts over multiple iterations based on ground truth.
-   **🔄 Feedback Loop:** Uses a "teacher" model to analyze errors and suggest specific fixes.
-   **📊 Local Vector Scoring:** Evaluates output quality using cosine similarity with [fastembed](https://github.com/qdrant/fastembed) (no API costs!).
-   **🔌 Multi-Provider Support:** Seamlessly switch between OpenAI and Google Gemini models.
-   **📂 Multi-Sample Support:** Process multiple samples automatically - just add folders to the Dataset directory.


## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/learnoAI/prompt-optimization.git
    cd prompt-optimization
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


## 🚀 Usage

Run the optimizer via the CLI. The tool uses two models:
1.  **Student Model (Test Model):** The smaller/cheaper model you want to optimize the prompt for.
2.  **Teacher Model (Improve Model):** A stronger model that analyzes errors and writes better prompts.


> [!TIP]
> The provider (OpenAI or Gemini) is **automatically detected** from the model name!

```bash
# Example: Using OpenAI models
python main.py --iterations 5 --test-model gpt-4o-mini --improve-model gpt-5

# Example: Using Gemini models
python main.py --iterations 5 --test-model gemini-2.0-flash --improve-model gemini-3-pro-preview
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Path to dataset directory | `Dataset` |
| `--iterations` | Number of optimization loops | `10` |
| `--test-model` | **Student:** Model to test prompts | `gpt-4o-mini` |
| `--improve-model` | **Teacher:** Model for feedback | `gpt-5` |


## 📂 Dataset Structure

Create a folder for each sample you want to optimize. The tool auto-discovers all sample folders:

```
Dataset/
├── worksheet_math/           # Sample 1
│   ├── images/               # Input images (jpg/png)
│   │   ├── page1.jpg
│   │   └── page2.jpg
│   ├── prompts/              # Initial prompt
│   │   └── prompt.txt
│   └── outputs/              # Target JSON
│       └── expected.json
│
├── worksheet_physics/              # Sample 2
│   ├── images/
│   ├── prompts/
│   └── outputs/
```

**Results** are automatically saved by sample name:

```
Results/
├── worksheet_math/
│   ├── optimized_prompt.txt
│   └── best_output.json
├── worksheet_physics/
│   └── optimized_prompt.txt
│   └── best_output.json
```

## 🤝 Contributing

Contributions are always welcome! Please check out the [contribution guidelines](CONTRIBUTING.md) first.

---

<p align="center">
  Built with ❤️ for education at <a href="https://peach.study">Peach.study</a>
</p>
