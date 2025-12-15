
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
-   **📊 Vector-Based Scoring:** Evaluates output quality using cosine similarity of embeddings.
-   **🔌 Multi-Provider Support:** Seamlessly switch between OpenAI and Google Gemini models.
-   **📂 Custom Dataset Support:** easily plug in your own images and target JSONs.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ayam04/prompt-optimization.git
    cd gepa-tests
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
1.  **Student Model (Test Model):** The generally smaller/cheaper model you want to optimize the prompt for (e.g., `gpt-4o-mini`, `gemini-2.0-flash`).
2.  **Teacher Model (Improve Model):** A stronger, reasoning-capable model that analyzes errors and writes better prompts (e.g., `gpt-4o`, `gemini-1.5-pro`).

> [!TIP]
> The provider (OpenAI or Gemini) is **automatically detected** from the model name. No need to specify it manually!

```bash
# Example: Using OpenAI models
python main.py --iterations 10 --test-model gpt-4o-mini --improve-model gpt-4o

# Example: Using Gemini models
python main.py --iterations 5 --test-model gemini-2.0-flash --improve-model gemini-1.5-pro

# Example: Mix and match providers
python main.py --test-model gemini-2.0-flash --improve-model gpt-4o
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Path to dataset directory | `Dataset` |
| `--iterations` | Number of optimization loops | `10` |
| `--test-model` | **Student:** Model to accept the prompt | `gpt-4o-mini` |
| `--improve-model` | **Teacher:** Model for feedback | `gpt-4o` |

## 📂 bringing Your Own Data

Structure your dataset like this to use your own files:

```
Dataset/
├── images/           # Input images (jpg/png)
│   ├── invoice-001.jpg
├── prompts/          # Initial prompt text
│   ├── invoice.txt
└── outputs/          # Target valid JSON
    ├── invoice.json
```
*Note: Filenames must match the prefix (e.g., `invoice` in the example above).*

## 🤝 Contributing

Contributions are always welcome! Please check out the [contribution guidelines](CONTRIBUTING.md) first.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ for education at <a href="https://peach.study">Peach.study</a>
</p>
