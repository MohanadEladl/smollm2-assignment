# SmolLM2 Fine-Tuning & Deployment 🚀

Welcome to the repository for the SmolLM2 fine-tuning assignment. This project demonstrates the end-to-end pipeline of taking a pre-trained Small Language Model (SLM), fine-tuning it on a custom dataset, and deploying it as an interactive web application.

## 🌐 Live Demo
You can interact with the fine-tuned model directly through our web interface hosted on GitHub Pages:
**👉 [Test the Live Application Here](https://MohanadEladl.github.io/smollm2-assignment/)**

> ⚠️ **IMPORTANT DISCLAIMER regarding Uptime:**
> The backend of this application is hosted on the free tier of Hugging Face Spaces, which automatically goes to **"Sleep Mode"** after 48 hours of inactivity to conserve server resources. 
> 
> *If the live link above is unresponsive or endlessly loading, the Hugging Face server has likely gone to sleep. It must be manually restarted from the Hugging Face dashboard before the web interface will function again.*

---

## 🏗️ Project Architecture
This repository contains the source code used to train the model and deploy the frontend:

- **`Smollm_assignment.ipynb` & `mohanad-finetune.ipynb`**: The core Jupyter notebooks containing the data preparation, hyperparameter configuration, and LoRA/PEFT fine-tuning pipelines for the `HuggingFaceTB/SmolLM` model family.
- **`app.py`**: The Python backend script utilizing the `gradio` library to serve the model inferences and manage the API pipeline.
- **`index.html`**: A lightweight, static frontend designed to be hosted on GitHub Pages, which securely embeds the Hugging Face Gradio backend via an iframe.
- **`requirements.txt`**: The dependency map required to replicate the Python environment (including `transformers`, `peft`, `accelerate`, and `gradio`).

*(Note: The heavy model weights like `.safetensors` and vocabulary files are intentionally excluded from this repository and are hosted exclusively on the Hugging Face Model Hub.)*

## ⚙️ Technical Stack
* **Base Model**: SmolLM-135M / SmolLM2-360M
* **Fine-Tuning**: Parameter-Efficient Fine-Tuning (PEFT)
* **Serving**: FastAPI & Gradio (via Hugging Face Spaces)
* **Frontend Hosting**: GitHub Pages
