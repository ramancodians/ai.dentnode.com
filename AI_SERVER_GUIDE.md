# DentNode AI Server Guide

This guide describes how to access and use the AI models (Qwen, GLM, Gemma) hosted on the DentNode AI server.

## 🚀 Server Overview

- **Server Name**: `ai-coder-server`
- **Machine Type**: `g2-standard-8` (L4 GPU)
- **Region**: `us-central1-a`
- **Public IP**: `35.238.98.15`

## 🛠 Available Services

### 1. Open WebUI (Browser Interface)
A user-friendly web interface for chatting with all available models.
- **URL**: [http://35.238.98.15:8080](http://35.238.98.15:8080)

### 2. Ollama API
The primary backend serving most models. It is OpenAI-compatible.
- **Base URL**: `http://35.238.98.15:11434`
- **API Endpoint**: `http://35.238.98.15:11434/v1` (OpenAI-compatible)

---

## 🤖 Available Models

| Model Name | Ollama ID / HF Path | Description |
|------------|-----------|-------------|
| **Qwen 2.5 Coder** | `qwen2.5-coder:32b` | High-performance coding and reasoning model. |
| **Gemma 2** | `gemma2:27b` | Google's high-efficiency open model (User referred as Gamma). |
| **GLM-5** | `zai-org/GLM-5` | (In preparation) Next-gen General Language Model from Zhipu AI. |

---

## 💻 How to Use

### Using with `curl` (Ollama API)
```bash
curl http://35.238.98.15:11434/api/generate -d '{
  "model": "qwen2.5-coder:32b",
  "prompt": "Why is the sky blue?"
}'
```

### Using as an OpenAI-Compatible API
You can use the server with any OpenAI client library by setting the `base_url`.

**Python Example:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://35.238.98.15:11434/v1",
    api_key="ollama" # Required but ignored by Ollama
)

response = client.chat.completions.create(
    model="qwen2.5-coder:32b",
    messages=[{"role": "user", "content": "Write a TypeScript interface for a Dental Case."}]
)
print(response.choices[0].message.content)
```

### Using in VS Code (Continue / Cline / Roo Code)
To use these models in your IDE for coding assistance:

1. **Continue / Cline**:
   - Provider: `Ollama`
   - Base URL: `http://35.238.98.15:11434`
   - Model ID: `qwen2.5-coder:32b`

2. **OpenAI-Compatible IDE Agents**:
   - Base URL: `http://35.238.98.15:11434/v1`
   - API Key: `any-string`
   - Model: `qwen2.5-coder:32b` or `gemma2:27b`

---

## 🛠 Administration

### To manage models (SSH into server):
```bash
# List models
ollama list

# Pull a new model
ollama pull llama3.1

# Check server status
sudo docker ps # For Open WebUI
```

> **Note**: This server is hosted on Google Cloud Platform (`app-dentnode-com` project).
