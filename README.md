# Qwen3-ASR 使用

本指南基于官方文档，记录了如何在本地环境中使用 Qwen3-ASR-0.6B 模型进行语音识别的完整步骤。

## 项目说明

本项目是一个使用 Qwen3-ASR-0.6B 模型的实践笔记，包含了：
- 环境搭建和依赖安装
- 模型下载与配置
- 两种后端的使用方法（Transformers 和 vLLM）
- OpenAI 兼容 API 服务的部署
- 语音文件转写的完整流程

## 环境要求

- Python 3.8+
- CUDA 兼容的 GPU（推荐，用于加速推理）
- 至少 4GB GPU 内存

## 安装依赖

```bash
pip install huggingface_hub[cli] qwen-asr[vllm] flash-attn vllm -U
```

## 使用步骤

### 1. 下载测试音频文件

```bash
wget -O csgo.wav https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR/csgo.wav
```

### 2. 下载 Qwen3-ASR-0.6B 模型

```bash
hf download Qwen/Qwen3-ASR-0.6B --local-dir Qwen3-ASR-0.6B --local-dir-use-symlinks False
```

### 3. 方法一：使用 Transformers 后端进行语音识别

```python
import torch
from qwen_asr import Qwen3ASRModel, parse_asr_output

# 定义模型路径
model_path = 'Qwen3-ASR-0.6B'

# 初始化 ASR 模型
asr_model = Qwen3ASRModel.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=128,
)

print("ASR model loaded successfully.")

# 定义音频文件路径
audio_file_path = 'csgo.wav'

# 转写音频文件
transcription = asr_model.transcribe(
    audio_file_path,
    language='English'
)

# 解析并打印结果
language, text = parse_asr_output(transcription)

print("Transcription successful:")
print(f"Language: {language}")
print(f"Text: {text}")
```

### 4. 方法二：使用 vLLM 后端进行语音识别

```python
from qwen_asr import Qwen3ASRModel, parse_asr_output

# 定义模型路径
model_path = 'Qwen3-ASR-0.6B'

# 初始化 ASR 模型（使用 vLLM 后端）
asr_model = Qwen3ASRModel.LLM(
    model_path,
    max_inference_batch_size=128
)

# 转写音频文件
transcription = asr_model.transcribe(
    audio_file_path,
    language='English'
)

# 解析并打印结果
language, text = parse_asr_output(transcription)

print("Transcription successful:")
print(f"Language: {language}")
print(f"Text: {text}")
```

### 5. 方法三：启动 OpenAI 兼容的 API 服务

#### 5.1 启动服务

```bash
qwen-asr-serve Qwen3-ASR-0.6B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```

#### 5.2 通过 API 进行语音识别

```python
import requests
from qwen_asr import parse_asr_output

# 定义 API 端点和请求头
url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 打开音频文件
audio_file_path = "csgo.wav"
with open(audio_file_path, "rb") as audio_file:
    # 构建请求数据
    files = {"file": (audio_file_path, audio_file, "audio/wav")}
    data = {"model": "qwen3-asr-0.6b"}

    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=data, files=files, timeout=300)
    response.raise_for_status()

# 解析转录内容
content = response.json()["choices"][0]["message"]["content"]
language, text = parse_asr_output(content)

# 打印结果
print("Transcription successful:")
print(f"Language: {language}")
print(f"Text: {text}")
```


## 参考资料

- [Qwen3-ASR-0.6B 模型官方页面](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [qwen-asr 官方文档](https://github.com/QwenLM/qwen-asr)
- [vLLM 官方文档](https://docs.vllm.ai/)
