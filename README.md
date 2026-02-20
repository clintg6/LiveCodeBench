# LiveCodeBench
Official repository for the paper "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code"

<p align="center">
    <a href="https://livecodebench.github.io/">🏠 Home Page</a> •
    <a href="https://huggingface.co/livecodebench/">💻 Data </a> •
    <a href="https://livecodebench.github.io/leaderboard.html">🏆 Leaderboard</a> •
    <a href="https://huggingface.co/spaces/livecodebench/code_generation_samples">🔍 Explorer</a> 
</p>

## Introduction
LiveCodeBench provides holistic and contamination-free evaluation of coding capabilities of LLMs.  Particularly, LiveCodeBench continuously collects new problems over time from contests across three competition platforms -- LeetCode, AtCoder, and CodeForces. Next, LiveCodeBench also focuses on a broader range of code-related capabilities, such as self-repair, code execution, and test output prediction, beyond just code generation. Currently, LiveCodeBench hosts four hundred high-quality coding problems that were published between May 2023 and March 2024.

## Prerequisites

Before you start, ensure you have:

- MI300+ GPUs
- Local model files available at /models/DeepSeek-R1-MXFP4-Preview/ and /models/DeepSeek-R1-NextN (paths and models can be adjusted).

## Getting Started

### Docker
Spin up an SGLang docker using the following command:
```bash
docker run -it --network=host --group-add=video --privileged --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri -v /home/clgreene:/workspace -v /data:/models rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260217
```

### Set up folders and clone the repository
Create a project directory and clone the LiveCodeBench repository:
```bash
mkdir -p /app
cd /app
git clone -b sglang https://github.com/clintg6/LiveCodeBench
```

### Install LiveCodeBench
Create a project directory and clone the LiveCodeBench repository:
```bash
cd LiveCodeBench
pip install .
```

### Configure environment variables
Set the required environment variables for OpenAI API access:
```bash
export OPENAI_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
```

### Launch the SGLang server
Launch the SGLANG server with the following configuration:
```bash
SGLANG_AITER_MLA_PERSIST=1 \
AITER_MXFP4_MOE_SF=1 \
SGLANG_USE_AITER=1 \
SGLANG_INT4_WEIGHT=0 \
SGLANG_MOE_PADDING=1 \
SGLANG_SET_CPU_AFFINITY=1 \
SGLANG_ROCM_FUSED_DECODE_MLA=1 \
SGLANG_USE_ROCM700A=1 \
nohup python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-R1-MXFP4-Preview/ \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --log-requests \
  --mem-fraction-static 0.95 \
  --chunked-prefill-size 131072 \
  --attention-backend aiter \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-draft-model-path /models/DeepSeek-R1-NextN \
  --max-running-requests 64 \
  --disable-radix-cache \
  --kv-cache-dtype fp8_e4m3 \
  --served-model-name deepseek-r1-mxfp4 \
  --reasoning-parser deepseek-r1 \
  &
```

## Inference and Evaluation

### Run LiveCodeBench Generation Benchmark
Once the server is running, benchmark code generation of the model with:
```bash
python -m lcb_runner.runner.main \
  --model deepseek-r1-mxfp4 \
  --scenario codegeneration \
  --evaluate \
  --release_version release_v1 \
  --start_date 2024-01-01 \
  --n 1 \
  --max_tokens 16384
```

### Dataset Versions
Since LiveCodeBench is a continuously updated benchmark, we provide different versions of the dataset. Particularly, we provide the following versions of the dataset:
- `release_v1`: The initial release of the dataset with problems released between May 2023 and Mar 2024 containing 400 problems.
- `release_v2`: The updated release of the dataset with problems released between May 2023 and May 2024 containing 511 problems.
- `release_v3`: The updated release of the dataset with problems released between May 2023 and Jul 2024 containing 612 problems.
- `release_v4`: The updated release of the dataset with problems released between May 2023 and Sep 2024 containing 713 problems.
- `release_v5`: The updated release of the dataset with problems released between May 2023 and Jan 2025 containing 880 problems.
- `release_v6`: The updated release of the dataset with problems released between May 2023 and Apr 2025 containing 1055 problems.

You can use the `--release_version` flag to specify the dataset version you wish to use. Particularly, you can use the following command to run the evaluation on the `release_v2` dataset. Release version defaults to `release_latest`. Additionally, we have introduced fine-grained release versions such as `v1`, `v2`, `v1_v3`, `v4_v5` for specific versions of the dataset.

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario codegeneration --evaluate --release_version release_v2
```

## Adding Support for New Models

To add support for new models, we have implemented an extensible framework to add new models and customize prompts appropriately. 

Step 1: Add a new model to the [./lcb_runner/lm_styles.py](./lcb_runner/lm_styles.py) file. Particularly, extend the `LMStyle` class to add a new model family and extend the model to the `LanguageModelList` array.

Step 2: Since we use instruction tuned models, we allow configuring the instruction for each model. Modify the [./lcb_runner/prompts/generation.py](./lcb_runner/prompts/generation.py) file to add a new prompt for the model in the `format_prompt_generation` function. 
For example, the prompt for `DeepSeekCodeInstruct` family of models looks as follows

```python
# ./lcb_runner/prompts/generation.py
if LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
    prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n"
    prompt += f"{get_deepseekcode_question_template_answer(question)}"
    return prompt
```

## Citation

```bibtex
@article{jain2024livecodebench,
  author    = {Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica},
  title     = {LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  year      = {2024},
  journal   = {arXiv preprint},
}
```
