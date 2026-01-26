from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams
import os

# 使用与 run_vllm.py 中相同的本地模型路径
LLM_MODEL = "OpenBMB/MiniCPM4-0.5B-QAT-Int4-GPTQ-format"
LLM_DIR = f"/root/my_python_server/models/{LLM_MODEL.replace('/', '_')}"

# 检查本地模型是否存在，如果不存在则从 ModelScope 下载
if not os.path.exists(LLM_DIR) or not os.listdir(LLM_DIR):
    print(f"本地模型不存在，将从 ModelScope 下载: {LLM_MODEL}")
    from modelscope import snapshot_download
    snapshot_download(model_id=LLM_MODEL, local_dir=LLM_DIR)

model_name = LLM_DIR  # 使用本地模型路径
prompt = [{"role": "user", "content": "推荐5个北京的景点。"}]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

llm = LLM(
    model=model_name,
    quantization="gptq_marlin",
    trust_remote_code=True,
    max_num_batched_tokens=512,
    dtype="bfloat16",
    gpu_memory_utilization=0.1,
)
sampling_params = SamplingParams(top_p=0.7, temperature=0.7, max_tokens=1024, repetition_penalty=1.02)

outputs = llm.generate(prompts=input_text, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)