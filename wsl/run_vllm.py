import os
from modelscope import snapshot_download

# ======================
# 配置模型
# ======================
LLM_MODEL = "Qwen/Qwen2.5-0.5B"
VLM_MODEL = "OpenBMB/MiniCPM-V-2_6-int4"  # 或 MiniCPM-V-2_6（非量化）

LLM_DIR = f"./models/{LLM_MODEL.replace('/', '_')}"
VLM_DIR = f"./models/{VLM_MODEL.replace('/', '_')}"

# ======================
# 下载模型（ModelScope）
# ======================
for name, path in [(LLM_MODEL, LLM_DIR), (VLM_MODEL, VLM_DIR)]:
    if not os.path.exists(path) or not os.listdir(path):
        print(f"📥 下载模型: {name}")
        os.makedirs(path, exist_ok=True)
        snapshot_download(model_id=name, local_dir=path)
    else:
        print(f"✅ 模型已存在: {path}")

# ======================
# 启动服务
# ======================
print("\n🚀 启动 LLM 服务 (端口 8000)...")
llm_cmd = (
    f"python -m vllm.entrypoints.openai.api_server "
    f"--model {LLM_DIR} "
    f"--host 0.0.0.0 --port 8000 "
    f"--dtype bfloat16"
)

print("🚀 启动 VLM 服务 (端口 8001)...")
vlm_cmd = (
    f"python -m vllm.entrypoints.openai.api_server "
    f"--model {VLM_DIR} "
    f"--host 0.0.0.0 --port 8001 "
    f"--dtype bfloat16 "
    f"--trust-remote-code"  # ⚠️ 关键参数！
)

print("\n🌐 访问地址:")
print(f"   - LLM: http://localhost:8000/v1")
print(f"   - VLM: http://localhost:8001/v1")

# 后台启动 VLM，前台运行 LLM
os.system(f"nohup {vlm_cmd} > vlm.log 2>&1 &")
os.system(llm_cmd)