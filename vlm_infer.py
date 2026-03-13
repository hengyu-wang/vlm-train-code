# --------------------------------------------------------------
# 1️⃣ 环境设置
# --------------------------------------------------------------
import os, sys, json, math, pandas as pd
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import time
os.environ["VLLM_USE_V1"] = "0"          # ParaThinker 不支持 V1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.setrecursionlimit(50_000)

# --------------------------------------------------------------
# 2️⃣ 路径配置
# --------------------------------------------------------------
MODEL_PATH = ""
PROCESSED_JSONL_PATH = "" # Vstar
OUTPUT_EXCEL_PATH = "./3b-all/test_50-lbd0.3.csv"

# --------------------------------------------------------------
# 3️⃣ 视觉 Token 计数（保留原实现）
# --------------------------------------------------------------
def calculate_vision_tokens(
    image_path: str,
    patch_size: int = 14,
    merge_size: int = 2,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 16384,
) -> int:
    def smart_resize(height: int, width: int, factor: int):
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size  # 正确：width, height
    except Exception as e:
        print(f"[ERR] open {image_path}: {e}")
        return 0

    resize_factor = patch_size * merge_size
    # 注意：传入 smart_resize 的顺序要和单条脚本一致：height, width
    resized_height, resized_width = smart_resize(
        original_height, original_width, factor=resize_factor
    )
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    num_vision_tokens = (grid_h * grid_w) // (merge_size ** 2)
    return num_vision_tokens


# --------------------------------------------------------------
# 4️⃣ 数据读取
# --------------------------------------------------------------
def convert_processed_jsonl_to_image_messages(fp: str):
    msgs = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if "image" in d and "text" in d:
                    msgs.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": d["image"]},
                            {"type": "text",   "text": d["text"]},
                        ],
                    })
            except Exception as e:
                print(f"[skip] {e}")
    return msgs

def load_original_jsonl(fp: str):
    data = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

# --------------------------------------------------------------
# 5️⃣ 模型 & 处理器初始化
# --------------------------------------------------------------
print("🔧 Initializing processor & model …")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

# Pad‑token configuration (ParaThinker‑specific)
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<vllm_pad>")
tokenizer.pad_token = "<vllm_pad>"
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1},
    block_size=16,                 # 与原脚本保持一致
    gpu_memory_utilization=0.9,
    dtype="float16",
)

# --------------------------------------------------------------
# 6️⃣ 推理超参数（保持原始数值）
# --------------------------------------------------------------
think_token_ids = [151665, 151667, 151669, 151671,
                   151673, 151675, 151677, 151679]
stop_token_ids  = [151643, 151666, 151668, 151670,
                   151672, 151674, 151676, 151678,
                   151680, 151682, 151684]
summary_prompt = "<summary>By analyzing multiple reasoning processes above, I concluded that: The final answer is"
summary_token_ids = tokenizer.encode(summary_prompt, add_special_tokens=False)
okay_token_ids = [[]] * 8
parthink_size = 4
max_tokens = 1024 * 32   # 与原脚本相同

sampling_params = SamplingParams(
    n=1,
    temperature=1.0,
    top_p=1.0,
    max_tokens=max_tokens,
    stop_token_ids=stop_token_ids,
    ignore_eos=True,
    repetition_penalty=1.0,
)

# --------------------------------------------------------------
# 7️⃣ 读取数据
# --------------------------------------------------------------
all_image_messages = convert_processed_jsonl_to_image_messages(PROCESSED_JSONL_PATH)
original_data = load_original_jsonl(PROCESSED_JSONL_PATH)

print(f"📂 Loaded {len(all_image_messages)} examples – start single‑item inference …")
results = []
start_time = time.time() 

total_output_tokens = 0  # 统计总输出token数
correct_count = 0        # 如果需要统计准确率
# --------------------------------------------------------------
# 8️⃣ **单条**推理循环
# --------------------------------------------------------------
for idx, msg in enumerate(all_image_messages, start=1):
    # print("-" * 60)
    # print(f"🔎 Item {idx}/{len(all_image_messages)}")
    # xinjia
    img_path = msg["content"][0]["image"]
    user_content = msg["content"][1]["text"]
    temp_messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_content}]}
    ]
    # import pdb; pdb.set_trace()
    full_prompt_text1 = processor.apply_chat_template(temp_messages, tokenize=False, add_generation_prompt=True)
    # import pdb; pdb.set_trace()
    # full_prompt_text1 += "<|im_start|>assistant"
    # print("full_prompt_text1",full_prompt_text1)
    # import pdb; pdb.set_trace()
    # 获取文本 Token 长度
    text_token_len = len(processor.tokenizer.encode(full_prompt_text1))
    # 获取视觉 Token 长度
    vision_token_len = calculate_vision_tokens(img_path)
    
    # Qwen2.5-VL 逻辑中，总长度 = 文本Tokens + 视觉Tokens - 1 (因为 <|vision_start|> 等占位符会被替换)
    # 或者简单理解为模型输入的实际序列长度
    total_len = text_token_len + vision_token_len - 2
    
    # 4. 计算需要补多少个 vllm_pad
    num_pad = 0
    block_size = 16
    if total_len % block_size != 0:
        num_pad = block_size - (total_len % block_size)
        # print("在userprompt中补多少个<vllm_pad>",num_pad)
    # 5. 将补全符号加到原始 User Content 之后
    final_user_content = user_content
    # final_user_content = user_content
    msg["content"][1]["text"] = final_user_content
    # print("最后的user prompt",final_user_content)
    try:
        # ---- 8.1 取图片路径并计数视觉 token ----
        img_path = msg["content"][0]["image"]
        vision_tok_cnt = calculate_vision_tokens(img_path)
        if vision_tok_cnt <= 0:
            raise ValueError(f"vision token count = {vision_tok_cnt}")

        # ---- 8.2 构造文本 Prompt（含 <|im_start|>assistant）----
        raw_prompt = processor.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=False
        )
        raw_prompt += "<|im_start|>assistant"+ ("<vllm_pad>" * num_pad)
        # print("最后的user prompt",raw_prompt)
        # ---- 8.3 Token‑level padding（保持 block_size 对齐）----
        txt_ids = tokenizer.encode(raw_prompt)
        total_len = len(txt_ids) + vision_tok_cnt - 1   # “‑1” 为 <image> 占位符的补偿
        if total_len % 16 != 0:                       # block_size = 16
            pad_num = 16 - total_len % 16
            # txt_ids += [tokenizer.pad_token_id] * pad_num
            print(f"有问题，还需要补pad ↪ padding {pad_num} token(s)")
        final_prompt = tokenizer.decode(txt_ids)

        # ---- 8.4 处理视觉输入（Qwen‑VL utils）----
        image_inputs, _, mm_kwargs = process_vision_info(
            [msg], return_video_kwargs=True
        )

        # ---- 8.5 组装 vLLM 输入字典 ----
        vllm_input = {
            "prompt": final_prompt,
            "multi_modal_data": {"image": image_inputs} if image_inputs else {},
            "mm_processor_kwargs": mm_kwargs or {},
        }

        out = llm.generate(
            prompts=[vllm_input],
            sampling_params=sampling_params,
            cot_token_ids=think_token_ids,
            okay_token_ids=okay_token_ids,
            summary_token_ids=summary_token_ids,
            parthink_size=parthink_size,
            pad_token_id=int(tokenizer.pad_token_id),
        )[0]   # 只会返回一个元素

        # ---- 8.7 统计所有输出的token数（思考过程+summary）----
        current_sample_tokens = 0
        for output_idx, output in enumerate(out.outputs):
            path_tokens = len(output.token_ids)
            current_sample_tokens += path_tokens
            print(f"     Path {output_idx}: {path_tokens} tokens")
        
        # 累加到总token数
        total_output_tokens += current_sample_tokens
        
        # 取最后一个子输出作为最终结果（summary）
        last_idx = len(out.outputs) - 1
        gen_ids = out.outputs[last_idx].token_ids
        gen_text = tokenizer.decode(gen_ids)
        
        print(f"✅ Item {idx}: Total tokens for this sample = {current_sample_tokens}")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        gen_text = f"[ERROR] {e}"
        current_sample_tokens = 0

    # ---- 8.8 保存本条记录 ----
    src = original_data[idx - 1] if idx - 1 < len(original_daata) else {}
    results.append({
        "Original Input Text": src.get("text", "N/A"),
        "original_input_label": src.get("label", "N/A"),
        "Generated Output": gen_text,
        "Total Output Tokens": current_sample_tokens,  # 记录本样本的总token数
    })

end_time = time.time()

# --- 时间与统计计算 ---
total_duration = end_time - start_time
avg_time_per_sample = total_duration / len(results) if results else 0
samples_per_second = len(results) / total_duration if total_duration > 0 else 0
avg_tokens_per_sample = total_output_tokens / len(results) if results else 0
tokens_per_second = total_output_tokens / total_duration if total_duration > 0 else 0

print("-" * 60)
print(f"🔤 Total Output Tokens (All Paths): {total_output_tokens:,}")
print(f"🔤 Average Tokens per Sample: {avg_tokens_per_sample:.2f}")
print(f"🚀 Token Generation Speed: {tokens_per_second:.2f} tokens/s")
print(f"⏱️  Total Time Taken: {total_duration:.2f} seconds")
print(f"⏱️  Average Time per Sample: {avg_time_per_sample:.2f}s")
print(f"🚀 Throughput: {samples_per_second:.2f} samples/s")
accuracy = correct_count / len(results) if results else 0
print(f"📊 Final Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
# --------------------------------------------------------------
# 9️⃣ 保存为 Excel
# --------------------------------------------------------------
print("-" * 60)
print("💾 Writing results …")
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_EXCEL_PATH), exist_ok=True)
df.to_csv(OUTPUT_EXCEL_PATH, index=False)
print(f"✅ Done – saved to {OUTPUT_EXCEL_PATH}")
