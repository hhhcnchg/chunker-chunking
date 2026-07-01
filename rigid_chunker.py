"""
RIGID CHUNKER — 刚性段落锁定 + 柔性语义内拆
========================================
以自然段落为刚性边界，仅在段落内部做语义拆分。
存入 Chroma 时子块带完整父段落，检索自动替换。

模型检测自动降级链：
  Tier 1: 有 LLM → LLM 粗切语义大段 + Embedding 细切
  Tier 2: 有 Embedding → bge 算句间相似度，最低处切开
  Tier 3: 啥都没有 → 句号硬切兜底
  * 检测不到 LLM 时会询问用户是否配置 API key

Harness：
  1. 每段最多拆 6 块
  2. 子块最小 50 字
  3. LLM 只输出切分位置，不改写原文
"""

import os
import re
import json
import statistics
import requests
import chromadb

# ═══════════════════════════════════════════════
# 用户配置区（新用户改这里就行）
# ═══════════════════════════════════════════════

# --- 嵌入模型路径 ---
# 优先使用本地模型文件，不填则自动从 HuggingFace 下载
EMBEDDING_MODEL_PATH = ""  # 留空则从 HuggingFace 自动下载
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # HuggingFace 模型名

# --- 本地 LLM 权重目录 ---
# 程序会自动扫描以下目录中是否有 DeepSeek-R1/Qwen 等模型文件
# 不填则跳过本地模型权重检测
LOCAL_MODEL_DIRS = [
    # 把自己的模型目录加在这里，例如：
    # "d:/models",
    # os.path.expanduser("~/.cache/huggingface/hub"),
]

# --- 分块参数 ---
MAX_SUB_CHUNKS = 6        # 每段最多拆成几块
MIN_CHUNK_CHARS = 50      # 子块最小字符数
SHORT_PARA_LIMIT = 50     # 短段落阈值（低于此字数尝试合并）
BASE_THRESHOLD = 500      # 动态阈值锚点

# --- Chroma ---
COLLECTION_NAME = "rigid_chunker_demo"
CHROMA_PERSIST_DIR = ""  # 留空 = 内存模式，填路径 = 持久化（如 os.path.expanduser("~/chroma_data")）

# ═══════════════════════════════════════════════
# 模型检测（自动检测 + 交互询问）
# ═══════════════════════════════════════════════

def detect_models():
    """
    扫描当前环境可用的模型能力。检测不到 LLM 时会询问用户。
    返回 { "embedding": model|None, "llm": callable|None, "level": str }
    """
    embedding = None
    llm = None
    level = "none"

    # ── 1. Embedding 模型 ──
    try:
        from sentence_transformers import SentenceTransformer
        if EMBEDDING_MODEL_PATH and os.path.isdir(EMBEDDING_MODEL_PATH):
            m = SentenceTransformer(EMBEDDING_MODEL_PATH)
            source = EMBEDDING_MODEL_PATH
        else:
            m = SentenceTransformer(EMBEDDING_MODEL_NAME)
            source = EMBEDDING_MODEL_NAME
        embedding = m
        print(f"[模型] ✅ 嵌入模型: {source}")
    except Exception:
        print("[模型] ⚠ 未检测到嵌入模型（可修改文件顶部的 EMBEDDING_MODEL_PATH 配置）")

    # ── 2. LLM 检测（环境变量 + 本地 Ollama）──
    llm, llm_name = _auto_detect_llm()

    # ── 3. 没检测到 → 询问用户 ──
    if llm is None:
        print("\n[模型] ⚠ 未检测到 LLM")
        ans = input("  是否配置 API Key 使用云端 LLM 辅助切分？(y/N): ").strip().lower()
        if ans == "y":
            llm, llm_name = _interactive_llm_setup()

    # ── 4. 确定可用级别 ──
    if llm is not None:
        level = "llm"
        print(f"[模型] ✅ LLM: {llm_name}")
    elif embedding is not None:
        level = "embedding"
    else:
        level = "none"
    print(f"[模型] 当前分块策略: {level.upper()}")

    return {"embedding": embedding, "llm": llm, "level": level}


def _auto_detect_llm():
    """自动检测 LLM：环境变量 → 本地运行时 → 本地模型权重文件"""
    import requests

    # ── 1. 环境变量（云端 API）──
    key_result = _detect_cloud_llm()
    if key_result:
        return key_result

    # ── 2. 本地运行时（Ollama/LM Studio/llama.cpp 等）──
    runtime_result = _detect_local_runtime()
    if runtime_result:
        return runtime_result

    # ── 3. 本地模型权重文件（transformers 直接加载）──
    weights_result = _detect_local_weights()
    if weights_result:
        return weights_result

    return None, None


def _detect_cloud_llm():
    """检测环境变量中配置的云端 API。"""
    providers = [
        ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com",
         "DEEPSEEK_MODEL", "deepseek-v4-flash", "DeepSeek"),
        ("OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1",
         None, "gpt-4o-mini", "OpenAI"),
        ("DASHSCOPE_API_KEY", None, "https://dashscope.aliyuncs.com/compatible-mode/v1",
         None, "qwen-turbo", "通义千问"),
    ]

    for key_name, base_name, default_base, model_env, default_model, label in providers:
        api_key = os.environ.get(key_name)
        if not api_key:
            continue
        base_url = os.environ.get(base_name, default_base) if base_name else default_base
        model = os.environ.get(model_env, default_model) if model_env else default_model

        # 用列表封装 _api_key，闭包捕获引用而不是值
        env_ref = [key_name, base_url, model]

        class LLMCaller:
            """封装 LLM 调用，避免闭包连接复用问题。"""
            def __init__(self, key_name, base_url, model):
                self.key_name = key_name
                self.base_url = base_url
                self.model = model

            def __call__(self, prompt, system=None):
                api_key = os.environ.get(self.key_name)
                if not api_key:
                    print(f"    [LLM] 环境变量 {self.key_name} 已失效")
                    return None
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                try:
                    import requests
                    # 每次新建 Session，避免连接复用导致的认证混淆
                    with requests.Session() as sess:
                        resp = sess.post(f"{self.base_url}/chat/completions", json={
                            "model": self.model, "messages": messages, "temperature": 0
                        }, headers=headers, timeout=60)
                        if resp.status_code != 200:
                            print(f"    [LLM API错误 {resp.status_code}] {resp.text[:100]}")
                            return None
                        return resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"    [LLM 调用失败] {e}")
                    return None

        return LLMCaller(key_name, base_url, model), label
    return None


def _make_cloud_llm_call(api_key, base, model):
    """构建云端 LLM 调用函数。"""
    import requests
    def call(prompt, system=None):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = requests.post(f"{base}/chat/completions", json={
                "model": model, "messages": messages, "temperature": 0
            }, timeout=30)
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    [LLM 调用失败] {e}")
            return None
    return call


def _detect_local_runtime():
    """检测本地 LLM 运行时服务。"""
    import requests

    runtimes = [
        ("Ollama", "http://localhost:11434/api/tags", None, "ollama"),
        ("LM Studio", "http://localhost:1234/v1/models", "http://localhost:1234/v1", "openai"),
        ("llama.cpp", "http://localhost:8080/v1/models", "http://localhost:8080/v1", "openai"),
        ("vLLM", "http://localhost:8000/v1/models", "http://localhost:8000/v1", "openai"),
        ("LocalAI", "http://localhost:8080/v1/models", "http://localhost:8080/v1", "openai"),
    ]

    for runtime_name, tags_url, api_base, api_type in runtimes:
        try:
            resp = requests.get(tags_url, timeout=2)
            if resp.status_code != 200:
                continue

            if runtime_name == "Ollama":
                models = [m["name"] for m in resp.json().get("models", [])]
            else:
                models = [m["id"] for m in resp.json().get("data", [])]

            if not models:
                continue

            # 优先选中文友好的模型
            chosen = _pick_best_local_model(models)
            return _make_local_llm_call(chosen, api_base, runtime_name, api_type), f"{runtime_name} ({chosen})"
        except Exception:
            continue
    return None


def _pick_best_local_model(models):
    """从可用模型列表中选中文能力最好的。"""
    for prefer in ["qwen", "Qwen", "yi", "Yi", "glm", "GLM", "deepseek", "DeepSeek"]:
        match = [m for m in models if prefer in m]
        if match:
            return match[0]
    return models[0]


def _make_local_llm_call(model, api_base, runtime, api_type):
    """构建本地运行时 LLM 调用函数。"""
    import requests

    def call(prompt, system=None):
        if api_type == "ollama":
            try:
                resp = requests.post(f"{api_base}/api/generate", json={
                    "model": model, "prompt": prompt, "system": system or "",
                    "stream": False, "options": {"temperature": 0}
                }, timeout=60)
                return resp.json().get("response", "")
            except Exception as e:
                print(f"    [{runtime} 调用失败] {e}")
                return None
        else:
            # OpenAI 兼容接口
            headers = {"Content-Type": "application/json"}
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            try:
                resp = requests.post(f"{api_base}/chat/completions", json={
                    "model": model, "messages": messages, "temperature": 0
                }, timeout=60)
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"    [{runtime} 调用失败] {e}")
                return None
    return call


# ── 可扫描的本地模型权重路径（从配置区读取）──
# 如果用户没配 LOCAL_MODEL_DIRS，用一些默认路径
_DEFAULT_MODEL_DIRS = [
    os.path.expanduser("~/.cache/huggingface/hub"),
]
if os.environ.get("HF_HOME"):
    _DEFAULT_MODEL_DIRS.append(os.environ["HF_HOME"])
if os.environ.get("HUGGINGFACE_HUB_CACHE"):
    _DEFAULT_MODEL_DIRS.append(os.environ["HUGGINGFACE_HUB_CACHE"])
LOCAL_MODEL_PATHS = (LOCAL_MODEL_DIRS if LOCAL_MODEL_DIRS else _DEFAULT_MODEL_DIRS) + _DEFAULT_MODEL_DIRS

# ── 优先检测的本地模型名称 ──
LOCAL_MODEL_NAMES = [
    # DeepSeek 系列
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    # Qwen 系列
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    # GLM 系列
    "THUDM/glm-4-9b-chat",
    # Yi 系列
    "01-ai/Yi-1.5-9B-Chat",
    # 通用降级
    "gpt2",
]


def _detect_local_weights():
    """
    检测本地 transformer 模型权重文件。
    扫描 LOCAL_MODEL_PATHS 下是否存在 DeepSeek/Qwen 等模型文件，
    有则用 transformers pipeline 加载。
    """
    try:
        import transformers
        import torch
    except ImportError:
        return None

    def scan_for_model(search_paths, model_names):
        """在指定路径下扫描模型文件。"""
        for base_path in search_paths:
            if not base_path or not os.path.isdir(base_path):
                continue
            for name in model_names:
                # model_names 是 "org/model" 格式
                short_name = name.split("/")[-1] if "/" in name else name
                # 在 models 目录下直接搜子文件夹名
                model_dir = os.path.join(base_path, short_name)
                if os.path.isdir(model_dir):
                    # 检查是否有模型文件
                    has_files = any(
                        f.endswith((".bin", ".safetensors", ".pt"))
                        for f in os.listdir(model_dir)[:5]
                    )
                    if has_files:
                        return name, model_dir
        return None, None

    model_name, model_path = scan_for_model(LOCAL_MODEL_PATHS, LOCAL_MODEL_NAMES)
    if not model_name:
        # 降级：尝试用 model_names 中的名字直接下载加载（如果用户网络好）
        return None

    print(f"  [本地模型] 发现: {model_name} ({model_path})")

    try:
        # 用小模型加载：device_map="auto", 4bit 量化如果 bitsandbytes 可用
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path if model_path else model_name,
            trust_remote_code=True
        )
        pipe = transformers.pipeline(
            "text-generation",
            model=model_path if model_path else model_name,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            max_new_tokens=512
        )

        def weights_call(prompt, system=None):
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            try:
                result = pipe(full_prompt, max_new_tokens=512, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
                return result[0]["generated_text"][len(full_prompt):].strip()
            except Exception as e:
                print(f"    [本地模型推理失败] {e}")
                return None

        return weights_call, f"本地模型 ({model_name})"
    except Exception as e:
        print(f"  [本地模型加载失败] {e}")
        return None


def _interactive_llm_setup():
    """交互式配置 LLM API。"""
    print("\n  支持的 LLM 服务：")
    print("    1) DeepSeek（便宜，推荐）")
    print("    2) 通义千问（阿里云）")
    print("    3) OpenAI")
    print("    4) 其他（兼容 OpenAI 接口）")
    choice = input("  请选择 (1-4): ").strip()

    providers = {
        "1": ("DeepSeek", "https://api.deepseek.com", "deepseek-chat"),
        "2": ("通义千问", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo"),
        "3": ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    }

    if choice in providers:
        label, base_url, model = providers[choice]
        key = input(f"  输入你的 {label} API Key: ").strip()
        if not key:
            print("  [跳过] 未输入 API Key，继续使用降级策略")
            return None, None

        def make_call(api_key=key, base=base_url, model=model):
            import requests
            def call(prompt, system=None):
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                try:
                    resp = requests.post(f"{base}/chat/completions", json={
                        "model": model, "messages": messages, "temperature": 0
                    }, timeout=60)
                    return resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"    [LLM 调用失败] {e}")
                    return None
            return call

        print(f"  ✅ {label} 配置完成")
        return make_call(), label

    elif choice == "4":
        base_url = input("  输入 API Base URL (如 https://api.xxx.com/v1): ").strip()
        model = input("  输入模型名称: ").strip()
        key = input("  输入 API Key: ").strip()
        if not key or not base_url:
            print("  [跳过] 信息不完整")
            return None, None

        def make_call(api_key=key, base=base_url, model=model):
            import requests
            def call(prompt, system=None):
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                try:
                    resp = requests.post(f"{base}/chat/completions", json={
                        "model": model, "messages": messages, "temperature": 0
                    }, timeout=60)
                    return resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"    [LLM 调用失败] {e}")
                    return None
            return call

        print(f"  ✅ 自定义 LLM 配置完成")
        return make_call(), f"自定义 ({model})"

    return None, None


# ═══════════════════════════════════════════════
# 动态阈值计算
# ═══════════════════════════════════════════════

def compute_max_chars(text, base=None):
    """
    分析文档段落长度分布，自动计算合适的阈值。
    锚点：BASE_THRESHOLD（用户配置区可修改）。
    """
    if base is None:
        base = BASE_THRESHOLD
    paragraphs = text.split("\n\n")
    lengths = [len(p.strip()) for p in paragraphs if p.strip()]

    if not lengths:
        return base, "无段落数据，使用初始值 500"

    median = statistics.median(lengths)
    lengths.sort()
    q75 = lengths[int(len(lengths) * 0.75)]

    # 取中位数和 75 分位数的较小值作为基准
    ref = min(median, q75) if len(lengths) > 2 else median
    # 基准 × 1.5 = 阈值
    computed = int(ref * 1.5)

    # 约束范围 [200, 1000]
    threshold = max(min(computed, base * 2), 200)

    print(f"[阈值] 段落中位数: {int(median)}字, 75分位: {q75}字")
    print(f"[阈值] 计算阈值: {computed}字, 最终: {threshold}字 (锚点{base})")

    return threshold, f"中位数{int(median)}×1.5={computed}, 锚点{base}约束"


# ═══════════════════════════════════════════════
# 阶段一：段落锚定
# ═══════════════════════════════════════════════

def parse_paragraphs(text, llm=None, short_para_limit=None):
    """
    按空行切段落 → 短段落合并 → 检测子结构。
    """
    if short_para_limit is None:
        short_para_limit = SHORT_PARA_LIMIT
    raw = text.split("\n\n")
    raw = [p.strip() for p in raw if p.strip()]

    merged = []
    i = 0
    while i < len(raw):
        p = raw[i]
        if len(p) >= short_para_limit:
            merged.append(p)
            i += 1
            continue

        # 收集连续短段落
        short_group = []
        while i < len(raw) and len(raw[i]) < short_para_limit:
            short_group.append(raw[i])
            i += 1

        if len(short_group) >= 3:
            merged.append("\n".join(short_group))
        elif llm is not None:
            result = _llm_merge_short_paragraphs(
                short_group, merged[-1] if merged else "",
                raw[i] if i < len(raw) else "", llm
            )
            merged.extend(result)
        else:
            merged.extend(short_group)

    sub_pattern = re.compile(r"（[一二三四五六七八九十百]+）")
    result = []
    for i, p in enumerate(merged):
        has_sub = bool(sub_pattern.findall(p))
        result.append({
            "para_id": f"para_{i:03d}",
            "text": p,
            "char_len": len(p),
            "has_sub_structure": has_sub
        })

    print(f"[段落] 原始 {len(raw)} 段 → 处理后 {len(result)} 段")
    for p in result:
        tag = " [含子结构]" if p["has_sub_structure"] else ""
        print(f"  {p['para_id']}: {p['char_len']}字{tag}")
    return result


def _llm_merge_short_paragraphs(short_group, prev_para, next_para, llm):
    context = ""
    if prev_para:
        context += f"上一段：{prev_para[:100]}\n"
    context += "\n".join([f"[{i}] {p}" for i, p in enumerate(short_group)])
    if next_para:
        context += f"\n下一段：{next_para[:100]}"

    prompt = f"""你是一个文档结构分析专家。以下是一组短段落，需要决定如何合并。

{context}

输出 JSON：
- 如果这些是标题/章节号 → 应合并到下一段: "action": "merge_down"
- 如果这些是上一段延续 → 合并到上一段: "action": "merge_up"
- 如果独立有意义 → "action": "keep"
- 如需要分组合并 → "action": "groups", "groups": [[0,1]]
"""
    response = llm(prompt, system="只输出 JSON，不做解释。")
    if not response:
        return short_group

    # 清理 LLM 输出，只取 JSON 部分
    clean = response.strip()
    if "{" in clean and "}" in clean:
        start = clean.index("{")
        end = clean.rindex("}") + 1
        clean = clean[start:end]

    try:
        data = json.loads(clean)
        if isinstance(data, list):
            return short_group
        action = data.get("action", "keep")
        if action == "merge_up":
            return [prev_para + "\n" + "\n".join(short_group)] if prev_para else short_group
        elif action == "merge_down":
            return ["\n".join(short_group) + "\n" + next_para] if next_para else short_group
        elif action == "groups":
            groups = data.get("groups", [])
            result = []
            for g in groups:
                valid = [i for i in g if 0 <= i < len(short_group)]
                if not valid:
                    continue
                result.append("\n".join([short_group[i] for i in valid]))
            return result if result else short_group
        return short_group
    except (json.JSONDecodeError, KeyError, TypeError):
        return short_group


# ═══════════════════════════════════════════════
# 阶段二：段内拆分
# ═══════════════════════════════════════════════

def split_paragraph(para_dict, models, max_chars, max_sub_chunks=None, min_chars=None):
    """
    段内拆分，动态阈值。
    有 LLM → LLM 粗切 + Embedding 细切
    有 Embedding → 句间相似度最低处切开
    无模型 → 句号硬切
    """
    if max_sub_chunks is None:
        max_sub_chunks = MAX_SUB_CHUNKS
    if min_chars is None:
        min_chars = MIN_CHUNK_CHARS
    text = para_dict["text"]
    para_id = para_dict["para_id"]
    has_sub = para_dict["has_sub_structure"]
    char_len = para_dict["char_len"]
    llm = models.get("llm")
    embedding = models.get("embedding")

    # ── 优先：子结构检测 ──
    if has_sub:
        parts = _split_by_sub_structure(text)
        if len(parts) > 1 and any(len(p) > 20 for p in parts):
            glued = _glue_numbered_chunks(parts)
            if llm is not None and len(glued) > 1:
                glued = _llm_merge_tiny_subchunks(glued, llm)
            if len(glued) > 1:
                return _package_chunks(glued, para_id, text)

    # ── 短段落（< 动态阈值）→ 整段保留 ──
    if char_len <= max_chars:
        return [{
            "chunk_id": f"{para_id}_c0",
            "text": text,
            "parent_text": text,
            "para_id": para_id,
            "sub_idx": 0,
            "total_subs": 1
        }]

    # ── 超长段落 → 需要拆分 ──
    if llm is not None:
        # Tier 1: LLM 粗切（逻辑大段）
        split_points = _llm_semantic_split(text, llm, max_sub_chunks)
        if split_points:
            big_parts = _apply_split_points(text, split_points)
            if len(big_parts) > 1:
                # 对每个大段再做 Embedding 细切
                final_parts = []
                for bp in big_parts:
                    if embedding is not None and len(bp) > max_chars:
                        sub = _embedding_semantic_split(bp, embedding, max_sub_chunks)
                        final_parts.extend(sub)
                    else:
                        final_parts.append(bp)
                # 如果拆得不够碎，用 Embedding 再切一轮
                if len(final_parts) <= 1 and embedding is not None:
                    final_parts = _embedding_semantic_split(text, embedding, max_sub_chunks)
                final_parts = _merge_tiny_chunks(final_parts, min_chars)
                if len(final_parts) > 1:
                    return _package_chunks(final_parts, para_id, text)

    if embedding is not None:
        # Tier 2: Embedding 语义切分
        parts = _embedding_semantic_split(text, embedding, max_sub_chunks)
        if parts and len(parts) > 1:
            parts = _merge_tiny_chunks(parts, min_chars)
            if len(parts) > 1:
                return _package_chunks(parts, para_id, text)

    # Tier 3: 句号硬切
    parts = _rule_split(text)
    parts = _merge_tiny_chunks(parts, min_chars)
    return _package_chunks(parts, para_id, text)


# ── 子结构 ──

def _split_by_sub_structure(text):
    parts = re.split(r"(?=（[一二三四五六七八九十百]+）)", text)
    return [p.strip() for p in parts if p.strip()]


def _glue_numbered_chunks(parts):
    glued = []
    buf = ""
    for p in parts:
        if re.match(r"^\s*（[一二三四五六七八九十百]+）", p):
            if buf.strip():
                glued.append(buf.strip())
            buf = p
        else:
            if buf:
                glued.append(buf + p)
                buf = ""
            else:
                glued.append(p)
    if buf.strip():
        glued.append(buf.strip())
    return [g for g in glued if g.strip()]


# ── Tier 1: LLM ──

def _llm_semantic_split(text, llm, max_sub_chunks=6):
    prompt = f"""分析这段话的逻辑结构，找出语义转折点。

{text}

输出 JSON（不要其他内容）：
{{"splits": [句子序号1, 句子序号2, ...]}}
- 句子按句号/感叹号/问号分割，从 0 编号
- 最多切 {max_sub_chunks} 块，splits 最多 {max_sub_chunks-1} 个
- 示例: {{"splits": [2, 5]}} 表示在句子2后和句子5后切开→3块
- 无转折→"splits": []
"""
    resp = llm(prompt, system="只输出 JSON，不做解释。只分析段落内部逻辑转折。")
    if not resp:
        return None
    try:
        data = json.loads(resp.strip())
        sp = data.get("splits", [])
        return sorted(set(sp)) if isinstance(sp, list) else None
    except (json.JSONDecodeError, KeyError):
        return None


def _llm_merge_tiny_subchunks(glued_chunks, llm):
    tiny = [i for i, c in enumerate(glued_chunks) if len(c) < 30]
    if len(tiny) < 2:
        return glued_chunks

    ctx = "\n".join([f"[{i}] {c}" for i, c in enumerate(glued_chunks)])
    prompt = f"""子块序列，有些太短(<30字)：
{ctx}

输出 JSON: {{"merge_pairs": [[1,2], ...]}}
短块合并到相邻的完整内容块。
"""
    resp = llm(prompt, system="只输出 JSON，不做解释。")
    if not resp:
        return glued_chunks
    try:
        data = json.loads(resp.strip())
        pairs = data.get("merge_pairs", [])
        merged = glued_chunks[:]
        for src, dst in sorted(pairs, key=lambda x: -min(x[0], x[1])):
            if max(src, dst) < len(merged) and merged[src] and merged[dst]:
                merged[dst] = merged[dst] + merged[src]
                merged[src] = None
        return [m for m in merged if m is not None]
    except (json.JSONDecodeError, KeyError):
        return glued_chunks


def _apply_split_points(text, split_points):
    sentences = re.split(r"(?<=[。！？])", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]

    parts = []
    start = 0
    for sp in split_points:
        if sp <= start or sp >= len(sentences):
            continue
        block = "".join(sentences[start:sp])
        if block.strip():
            parts.append(block.strip())
        start = sp
    remain = "".join(sentences[start:])
    if remain.strip():
        parts.append(remain.strip())
    return parts if parts else [text]


# ── Tier 2: Embedding ──

def _embedding_semantic_split(text, embedding, max_sub_chunks=6):
    sentences = re.split(r"(?<=[。！？])", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text]
    if len(sentences) <= max_sub_chunks:
        return sentences

    vecs = embedding.encode(sentences)

    norms = [sum(v**2)**0.5 for v in vecs]
    sims = []
    for i in range(len(vecs) - 1):
        s = vecs[i] @ vecs[i+1] / (norms[i] * norms[i+1] + 1e-8)
        sims.append(s)

    n = min(max_sub_chunks - 1, len(sims))
    if n <= 0:
        return sentences

    cuts = sorted(range(len(sims)), key=lambda i: sims[i])[:n]
    cuts.sort()

    parts = []
    start = 0
    for c in cuts:
        block = "".join(sentences[start:c+1])
        if block.strip():
            parts.append(block.strip())
        start = c + 1
    remain = "".join(sentences[start:])
    if remain.strip():
        parts.append(remain.strip())
    return parts if parts else [text]


# ── Tier 3: 规则 ──

def _rule_split(text):
    sentences = re.split(r"(?<=[。！？])", text)
    return [s.strip() for s in sentences if s.strip()]


# ── 后处理 ──

def _merge_tiny_chunks(chunks, min_chars=50):
    merged = []
    buf = ""
    for c in chunks:
        if len(c) < min_chars and buf:
            buf += c
        else:
            if buf:
                merged.append(buf)
            buf = c
    if buf:
        if merged:
            merged[-1] += buf
        else:
            merged.append(buf)
    return [m for m in merged if m.strip()]


def _package_chunks(parts, para_id, full_text):
    result = []
    for i, p in enumerate(parts):
        if not p.strip():
            continue
        result.append({
            "chunk_id": f"{para_id}_c{i}",
            "text": p.strip(),
            "parent_text": full_text,
            "para_id": para_id,
            "sub_idx": i,
            "total_subs": len(parts)
        })
    if len(result) > 6:
        extra = "".join([r["text"] for r in result[5:]])
        result = result[:5]
        result.append({
            "chunk_id": f"{para_id}_c5",
            "text": extra,
            "parent_text": full_text,
            "para_id": para_id,
            "sub_idx": 5,
            "total_subs": 6
        })
    return result


# ═══════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════

def chunk_document(text, models, max_chars=None):
    """全自动管道。max_chars=None 时自动计算动态阈值。"""
    if max_chars is None:
        max_chars, _ = compute_max_chars(text)

    paragraphs = parse_paragraphs(text, llm=models.get("llm"))
    all_chunks = []
    for p in paragraphs:
        cs = split_paragraph(p, models, max_chars)
        all_chunks.extend(cs)

    print(f"\n[分块] 共 {len(all_chunks)} 个子块")
    for c in all_chunks[:5]:
        print(f"  {c['chunk_id']}: {c['text'][:40]}... ({len(c['text'])}字)")
    return all_chunks


# ═══════════════════════════════════════════════
# Chroma 存储
# ═══════════════════════════════════════════════

def store_chunks(chunks, embedding=None, collection_name=None):
    if collection_name is None:
        collection_name = COLLECTION_NAME

    # 持久化存储（有 CHROMA_PERSIST_DIR 时用磁盘模式）
    if CHROMA_PERSIST_DIR and CHROMA_PERSIST_DIR.strip():
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        print(f"[Chroma] 持久化模式: {CHROMA_PERSIST_DIR}")
    else:
        client = chromadb.Client()
        print("[Chroma] 内存模式（进程结束数据丢失）")

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name)

    documents = [c["text"] for c in chunks]
    metadatas = [{
        "para_id": c["para_id"],
        "parent_text": c["parent_text"],
        "sub_idx": c["sub_idx"],
        "total_subs": c["total_subs"]
    } for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    if embedding is not None:
        embeds = embedding.encode(documents).tolist()
        collection.add(documents=documents, embeddings=embeds, metadatas=metadatas, ids=ids)
        print(f"[Chroma] 存入 {collection.count()} 个子块（含向量检索）")
    else:
        # 没有 embedding 模型，不传 documents 也不传 embeddings
        # Chroma 会报错，直接跳过存储，只返回 chunks
        print("[Chroma] 未配置嵌入模型，跳过 Chroma 存储")
        print("[Chroma] 仅展示分块结果，无法做向量检索")
        return None

    print(f"[Chroma] 存入 {collection.count()} 个子块")
    return collection


# ═══════════════════════════════════════════════
# 检索（父段落去重）
# ═══════════════════════════════════════════════

def search(query, collection, embedding, top_k=5):
    qv = embedding.encode([query]).tolist()
    results = collection.query(query_embeddings=qv, n_results=top_k)

    seen = set()
    deduped = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        parent = meta["parent_text"]
        if parent in seen:
            continue
        seen.add(parent)
        deduped.append({
            "rank": len(deduped) + 1,
            "distance": results["distances"][0][i],
            "original_chunk": doc,
            "full_parent": parent,
            "para_id": meta["para_id"]
        })
    return deduped


def print_search_results(query, results):
    print(f"\n{'='*50}")
    print(f"问题: {query}")
    print(f"{'='*50}")
    for r in results:
        print(f"\n  >> Top-{r['rank']} (距离: {r['distance']:.4f}) [段落 {r['para_id']}]")
        print(f"    命中子块: {r['original_chunk'][:80]}...")
        print(f"    完整段落: {r['full_parent'][:120]}...")


# ═══════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text_path = sys.argv[1]
    else:
        print("用法: python rigid_chunker.py <文本文件路径>")
        print("例如: python rigid_chunker.py 我的文档.txt")
        sys.exit(1)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"全文: {len(text)}字符\n")

    models = detect_models()

    # 动态阈值（锚点 500）
    max_chars, reason = compute_max_chars(text)
    print(f"[阈值] 策略: {reason}\n")

    chunks = chunk_document(text, models, max_chars=max_chars)
    collection = store_chunks(chunks, embedding=models["embedding"])

    # 如果有向量库，跑几个示例检索（可自行修改测试问题）
    if collection is not None:
        test_queries = [
            "修改这里为你的第一个测试问题",
            "第二个测试问题",
        ]
        print(f"\n{'='*50}\n检索示例（修改上方 test_queries 列表即可）\n{'='*50}")
        for q in test_queries:
            res = search(q, collection, models["embedding"])
            print_search_results(q, res)
