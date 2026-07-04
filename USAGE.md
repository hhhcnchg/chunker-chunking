# Rigid Chunker 使用说明

## 目录

- [安装](#安装)
- [配置](#配置)
- [运行](#运行)
- [参数说明](#参数说明)
- [工作原理](#工作原理)
- [API 使用](#api-使用)
- [常见问题](#常见问题)

---

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/hhhcnchg/chunker-chunking.git
cd chunker-chunking
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

依赖列表：

| 包 | 用途 | 必需？ |
|:---|:-----|:-------|
| `chromadb` | 向量数据库存储 | 是 |
| `sentence-transformers` | Embedding 模型加载 | 是 |
| `requests` | LLM API 调用 | 是 |

### 3. 下载嵌入模型（可选）

默认运行时会自动从 HuggingFace 下载 `BAAI/bge-small-zh-v1.5`。
国内用户可用镜像手动下载：

```bash
python download_model.py
```

下载完成后模型在 `~/models/bge-small-zh-v1.5`。

---

## 配置

### 基本配置

编辑 `rigid_chunker.py` 顶部的配置区：

```python
# 嵌入模型
EMBEDDING_MODEL_PATH = ""                    # 本地模型路径，留空自动下载
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # HuggingFace 模型名

# 分块参数
MAX_SUB_CHUNKS = 6        # 每段最多拆成几块
MIN_CHUNK_CHARS = 50      # 子块最小字符数（太短会合并）
SHORT_PARA_LIMIT = 50     # 短段落阈值（低于此字数尝试合并）
BASE_THRESHOLD = 500      # 动态阈值锚点

# Chroma
COLLECTION_NAME = "rigid_chunker_demo"
CHROMA_PERSIST_DIR = ""   # 持久化目录，留空 = 内存模式
```

### 配置 LLM（可选）

三种方式：

**方式一：环境变量（推荐）**

```bash
# DeepSeek
export DEEPSEEK_API_KEY="sk-xxx"
export DEEPSEEK_MODEL="deepseek-v4-flash"

# 或 OpenAI
export OPENAI_API_KEY="sk-xxx"

# 或通义千问
export DASHSCOPE_API_KEY="sk-xxx"
```

**方式二：交互问答**

运行时如果检测不到 LLM，会询问是否配置 API Key。按提示输入即可。

**方式三：本地运行时**

Ollama、LM Studio、llama.cpp、vLLM 等本地运行时会自动检测，无需额外配置。

### 高级配置

复制配置模板进行自定义：

```bash
cp .config.example .config.py
```

`.config.example` 中有所有可配置项的说明。

---

## 运行

### 基本用法

```bash
python rigid_chunker.py 你的文档.txt
```

程序会：
1. 自动检测可用的模型（LLM / Embedding / 规则）
2. 分析文档段落长度分布，计算动态阈值
3. 执行分块
4. 存入 Chroma 向量数据库
5. 运行示例检索

### 输出示例

```
全文: 7736字符

[模型] ✅ 嵌入模型: d:/models/bge-small-zh-v1.5
[模型] ⚠ 未检测到 LLM
是否配置 API Key 使用云端 LLM 辅助切分？(y/N): y
  1) DeepSeek（便宜，推荐）
  2) 通义千问（阿里云）
  3) OpenAI
  请选择 (1-4): 1
  输入你的 DeepSeek API Key: sk-xxx
  ✅ DeepSeek 配置完成
[模型] ✅ LLM: DeepSeek
[模型] 当前分块策略: LLM

[阈值] 段落中位数: 214字, 75分位: 374字
[阈值] 计算阈值: 321字, 最终: 321字 (锚点500)

[段落] 原始 22 段 → 处理后 20 段
  para_000: 6字
  para_001: 80字
  ...

[分块] 共 35 个子块
[Chroma] 持久化模式: ~/chroma_data
[Chroma] 存入 35 个子块

==================================================
问题: 长期护理保险多少钱
==================================================

  >> Top-1 (距离: 0.3821) [段落 para_005]
    命中子块: 第十五条 长期护理保险...
    完整段落: 第十五条 长期护理保险制度...
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `EMBEDDING_MODEL_PATH` | `""` | 本地模型路径，留空从 HuggingFace 自动下载 |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | HuggingFace 模型名 |
| `LOCAL_MODEL_DIRS` | `[]` | 本地 LLM 权重扫描目录 |
| `MAX_SUB_CHUNKS` | `6` | 每段最多拆多少块 |
| `MIN_CHUNK_CHARS` | `50` | 子块最小字符数，低于此值会合并 |
| `SHORT_PARA_LIMIT` | `50` | 短段落阈值，低于此值的段落会尝试合并 |
| `BASE_THRESHOLD` | `500` | 动态阈值锚点，影响段内拆分粒度 |
| `COLLECTION_NAME` | `rigid_chunker_demo` | Chroma 集合名称 |
| `CHROMA_PERSIST_DIR` | `""` | Chroma 持久化目录，留空为内存模式 |

### 动态阈值说明

`BASE_THRESHOLD` 控制"多长的段落需要拆分"。系统会根据文档段落长度分布自动计算：

```
动态阈值 = min(median, Q75) × 1.5
最终阈值 = clamp(动态阈值, 200, BASE_THRESHOLD × 2)
```

- 文档段落普遍较短 → 阈值自动调低，少拆分
- 文档段落普遍较长 → 阈值自动调高，多拆分
- 想更倾向拆分：调小 `BASE_THRESHOLD`
- 想更倾向保留：调大 `BASE_THRESHOLD`

---

## 工作原理

### 阶段一：段落锚定

1. 按空行切分原始文本为段落
2. 短段落检测：低于 `SHORT_PARA_LIMIT` 字的连续短段落会被智能合并
   - 有 LLM 时：LLM 判断是合并到上一段、下一段、还是保持独立
   - 无 LLM 时：3 个以上连续短段落直接拼接
3. 子结构扫描：检测 `（一）（二）（三）` 等序号标记

### 阶段二：段内拆分

对每个段落，按以下优先级拆分：

1. **有子结构** → 按 `（一）（二）（三）` 拆成独立子块
2. **短段落**（< 动态阈值）→ 整段保留为一个 chunk
3. **超长段落**：
   - Tier 1（有 LLM）：LLM 识别语义转折点粗切 → Embedding 细切每个大段
   - Tier 2（有 Embedding）：计算句间余弦相似度，在最低点切开
   - Tier 3（无模型）：按句号硬切

### 阶段三：存储与检索

分块结果存入 Chroma：

```python
{
    "chunk_id": "para_005_c1",      # 唯一标识
    "text": "第十五条 ...",          # 子块文本（用于检索）
    "parent_text": "第十五条 ...",   # 完整父段落（用于生成）
    "para_id": "para_005",          # 段落编号
    "sub_idx": 1,                   # 段内序号
    "total_subs": 3                 # 该段共几个子块
}
```

检索时自动用 `parent_text` 替换子块，保证 LLM 拿到完整上下文。

---

## API 使用

在自己的 Python 代码中使用：

```python
from rigid_chunker import detect_models, chunk_document, store_chunks, search

# 1. 检测模型
models = detect_models()

# 2. 读取文档
with open("文档.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 3. 分块
chunks = chunk_document(text, models)

# 4. 存入 Chroma
collection = store_chunks(chunks, embedding=models["embedding"])

# 5. 检索
if collection:
    results = search("你的问题", collection, models["embedding"])
    for r in results:
        print(f"[{r['para_id']}] {r['full_parent'][:100]}")
```

### 只用分块，不用 Chroma

```python
from rigid_chunker import chunk_document, detect_models

models = detect_models()
chunks = chunk_document(text, models)

for c in chunks:
    print(c["chunk_id"], len(c["text"]), c["text"][:50])
```

### 自定义参数

```python
chunks = chunk_document(
    text,
    models,
    max_chars=300    # 自定义阈值，不使用动态计算
)
```

---

## 常见问题

### Q: 模型下载太慢怎么办？

```bash
# 使用国内镜像
python download_model.py

# 或手动设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
python rigid_chunker.py 文档.txt
```

### Q: 没有 LLM 能用吗？

完全可以。Tier 2（Embedding）和 Tier 3（规则）都不需要 LLM。只有遇到特别长的段落时，LLM 才能提供更准确的语义切分点。

### Q: Chroma 持久化和内存模式有什么区别？

- **内存模式**（默认）：数据存在内存中，程序结束即丢失。适合测试。
- **持久化模式**：数据存在磁盘上，重启后保留。适合生产环境。

```python
CHROMA_PERSIST_DIR = "~/chroma_data"   # 开启持久化
CHROMA_PERSIST_DIR = ""                # 内存模式
```

### Q: 如何用自己下载的模型？

把模型放在任意目录，修改配置：

```python
EMBEDDING_MODEL_PATH = "d:/models/bge-large-zh-v1.5"
```

### Q: 支持英文文档吗？

支持。分块逻辑是语言无关的。但子结构识别（`（一）（二）`）是针对中文设计的。英文的 `1. 2. 3.` 序号在后续版本中会支持。

### Q: 分块结果怎么接入 LangChain？

分块结果是标准的字典列表，可以直接转换为 LangChain 的 `Document` 对象：

```python
from langchain.schema import Document

chunks = chunk_document(text, models)
documents = [
    Document(
        page_content=c["text"],
        metadata={"para_id": c["para_id"], "parent_text": c["parent_text"]}
    )
    for c in chunks
]
```
