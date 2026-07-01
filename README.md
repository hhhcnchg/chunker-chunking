# Rigid Chunker 🧩

**刚性段落锁定 + 柔性语义内拆** — 一种创新的 RAG 文本分块策略。

以自然段落为「高压线」永不跨越，仅在段落内部做语义拆分。专为解决传统分块中**语义断裂**和**跨段落碎片**问题而设计。

## 核心思想

```
段落边界 ── 永不跨越 ── ── ── ── ── ── ── ── 
            ┌──────────────────────┐
  子块 A    │  完整语义单元 A       │  ← 子块用于检索匹配
            └──────────────────────┘
            ┌──────────────────────┐
  子块 B    │  完整语义单元 B       │
            └──────────────────────┘
            ┌──────────────────────┐
  父段落    │  全文（包含 A+B）     │  ← 完整上下文用于生成
            └──────────────────────┘
```

- **子块**匹配时提供精细定位
- **父段落**在生成阶段替换上去，避免只看碎片丢失上下文

## 核心特性

| 特性 | 说明 |
|------|------|
| **段落锁定** | 自然段落为绝对边界，永不跨段切分 |
| **子结构识别** | 自动识别 `（一）（二）（三）` 等序号结构，每条独立成块 |
| **父段落存储** | 子块带完整父文本，检索时自动替换 |
| **模型检测链** | 自动检测 LLM → Embedding → 规则，逐级降级 |
| **智能合并** | 短段落自动合并，目录/标题行智能附着 |
| **动态阈值** | 基于文档段落长度分布自动计算最适切分点 |
| **无需 LLM** | 有 Embedding 模型即可工作，LLM 非必需 |

## 三模型检测链

```
Tier 1: LLM   → 语义粗切大段 + Embedding 细切
Tier 2: Embed → bge 算句间相似度，最低处切开
Tier 3: 规则  → 句号硬切兜底
```

检测不到 LLM 时会**询问用户是否配置 API Key**，不会默默降级。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载嵌入模型（可选）

项目默认从 HuggingFace 自动下载 `BAAI/bge-small-zh-v1.5`。
国内用户可用镜像：

```bash
python download_model.py
```

### 配置 LLM（可选）

支持三种方式：

1. **环境变量**（推荐）：
   ```bash
   export DEEPSEEK_API_KEY="sk-xxx"
   export DEEPSEEK_MODEL="deepseek-v4-flash"
   ```
2. **交互问答**：运行时会自动询问是否配置
3. **本地模型**：Ollama、LM Studio 等运行时会自动检测

### 运行

```bash
# 对文档分块
python rigid_chunker.py 你的文档.txt

# 导入自己的配置（可选）
cp .config.example .config.py
# 编辑 .config.py 定制参数
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL_PATH` | `""` | 本地模型路径，留空自动下载 |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | HuggingFace 模型名 |
| `MAX_SUB_CHUNKS` | `6` | 每段最多拆多少块 |
| `MIN_CHUNK_CHARS` | `50` | 子块最小字符数 |
| `SHORT_PARA_LIMIT` | `50` | 短段落阈值 |
| `BASE_THRESHOLD` | `500` | 动态阈值锚点 |
| `COLLECTION_NAME` | `rigid_chunker_demo` | Chroma 集合名 |

## 工作原理

### 阶段一：段落锚定

1. 按空行切分原始段落
2. 短段落检测与智能合并（有 LLM 时用 LLM 判断合并策略）
3. 扫描子结构标记（`（一）...（十）`）

### 阶段二：段内拆分

1. 检测到子结构 → 按序拆分，每条独立成块
2. 短段落（< 动态阈值）→ 整段保留
3. 超长段落：
   - **有 LLM**：LLM 粗切语义大段 → Embedding 细切
   - **有 Embedding**：句间余弦相似度最低处切开
   - **兜底**：句号硬切

### 阶段三：存储与检索

- Chroma 向量数据库存储
- 子块用于检索匹配
- 父段落文本用于生成阶段替换
- 检索结果按父段落去重

## 与 LumberChunker / Meta-Chunking 对比

| 方面 | Rigid Chunker | LumberChunker | Meta-Chunking |
|------|---------------|---------------|---------------|
| **段落锁定** | ✅ 刚性边界 | ❌ 可能跨段 | ❌ 可能跨段 |
| **LLM 依赖** | 可选（自动降级） | 必需 | 必需 |
| **子结构识别** | ✅ 自动 | ❌ 无 | ❌ 无 |
| **父段落存储** | ✅ 有 | ❌ 无 | ❌ 无 |
| **动态阈值** | ✅ 自动 | 固定 | 固定 |
| **模型成本** | 低（Embedding 为主） | 高（逐段调 LLM） | 高（边界调 LLM） |
| **确定性** | 高（段落锁定） | 中（LLM 输出波动） | 中（LLM 输出波动） |
| **中文优化** | ✅ 是 | ❌ 英文优先 | ❌ 英文优先 |

## 项目结构

```
chunker-chunking/
├── rigid_chunker.py      # 核心分块器
├── download_model.py     # 模型下载工具
├── .config.example       # 配置模板
├── requirements.txt      # Python 依赖
├── LICENSE               # MIT 许可证
└── README.md             # 使用说明
```

## License

MIT

---

[English version →](README.en.md)
