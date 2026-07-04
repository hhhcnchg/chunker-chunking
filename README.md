# Rigid Chunker 🧩

<p align="center">
  <b>刚性段落锁定 + 柔性语义内拆</b><br>
  一种创新的 RAG 文本分块策略<br><br>
  <img src="https://img.shields.io/badge/python-3.8+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/RAG-chunking-orange" />
  <a href="https://hhhcnchg.github.io/chunker-chunking/"><img src="https://img.shields.io/badge/🌐_项目主页-Live-brightgreen" /></a>
  <a href="README.en.md"><img src="https://img.shields.io/badge/README-English-0366d6" /></a>
</p>

---

## 它是什么

Rigid Chunker 是一个面向 RAG（检索增强生成）的文本分块器，核心设计只有一个原则：

> **段落边界是高压线，永不跨越。**

传统分块方案（如 LangChain 的 `RecursiveCharacterTextSplitter`）按固定字数切割文本，经常把一句话切成两半、把一个论点拆到两个 chunk 里。Rigid Chunker 用自然段落作为刚性边界，只在段落内部做语义拆分，从根本上杜绝跨段落碎片。

## 为什么需要它

RAG 系统的检索质量取决于分块质量。碎片化的 chunk 会让 LLM 拿到不完整的上下文，生成错误答案。Rigid Chunker 通过三个机制保证分块质量：

1. **段落锁定** — 每个自然段落是一个独立单元，绝不跨越
2. **子块 + 父段落** — 子块用于精细检索，父段落用于生成时提供完整上下文
3. **三模型降级** — 有 LLM 用 LLM，没有用 Embedding，都没有用规则，不会挂掉

## 核心特性

| 特性 | 说明 |
|:-----|:-----|
| 🔒 **段落锁定** | 自然段落为绝对边界，永不跨段切分 |
| 🧱 **子结构识别** | 自动识别 `（一）（二）（三）` 等序号结构，每条独立成块 |
| 📦 **父段落存储** | 子块带完整父文本，检索时自动替换 |
| 🔍 **模型检测链** | 自动检测 LLM → Embedding → 规则，逐级降级 |
| 🔗 **智能合并** | 短段落自动合并，目录/标题行智能附着 |
| 📊 **动态阈值** | 基于文档段落长度分布自动计算最适切分点 |
| ⚡ **无需 LLM** | 有 Embedding 模型即可工作，LLM 非必需 |
| 🇨🇳 **中文优化** | 针对中文段落结构、序号体系专门设计 |

## 与现有方案对比

| 方面 | Rigid Chunker | LumberChunker | Meta-Chunking |
|:-----|:--------------|:--------------|:--------------|
| **段落锁定** | ✅ 刚性边界 | ❌ 可能跨段 | ❌ 可能跨段 |
| **LLM 依赖** | 可选（自动降级） | 必需 | 必需 |
| **子结构识别** | ✅ 自动 | ❌ 无 | ❌ 无 |
| **父段落存储** | ✅ 有 | ❌ 无 | ❌ 无 |
| **动态阈值** | ✅ 自动 | 固定 | 固定 |
| **模型成本** | 低（Embedding 为主） | 高（逐段调 LLM） | 高（边界调 LLM） |
| **确定性** | 高（段落锁定） | 中（LLM 输出波动） | 中（LLM 输出波动） |
| **中文优化** | ✅ 是 | ❌ 英文优先 | ❌ 英文优先 |

## 快速体验

```bash
pip install -r requirements.txt
python rigid_chunker.py 你的文档.txt
```

> 详细安装、配置、参数说明请查看 [USAGE.md](USAGE.md) | [English](README.en.md)

## 项目结构

```
chunker-chunking/
├── rigid_chunker.py            # 核心分块器
├── download_model.py           # 模型下载工具
├── .config.example             # 配置模板
├── requirements.txt            # Python 依赖
├── LICENSE                     # MIT 许可证
├── README.md                   # 本文件（项目介绍）
├── USAGE.md                    # 详细使用说明
├── README.en.md                # English version
├── docs/
│   └── index.html              # 项目主页
└── .github/
    └── workflows/
        └── pages.yml           # 自动部署
```

## License

[MIT](LICENSE)

---

🌐 [项目主页](https://hhhcnchg.github.io/chunker-chunking/) · 📖 [使用说明](USAGE.md) · [English →](README.en.md)
