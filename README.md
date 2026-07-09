# 🧪 AI 多模型分类算法黑箱可视化探究沙盒

> **从"一言堂"到"百家争鸣"——同一份数据，四种算法，四个决策边界**

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://aimultimodellab-h8xoktgasa2zvxvxqd4hyx.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 📖 项目简介

这是一个面向 AI 初学者的**多模型决策对比可视化工具**。它将机器学习最常见的四种分类算法——KNN、决策树、SVM、逻辑回归——**放在同一擂台**上，让使用者直观地观察：

- ✅ 各算法的决策边界有何不同（边界形状、走向、鲁棒性）
- ❌ 面对噪声/干扰时，谁先「扛不住」
- 🤝 "共识 vs 分歧"背后的算法本质

### 🎯 核心理念

> 没有"最好"的算法，只有最合适的场景。
> **多模型投票**往往比单一模型更可靠。

---

## ✨ 功能特色

### 🔬 四模型同台竞技

在完全相同的训练数据上同时训练 **KNN / 决策树 / SVM / 逻辑回归**四种模型，以 **2×2 子图矩阵**直观对比决策边界差异。

### 🎛️ 交互式实验控制台

侧边栏提供两组"破坏器"滑块：

| 破坏器 | 参数 | 作用 |
|--------|------|------|
| 🎨 **非结构化感知破坏** | 图像模糊 (0-15级) | 模拟拍摄失焦、雨雾等感知退化 |
| 🎨 **非结构化感知破坏** | 局部遮挡 (0-80%) | 模拟物体被遮挡的面积比例 |
| 📊 **结构化数据破坏** | 特征重叠度 (0-0.9) | 拉高 → 好次品长太像，AI 分不清 |
| 📊 **结构化数据破坏** | 标注投毒率 (0-0.3) | 拉高 → 人工贴错标签，AI 学坏 |

### 🎯 共识仪表盘

实时显示四种模型的判定结果：
- ✅ **全体一致** → 问题简单明确，或干扰太强全倒了
- ⚡ **出现分歧** → 问题处于临界区，需要人工复核
- 💥 **全体误判** → 干扰已超出所有模型的承受极限

### 🖼️ 盖章动画反馈

当模型集体"看走眼"时，样本图上会弹出红色检验印章动画（带呼吸光效），视觉反馈直观有趣。

### 🤖 AI 实验解读 + 智能问答助教

- **实验解读**：一键调用 LLM 用通俗语言解释当前实验结果
- **问答助教**：学生对实验有疑问随时问，助教结合当前参数回答

支持主流国产大模型后端：Ollama（本地）、DeepSeek、SiliconFlow（含免费额度）。

### 🌾 多场景切换

内置多种贴近生活的教学场景，一键切换：

| 场景 | 正例 | 负例 | 样本图 |
|------|------|------|--------|
| 🍑 荔枝品质分级 | 一级果 | 次级果 | 🖼️ |
| 🍵 茶叶品质鉴别 | 优质茶 | 劣质茶 | 🖼️ |
| 🌾 水稻病害诊断 | 健康稻株 | 染病稻株 | 🖼️ |

### 📊 实验记录与导出

自动记录每次实验的参数与结果，支持导出 CSV 格式实验报告，便于课堂回顾。

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- pip

### 安装

```bash
# 克隆仓库
git clone https://github.com/wumingzhao/ai_multi_model_lab.git
cd ai_multi_model_lab

# 安装依赖
pip install -r requirements.txt
```

### 配置 LLM（可选）

编辑 `config.json` 中的 `llm` 字段，支持三种后端：

#### 选项 A：Ollama（本地，完全免费）

```json
"llm": {
    "provider": "ollama",
    "api_url": "http://localhost:11434",
    "api_key": "",
    "model": "qwen2.5:7b"
}
```

需提前安装 [Ollama](https://ollama.com) 并拉取模型：
```bash
ollama pull qwen2.5:7b
```

#### 选项 B：SiliconFlow（在线，有免费额度）

```json
"llm": {
    "provider": "siliconflow",
    "api_url": "https://api.siliconflow.cn/v1",
    "api_key": "你的API密钥",
    "model": "Qwen/Qwen2.5-7B-Instruct"
}
```

#### 选项 C：DeepSeek

```json
"llm": {
    "provider": "deepseek",
    "api_url": "https://api.deepseek.com",
    "api_key": "你的API密钥",
    "model": "deepseek-chat"
}
```

> **💡 智能提示**：不配置 LLM 不影响核心可视化功能，仅 AI 解读和问答不可用。

### 运行

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

---

## 🧠 四种算法简介

| 算法 | 通俗理解 | 边界特点 |
|------|---------|---------|
| **KNN (K近邻)** | "看看邻居怎么说" | 以距离投票，对局部密度敏感，边界不规则 |
| **决策树** | "按规则问到底" | 逐层 if-else 判断，边界呈阶梯状 |
| **SVM (支持向量机)** | "找最宽分界线" | 追求最大间隔，受边界点影响大，RBF核边界弯曲 |
| **逻辑回归** | "算概率定阈值" | 线性分界，简单但表达能力有限 |

---

## 📁 项目结构

```
ai_multi_model_lab/
├── app.py                    # 主程序（Streamlit 应用）
├── llm_helper.py             # LLM 辅助模块（实验解读 + 问答助教）
├── config.json               # 运行时配置（场景 + LLM 设置）
├── requirements.txt          # Python 依赖
├── .gitignore
├── __init__.py
├── fonts/
│   └── SimHei.ttf            # 中文字体（黑体）
├── assets/
│   ├── lichi.png             # 荔枝样本图
│   ├── tea.png               # 茶叶样本图
│   ├── crop.png              # 水稻样本图
│   ├── guangcai.png          # 广彩样本图（预留）
│   └── seal.png              # 红色检验印章
├── config-sample-fruit.json  # 荔枝品质分级场景模板
├── config-sample-tea.json    # 茶叶品质鉴别场景模板
└── config-sample-crop.json   # 水稻病害诊断场景模板
```

---

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| [Streamlit](https://streamlit.io) | 交互式 Web UI 框架 |
| [scikit-learn](https://scikit-learn.org) | 四种分类算法实现 |
| [Matplotlib](https://matplotlib.org) | 决策边界可视化绘图 |
| [NumPy](https://numpy.org) / [Pandas](https://pandas.dev) | 数据处理与分析 |
| [Pillow](https://python-pillow.org) | 图像模糊/遮挡处理 |
| [httpx](https://www.python-httpx.org) | LLM API 调用 |

---

## 🎓 教学应用场景

本项目特别适合以下场景：

1. **AI 入门课程** — 直观理解不同算法的决策边界差异
2. **机器学习实验课** — 探究噪声/干扰对不同模型的影响
3. **模型鲁棒性教学** — 理解"过拟合"、"欠拟合"的视觉表现
4. **数据质量教育** — 展示标注错误（投毒）对模型的影响
5. **科普展览** — 互动展项，零代码上手体验 AI

---

## 📸 界面预览

| 区域 | 说明 |
|------|------|
| 🎛️ 左侧边栏 | 四种实验参数滑块 + 场景切换 + AI 问答助教 |
| 🖼️ 左栏上半 | 样本图（受模糊/遮挡影响）+ 印章反馈动画 |
| 🎯 左栏下半 | 模型共识仪表盘（✅/❌ + 准确率） |
| 📐 右栏 | 2×2 四模型决策边界对比图 + 详细指标表 |
| 📊 底部 | 实验记录表格 + CSV 导出 |

---

## 🤝 贡献指南

欢迎贡献！你可以：

- 🐛 提交 [Issue](https://github.com/wumingzhao/ai_multi_model_lab/issues) 报告 Bug 或提出建议
- 🌟 提交 Pull Request 改进代码或新增场景
- 🌾 贡献新的教学场景（参考 `config-sample-*.json` 格式）

---

## 📄 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢 [Streamlit](https://streamlit.io) 提供了优秀的交互式数据应用框架
- 感谢 [scikit-learn](https://scikit-learn.org) 社区提供的经典算法实现
- 所有样本图片为 AI 生成，仅供教学演示使用

---

<p align="center">
  <strong>从「凭眼力」到「AI分级」—— 让算法决策不再是一个黑箱 🌟</strong>
</p>
