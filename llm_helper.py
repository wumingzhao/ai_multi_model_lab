"""
LLM 辅助模块 — 支持 AI 实验解读 + AI 智能问答助教

支持的 LLM 后端（国产大模型）：
- ollama（本地运行 qwen2.5，完全免费）
- deepseek（在线 API，需 key）
- siliconflow（在线 API，免费额度）
"""

import streamlit as st
import httpx
import json
import os

# ── 默认超时 ──
TIMEOUT = 30


def _resolve_api_key(cfg_key: str) -> str:
    """按优先级读取 API Key：st.secrets → 环境变量 → config.json"""
    # 1. st.secrets（Streamlit Cloud）
    try:
        return st.secrets.get("llm", {}).get("api_key", "") or cfg_key
    except Exception:
        pass
    # 2. 环境变量（飞桨 AI Studio / 服务器部署）
    env_key = os.environ.get("LLM_API_KEY", "")
    if env_key:
        return env_key
    # 3. config.json 中的值
    return cfg_key


def _build_client(api_url: str, api_key: str = ""):
    """构建 httpx 客户端，设置通用请求头"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return httpx.Client(base_url=api_url.rstrip("/"), headers=headers, timeout=TIMEOUT)


def _chat_completion(
    provider: str,
    api_url: str,
    api_key: str,
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 600,
    repetition_penalty: float = 1.1,
) -> str:
    """通用聊天补全接口（兼容 OpenAI 格式）"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
    }

    # 不同 provider 的 base_url 补全
    if provider == "ollama" and not api_url.endswith("/v1"):
        api_url = api_url.rstrip("/") + "/v1"

    try:
        with _build_client(api_url, api_key) as client:
            resp = client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        return "⚠️ 无法连接到 LLM 服务，请检查是否已启动（Ollama 需运行 `ollama serve`）"
    except httpx.TimeoutException:
        return "⏳ LLM 响应超时，请稍后重试"
    except Exception as e:
        return f"⚠️ LLM 调用出错：{str(e)}"


# ═══════════════════════════════════════
# Feature 1: AI 实验解读
# ═══════════════════════════════════════

def interpret_experiment(cfg: dict, params: dict, results: dict) -> str:
    """
    根据当前实验参数和模型结果，调用 LLM 生成中文解读。

    params:  {blur, occlude, overlap, poison, sample_num}
    results: {模型名: {accuracy, is_misjudged}}
    """
    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "ollama")
    api_url  = llm_cfg.get("api_url", "http://localhost:11434")
    api_key  = _resolve_api_key(llm_cfg.get("api_key", ""))
    model    = llm_cfg.get("model", "qwen2.5:7b")

    scene   = cfg.get("scene_name", "未知场景")
    pos     = cfg.get("positive_label", "正品")
    neg     = cfg.get("negative_label", "次品")

    blur    = params.get("blur", 0)
    occlude = params.get("occlude", 0)
    overlap = params.get("overlap", 0.0)
    poison  = params.get("poison", 0.0)

    # 构建结果摘要
    detail_lines = []
    correct_count = 0
    for name, r in results.items():
        ok = not r["is_misjudged"]
        if ok:
            correct_count += 1
        label = pos if ok else neg
        detail_lines.append(
            f"- {name}：准确率 {r['accuracy']:.1%}，判定为「{label}」{'✅' if ok else '❌'}"
        )
    total = len(results)
    detail_str = "\n".join(detail_lines)

    system_prompt = f"""你是一位AI教学助手，用一段连贯的话（150字以内）解读实验结果。
要求：通俗易懂，像老师在课堂上口头讲解一样自然，不要分点、不要编号。"""

    user_prompt = f"""当前实验：{scene}
参数：模糊{blur}级，遮挡{occlude}%，重叠度{overlap:.0%}，投毒率{poison:.0%}

模型结果：
{detail_str}
共识：{correct_count}/{total} 正确

请用一段连贯的话解读：现在发生了什么现象？为什么？说明什么原理？"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    return _chat_completion(provider, api_url, api_key, model, messages, max_tokens=300)


# ═══════════════════════════════════════
# Feature 2: AI 智能问答助教
# ═══════════════════════════════════════

def chat_with_ai(cfg: dict, params: dict, user_question: str) -> str:
    """
    学生提问，LLM 结合当前实验状态回答。
    """
    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "ollama")
    api_url  = llm_cfg.get("api_url", "http://localhost:11434")
    api_key  = _resolve_api_key(llm_cfg.get("api_key", ""))
    model    = llm_cfg.get("model", "qwen2.5:7b")

    scene   = cfg.get("scene_name", "未知场景")
    pos     = cfg.get("positive_label", "正品")
    neg     = cfg.get("negative_label", "次品")
    feat_x  = cfg.get("feature_x", "特征A")
    feat_y  = cfg.get("feature_y", "特征B")

    blur    = params.get("blur", 0)
    occlude = params.get("occlude", 0)
    overlap = params.get("overlap", 0.0)
    poison  = params.get("poison", 0.0)

    system_prompt = f"""你是一位AI教学助手，正在帮助中学生理解AI分类算法的原理。

当前教学场景：{scene}
- X轴特征（{feat_x}）
- Y轴特征（{feat_y}）
- 正品标签：{pos} / 次品标签：{neg}

当前实验参数：模糊={blur}，遮挡={occlude}%，重叠度={overlap:.0%}，投毒率={poison:.0%}

你正在辅助一个交互式AI可视化探究工具，该工具可以：
1. 同时展示 KNN、决策树、SVM、逻辑回归 四种算法的决策边界
2. 调节滑块观察特征漂移和边界扭曲
3. 查看模型共识度和判定分歧

请你用通俗易懂的语言回答学生的问题，多打比方、少讲公式，
多鼓励学生动手调节参数验证自己的想法。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_question},
    ]

    return _chat_completion(provider, api_url, api_key, model, messages, max_tokens=800)


def check_connection(cfg: dict) -> str:
    """检查 LLM 配置是否完整（不实际调用 API）"""
    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "ollama")
    api_url  = llm_cfg.get("api_url", "")
    api_key  = _resolve_api_key(llm_cfg.get("api_key", ""))
    model    = llm_cfg.get("model", "qwen2.5:7b")

    if not api_key:
        return "💡 未配置 API Key，AI 功能不可用（不影响核心可视化）"
    if not api_url:
        return "💡 未配置 API URL，AI 功能不可用"
    return f"🤖 LLM 已配置（{provider}/{model}）"
