# =============================================================================
# 从"一言堂"到"百家争鸣"：多模型决策对比可视化探究沙盒
# =============================================================================
# 核心创意：同一份数据，交给四种不同的AI算法，
# 观察它们的决策边界如何不同、面对干扰时的抗性差异，
# 以及"共识vs分歧"背后的算法本质。
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
import io
import base64
import json
from PIL import Image, ImageDraw, ImageFilter
import os
import glob
import warnings
import httpx
warnings.filterwarnings('ignore')
from llm_helper import interpret_experiment, chat_with_ai, check_connection

# ============================
# 0. 配置加载（支持运行时切换场景）
# ============================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

if "cfg" not in st.session_state:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        st.session_state.cfg = json.load(f)
CFG = st.session_state.cfg

# LLM 配置
LLM_CFG = CFG.get("llm", {})

# ============================
# 0.5 中文字体设置
# ============================
def setup_chinese_font():
    bundled_font = os.path.join(os.path.dirname(__file__), "fonts", "SimHei.ttf")
    if os.path.exists(bundled_font):
        font_prop = fm.FontProperties(fname=bundled_font)
        plt.rcParams['font.family'] = font_prop.get_name()
        fm.fontManager.addfont(bundled_font)
        plt.rcParams['axes.unicode_minus'] = False
        return

    system = platform.system()
    font_path = None
    if system == 'Windows':
        font_path = 'C:/Windows/Fonts/simhei.ttf'
        if not os.path.exists(font_path):
            font_path = 'C:/Windows/Fonts/msyh.ttc'
    elif system == 'Darwin':
        font_path = '/System/Library/Fonts/PingFang.ttc'
    else:
        font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    try:
        if font_path and os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            plt.rcParams['font.sans-serif'] = [
                'SimHei', 'Microsoft YaHei', 'PingFang SC',
                'WenQuanYi Zen Hei', 'Arial Unicode MS'
            ]
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()


# ============================
# 1. 核心算法引擎（多模型版）
# ============================
def run_multi_model_simulation(
    overlap, poison_rate, sample_size, blur=0, occlude=0
):
    """
    在完全相同的训练数据上，训练 KNN / 决策树 / SVM / 逻辑回归 四种模型。
    返回各模型的结果 + 数据 + 测试点（供绘图用）。
    """
    # --- 1a. 生成基础数据（所有模型共用） ---
    center_distance = 4.0 * (1.0 - overlap)
    centers = np.array([
        [center_distance / 2,  center_distance / 2],
        [-center_distance / 2, -center_distance / 2],
    ])
    X, y = make_blobs(
        n_samples=sample_size, centers=centers,
        cluster_std=1.2, random_state=42
    )

    # --- 投毒 ---
    if poison_rate > 0:
        rng = np.random.RandomState(42)
        poison_count = int(len(y) * poison_rate)
        poison_indices = rng.choice(len(y), poison_count, replace=False)
        y[poison_indices] = 1 - y[poison_indices]

    # --- 1b. 测试点坐标（受左侧图像干扰影响） ---
    base_x = center_distance / 2 * 0.6
    base_y = center_distance / 2 * 0.6
    drift_x = -blur * 0.25        # 模糊 → 向左（危险区）漂移
    drift_y = -occlude * 0.1      # 遮挡 → 向下（危险区）漂移
    test_point = np.array([[base_x + drift_x, base_y + drift_y]])
    real_label = 0  # 真实标签为"正品"

    # --- 1c. 定义模型池 ---
    model_pool = {
        "KNN (K近邻)": KNeighborsClassifier(n_neighbors=5),
        "决策树":        DecisionTreeClassifier(max_depth=3, random_state=42),
        "SVM (支持向量机)": SVC(kernel='rbf', gamma='scale', random_state=42),
        "逻辑回归":      LogisticRegression(random_state=42, max_iter=500),
    }

    # --- 1d. 统一坐标网格（便于四张图对比） ---
    all_pts = np.vstack([X, test_point])
    x_min = all_pts[:, 0].min() - 1.5
    x_max = all_pts[:, 0].max() + 1.5
    y_min = all_pts[:, 1].min() - 1.5
    y_max = all_pts[:, 1].max() + 1.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.15),
        np.arange(y_min, y_max, 0.15),
    )

    # --- 1e. 训练所有模型，收集结果 ---
    results = {}
    for name, model in model_pool.items():
        model.fit(X, y)

        accuracy  = model.score(X, y)
        predicted = model.predict(test_point)[0]
        is_misjudged = (real_label != predicted)

        # 尝试获取预测概率（有些模型不支持）
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(test_point)[0]
            except Exception:
                proba = None

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "predicted": predicted,
            "is_misjudged": is_misjudged,
            "proba": proba,
        }

    return results, X, y, test_point, real_label, xx, yy


# ============================
# 2. 绘图函数
# ============================
def _plot_one_model(ax, model, X, y, test_point, xx, yy,
                    model_name, accuracy, is_misjudged):
    """在给定的 ax 上绘制单个模型的决策边界"""
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.30, cmap='RdBu_r')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr',
               edgecolors='black', s=22, alpha=0.75)

    marker = 'X' if is_misjudged else '*'
    color  = 'red' if is_misjudged else 'lime'
    size   = 260 if is_misjudged else 180
    ax.scatter(test_point[:, 0], test_point[:, 1],
               marker=marker, s=size, c=color,
               edgecolors='black', linewidths=2, zorder=5)

    # 标题：模型名 + 准确率
    ax.set_title(f"{model_name}  ·  准确率 {accuracy:.1%}",
                 fontsize=11, fontweight='bold', pad=6)

    ax.set_xlabel(f"特征 A ({CFG['feature_x']})", fontsize=8)
    ax.set_ylabel(f"特征 B ({CFG['feature_y']})", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.35)

    # 判定结果角标
    label = CFG["negative_label"] if is_misjudged else CFG["positive_label"]
    verdict = rf"$\bigotimes$ 误判为 {label}" if is_misjudged \
              else rf"$\checkmark$ 判定为 {label}"
    ax.text(0.03, 0.97, verdict,
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            color='red' if is_misjudged else 'green',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='lightgray', alpha=0.85))


def plot_comparison_grid(results, X, y, test_point, xx, yy):
    """生成 2×2 四模型对比总图"""
    names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    fig.patch.set_alpha(0.0)
    axes_flat = axes.flatten()

    for i, name in enumerate(names):
        r = results[name]
        _plot_one_model(axes_flat[i], r["model"], X, y, test_point, xx, yy,
                        name, r["accuracy"], r["is_misjudged"])

    # 多余的子图隐藏（本例刚好4个，不会触发）
    for j in range(len(names), 4):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    return fig


# ============================
# 3. 页面配置 & 资源加载
# ============================
st.set_page_config(
    page_title=CFG.get("page_title", f"多模型对比探究 - {CFG['scene_name']}"),
    layout="wide",
    page_icon="🧪",
)

_BASE_DIR  = os.path.dirname(__file__)

def _get_img_path():
    """获取当前场景的样本图片路径，不存在则 fallback"""
    path = os.path.join(_BASE_DIR, CFG.get("sample_img", "assets/lichi.png"))
    if os.path.exists(path):
        return path
    # fallback 到默认荔枝图
    default = os.path.join(_BASE_DIR, "assets/lichi.png")
    return default if os.path.exists(default) else path

def _get_seal_path():
    """获取当前场景的红章图片路径"""
    path = os.path.join(_BASE_DIR, CFG.get("seal_img", "assets/seal.png"))
    return path if os.path.exists(path) else os.path.join(_BASE_DIR, "assets/seal.png")

IMG_PATH  = _get_img_path()
SEAL_PATH = _get_seal_path()


@st.cache_resource
def _get_seal_base64(path):
    seal = Image.open(path).convert('RGBA')
    buf = io.BytesIO()
    seal.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

_SEAL_CACHE = {}

def _get_seal_uri():
    """获取当前场景的红章 base64 URI（带缓存）"""
    p = _get_seal_path()
    if p not in _SEAL_CACHE:
        _SEAL_CACHE[p] = _get_seal_base64(p)
    return _SEAL_CACHE[p]

SEAL_URI = _get_seal_uri()


# ============================
# 3.5 CSS 样式（印章动画复用）
# ============================
st.markdown("""
<style>
.seal-container {
    position: relative; display: inline-block; width: 100%;
    aspect-ratio: 4 / 3.0; border-radius: 12px;
}
.seal-container .base-image {
    width: 100%; height: 100%; object-fit:contain;
    display: block; border-radius: 12px;
}
.seal-container .seal-overlay {
    position: absolute; top: 50%; left: 50%; width: 55%; height: auto;
    pointer-events: none;
    animation: sealDrop 0.8s cubic-bezier(0.175,0.885,0.32,1.4) 2.0s forwards;
    opacity: 0;
}
@keyframes sealDrop {
    0%   { transform: translate(-50%,-200%) scale(0.3) rotate(-25deg); opacity: 0; }
    45%  { transform: translate(-50%,-48%)  scale(1.25) rotate(8deg);  opacity: 0.85; }
    65%  { transform: translate(-50%,-52%)  scale(0.92) rotate(-4deg); opacity: 0.72; }
    85%  { transform: translate(-50%,-49%)  scale(1.05) rotate(2deg);  opacity: 0.68; }
    100% { transform: translate(-50%,-50%)  scale(1)    rotate(0deg);  opacity: 0.62; }
}
.seal-container.breathing {
    animation: breathe 1.8s ease-in-out 3.0s infinite; border-radius: 12px;
}
@keyframes breathe {
    0%,100% { box-shadow: 0 0 6px 2px rgba(255,20,20,0.15),
               inset 0 0 6px 1px rgba(255,20,20,0.05); }
    50%     { box-shadow: 0 0 28px 10px rgba(255,20,20,0.55),
               inset 0 0 18px 5px rgba(255,20,20,0.2); }
}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ============================
# 3.6 Session State
# ============================
_EXP_COLS = [
    "序号", "图像模糊", "局部遮挡", "重叠度", "投毒率",
    "KNN判定", "决策树判定", "SVM判定", "逻辑回归判定",
    "共识度",
]
if 'exp_records' not in st.session_state:
    st.session_state.exp_records = pd.DataFrame(columns=_EXP_COLS)

if 'prev_params' not in st.session_state:
    st.session_state.prev_params = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'llm_interpretation' not in st.session_state:
    st.session_state.llm_interpretation = ""


# ============================
# 4. 页面标题
# ============================
st.title(CFG.get("title", f"🧪 多模型决策对比探究 - {CFG['scene_name']}"))
st.caption(CFG.get("caption", "同一份数据，四种算法，四个决策边界 —— 看看谁在「看走眼」，谁在「画错线」，谁最靠谱"))


# ============================
# 5. 侧边栏控制台
# ============================
with st.sidebar:
    st.header("🎛️ 实验环境控制台")
    st.divider()

    st.info(
        "🧠 **四模型同台**\n\n"
        "以下参数影响 **所有模型共用的训练数据**，"
        "公平对比，谁强谁弱一目了然。"
    )

    st.subheader("① 非结构化感知破坏器")
    blur_level = st.slider(
        "图像模糊干扰", 0, 15, 0, key="blur",
        help="模拟拍摄失焦、雨雾等"
    )
    occlude_level = st.slider(
        "局部遮挡干扰", 0, 80, 0, key="occlude",
        help="模拟被遮挡的面积比例(%)"
    )

    st.subheader("② 结构化数据破坏器")
    overlap = st.slider(
        "特征重叠度 (数据混淆)", 0.0, 0.9, 0.0, step=0.1,
        help="拉高 → 好次品长太像，AI 分不清"
    )
    poison = st.slider(
        "标注投毒率 (脏数据)", 0.0, 0.3, 0.0, step=0.05,
        help="拉高 → 人工贴错标签，AI 学坏"
    )

    with st.expander("高级参数 (一般不改)"):
        sample_num = st.slider("样本数量", 50, 300, 100, step=10)

        st.divider()
        st.markdown("**🔄 一键切换场景**")

        # 扫描所有 config-sample-*.json 模板
        scene_files = sorted(glob.glob(
            os.path.join(os.path.dirname(__file__), "config-sample-*.json")
        ))
        scene_options = {}
        scene_names = []
        for fp in scene_files:
            try:
                with open(fp, "r", encoding="utf-8") as _f:
                    sc = json.load(_f)
                name = sc.get("title", os.path.basename(fp))
                scene_options[name] = sc
                scene_names.append(name)
            except Exception:
                continue

        if scene_names:
            current_name = CFG.get("title", "")
            idx = scene_names.index(current_name) if current_name in scene_names else 0
            selected = st.selectbox("选择教学场景", scene_names, index=idx)
            if st.button("🔄 应用场景", use_container_width=True, type="primary"):
                new_cfg = dict(scene_options[selected])
                # 保留当前的 LLM 配置（api_key、provider 等不丢失）
                llm_snapshot = CFG.get("llm", {})
                if llm_snapshot:
                    new_cfg["llm"] = llm_snapshot
                st.session_state.cfg = new_cfg
                st.rerun()

    st.divider()
    # ── Feature 2: AI 智能问答助教（常驻显示） ──
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;"
        "background:linear-gradient(135deg,#667eea,#764ba2);padding:8px 12px;"
        "border-radius:10px'>"
        "<span style='font-size:20px'>🤖</span>"
        "<span style='font-size:15px;font-weight:bold;color:white;margin-left:4px'>"
        "AI 问答助教</span>"
        "<span style='background:rgba(255,255,255,0.25);color:white;font-size:10px;"
        "padding:1px 8px;border-radius:8px;margin-left:auto;font-weight:bold'>LIVE</span>"
        "</div>",
        unsafe_allow_html=True
    )
    st.caption("对当前实验有疑问？直接问AI👇")

    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                "<p style='color:#aaa;font-size:13px;text-align:center;"
                "padding:8px 0'>💡 试试问：为什么SVM的边界是弯的？</p>",
                unsafe_allow_html=True
            )
        for q, a in st.session_state.chat_history:
            # HTML 转义，防止 markdown 符号导致样式错乱
            q_esc = q.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            a_esc = a.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            # 把换行转成 <br>
            a_esc = a_esc.replace("\n", "<br>")
            st.markdown(
                f"<div style='background:#e8f4fd;padding:8px 12px;"
                f"border-radius:10px;margin:4px 0;font-size:13px'>"
                f"<b>🧑‍🎓 你：</b>{q_esc}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='background:#f0f0f0;padding:8px 12px;"
                f"border-radius:10px;margin:4px 0;font-size:13px'>"
                f"<b>🤖 助教：</b>{a_esc}</div>",
                unsafe_allow_html=True
            )

    user_q = st.text_input(
        "", key="chat_input",
        placeholder="输入你的问题，按回车发送……"
    )
    col_send, col_clear = st.columns([3, 1])
    with col_send:
        if st.button("💬 发送", use_container_width=True) and user_q:
            if not CFG.get("llm", {}).get("provider"):
                st.warning("⚠️ 请在 config.json 中配置 LLM 参数")
            else:
                with st.spinner("AI 助教思考中……"):
                    params = {"blur": blur_level, "occlude": occlude_level,
                              "overlap": overlap, "poison": poison}
                    answer = chat_with_ai(CFG, params, user_q)
                st.session_state.chat_history.append((user_q, answer))
                st.rerun()
    with col_clear:
        if st.session_state.chat_history:
            if st.button("🗑️", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

# ============================
# 6. 主界面：双栏联动
# ============================
run_params = (overlap, poison, sample_num, blur_level, occlude_level)
results, X, y, test_point, real_label, xx, yy = run_multi_model_simulation(
    overlap, poison, sample_num, blur_level, occlude_level
)

# --- 统计共识 ---
judgements = {n: r["is_misjudged"] for n, r in results.items()}
consensus_correct = sum(1 for v in judgements.values() if not v)
total_models = len(judgements)
any_failed = any(judgements.values())

col_left, col_right = st.columns(2, gap="large")

# ====== 左栏：图像 + 印章 + 共识仪表盘 ======
with col_left:
    st.subheader("📷 现象探秘：输入样本（所有模型看到的是同一张图）")

    # --- 图像处理 ---
    img = Image.open(_get_img_path())
    display_img = img.copy()
    if blur_level > 0:
        display_img = display_img.filter(
            ImageFilter.GaussianBlur(radius=blur_level)
        )
    if occlude_level > 0:
        draw = ImageDraw.Draw(display_img)
        w, h = display_img.size
        bw = int(w * occlude_level / 100)
        bh = int(h * occlude_level / 100)
        draw.rectangle([w - bw, h - bh, w, h], fill=(0, 0, 0))

    buf = io.BytesIO()
    display_img.convert('RGB').save(buf, format='JPEG', quality=80, optimize=True)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    img_uri = f"data:image/jpeg;base64,{img_b64}"

    # --- 盖章动画 ---
    if any_failed:
        html = (
            f'<div class="seal-container breathing">'
            f'<img src="{img_uri}" class="base-image">'
            f'<img src="{_get_seal_uri()}" class="seal-overlay">'
            f'</div>'
        )
    else:
        html = (
            f'<div class="seal-container">'
            f'<img src="{img_uri}" class="base-image">'
            f'</div>'
        )

    st.info("👁️ 调节左侧「图像模糊/遮挡」滑块 → 右侧四张图的 ⭐ 同步漂移！")
    st.markdown(html, unsafe_allow_html=True)

    # --- 共识仪表盘 ---
    st.divider()
    st.subheader("🎯 模型共识仪表盘")

    # 每模型一个"指示灯"
    cons_cols = st.columns(total_models)
    short_names = {"KNN (K近邻)": "KNN", "决策树": "DT",
                   "SVM (支持向量机)": "SVM", "逻辑回归": "LR"}
    for i, (name, r) in enumerate(results.items()):
        short = short_names.get(name, name[:3])
        ok = not r["is_misjudged"]
        with cons_cols[i]:
            icon = "✅" if ok else "❌"
            clr  = "green" if ok else "red"
            st.markdown(
                f"<p style='text-align:center;font-size:28px;margin:0;'>{icon}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align:center;font-weight:bold;"
                f"color:{clr};margin:0;'>{short}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align:center;font-size:13px;"
                f"color:gray;margin:0;'>{r['accuracy']:.0%}</p>",
                unsafe_allow_html=True,
            )

    # 共识结论
    st.divider()
    if consensus_correct == total_models:
        st.success(
            f"✅ **全体一致**：{total_models}/{total_models} "
            f"模型都判定为 **{CFG['positive_label']}**"
        )
    elif consensus_correct == 0:
        st.error(
            f"💥 **全体误判**：{total_models}/{total_models} "
            f"模型全部误判为 **{CFG['negative_label']}**！"
        )
    else:
        分歧数 = total_models - consensus_correct
        st.warning(
            f"⚡ **出现分歧**：{consensus_correct}/{total_models} 模型正确 "
            f"vs {分歧数}/{total_models} 模型误判——"
            "同一份数据，不同算法得出不同结论！"
        )

# ====== 右栏：2×2 边界对比图 ======
with col_right:
    st.subheader("📐 机制解密：四种决策边界同屏对决")

    st.info(
        "🔧 调节侧边栏参数 → 观察各算法的决策边界如何扭曲、"
        "哪个模型最先「扛不住」"
    )

    # 生成 2×2 对比图
    fig = plot_comparison_grid(results, X, y, test_point, xx, yy)
    st.pyplot(fig, use_container_width=True)

    # --- 详细指标表 ---
    with st.expander("📊 各模型详细指标"):
        rows = []
        for name, r in results.items():
            label = (
                CFG["negative_label"] if r["is_misjudged"]
                else CFG["positive_label"]
            )
            rows.append({
                "模型": name,
                "准确率": f"{r['accuracy']:.1%}",
                "判定": "❌ 误判" if r["is_misjudged"] else "✅ 正确",
                "判为": label,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- 原理说明 ---
    with st.expander("💡 读懂这张图（核心原理）"):
        st.markdown(f"""
        **🔴🔵 四张图，同一条数据**

        四张图用的是 **完全相同的训练数据（红蓝圆点）**
        和 **完全相同的测试样本（⭐）**，区别只在于算法的"脑回路"不同。

        ---

        **🧠 四种算法的思维方式**

        | 算法 | 通俗理解 | 特点 |
        |------|---------|------|
        | **KNN** | "看看邻居怎么说" | 以距离投票，对局部密度敏感 |
        | **决策树** | "按规则问到底" | 逐层判断，边界呈阶梯状 |
        | **SVM** | "找最宽分界线" | 追求最大间隔，受边界点影响大 |
        | **逻辑回归** | "算概率定阈值" | 线性分界，简单但表达能力有限 |

        ---

        **💥 为什么同一个 ⭐，不同算法反应不同？**

        图像模糊/遮挡导致特征坐标漂移后：
        - 有些算法的**决策边界刚好卡在漂移路径上** → ⭐ 跨过边界 → **误判**
        - 有些算法的**决策边界离得更远** → ⭐ 还在安全区 → **正确**

        > **关键认知**：没有"最好"的算法，不同算法对不同类型的干扰
        > （模糊/遮挡/重叠/投毒）的抵抗力各不相同。
        > 实际应用中，**多模型投票**往往比单一模型更可靠。

        ---

        **📊 共识度告诉你什么？**
        - **4/4 一致** → 问题简单明确，或干扰太强全倒了
        - **3/1 或 2/2 分歧** → 问题处于临界区，需要人工复核
        - **0/4 全误判** → 干扰已超出所有模型的承受极限
        """, unsafe_allow_html=True)

    # ── Feature 1: AI 实验解读 ──
    st.divider()
    st.subheader("🤖 AI 实验解读")
    col_interpret_btn, col_status = st.columns([3, 2])
    with col_interpret_btn:
        if st.button("🎯 解读当前实验", use_container_width=True, type="primary"):
            if not CFG.get("llm", {}).get("provider"):
                st.warning("⚠️ 请在 config.json 中配置 LLM 参数（参考 llm 字段示例）")
            else:
                with st.spinner("AI 正在分析实验结果……"):
                    params = {"blur": blur_level, "occlude": occlude_level, "overlap": overlap, "poison": poison, "sample_num": sample_num}
                    interpretation = interpret_experiment(CFG, params, results)
                    st.session_state.llm_interpretation = interpretation
                    st.rerun()
    with col_status:
        llm_status = check_connection(CFG)
        st.markdown(f"<p style='font-size:13px;margin-top:6px'>{llm_status}</p>", unsafe_allow_html=True)
    
    if st.session_state.llm_interpretation:
        interp = st.session_state.llm_interpretation
        interp_esc = interp.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        interp_esc = interp_esc.replace("\n", "<br>")
        st.markdown(
            f"<div style='background:#f8f9ff;padding:18px 22px;border-radius:14px;"
            f"border:1px solid #e0e3ff;font-size:15px;line-height:1.8;color:#1a1a2e'>"
            f"💡 {interp_esc}</div>",
            unsafe_allow_html=True
        )
        if st.button("🗑️ 清除解读", use_container_width=False):
            st.session_state.llm_interpretation = ""
            st.rerun()

# ============================
# 7. 实验记录
# ============================
st.divider()
st.subheader("📊 探究实验台：多模型对比记录")

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("📸 记录当前实验数据", use_container_width=True, type="primary"):
        row = {
            "序号": len(st.session_state.exp_records) + 1,
            "图像模糊": f"Lv.{blur_level}",
            "局部遮挡": f"{occlude_level}%",
            "重叠度": f"{overlap:.0%}",
            "投毒率": f"{poison:.0%}",
        }
        for name, r in results.items():
            short = short_names.get(name, name[:3])
            row[f"{short}判定"] = "✅" if not r["is_misjudged"] else "❌"
        row["共识度"] = f"{consensus_correct}/{total_models}"
        new_df = pd.DataFrame([row])
        st.session_state.exp_records = pd.concat(
            [st.session_state.exp_records, new_df], ignore_index=True
        )
        st.toast("✅ 记录成功！", icon="✅")

with col_btn2:
    if st.button("🗑️ 清空所有记录", use_container_width=True):
        st.session_state.exp_records = pd.DataFrame(columns=_EXP_COLS)
        st.toast("记录已清空", icon="🗑️")

st.dataframe(
    st.session_state.exp_records,
    use_container_width=True,
    hide_index=True,
)

if not st.session_state.exp_records.empty:
    csv_data = st.session_state.exp_records.to_csv(
        index=False
    ).encode('utf-8-sig')
    st.download_button(
        label="📥 导出实验报告 (CSV格式，可用Excel打开)",
        data=csv_data,
        file_name='多模型决策对比探究报告.csv',
        mime='text/csv',
    )
