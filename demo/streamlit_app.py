"""
OmniNPC Streamlit Demo — 与 NPC 角色实时对话的可视化界面。

Phase 2 升级:
- 多角色选择（凌霜 / 柳轻吟）
- 情绪打分可视化
- 知识图谱事实展示
- 遗忘曲线状态指示

运行方式:
    streamlit run demo/streamlit_app.py
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python Path 中
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.engine import get_engine

# ── 页面配置 ──
st.set_page_config(
    page_title="OmniNPC Demo",
    page_icon="🎭",
    layout="wide",
)

# ── 自定义样式 ──
st.markdown("""
<style>
.stChatMessage [data-testid="stMarkdownContainer"] {
    font-size: 1.05rem;
}
.inner-monologue {
    background: #1e1e2e;
    border-left: 3px solid #f5c2e7;
    padding: 10px 15px;
    border-radius: 5px;
    font-style: italic;
    color: #cdd6f4;
    margin: 5px 0;
}
.emotion-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 2px 4px;
}
.emotion-high { background: #f38ba8; color: #1e1e2e; }
.emotion-mid { background: #fab387; color: #1e1e2e; }
.emotion-low { background: #a6e3a1; color: #1e1e2e; }
</style>
""", unsafe_allow_html=True)

# 角色映射
CHARACTER_MAP = {
    "tsundere_sister": "🗡️ 凌霜（傲娇师姐）",
    "gentle_healer": "🌿 柳轻吟（温柔药师）",
}


def get_emotion_badge(score: float, emotion_type: str) -> str:
    """生成情绪徽章 HTML。"""
    if score >= 7:
        css_class = "emotion-high"
    elif score >= 4:
        css_class = "emotion-mid"
    else:
        css_class = "emotion-low"
    return f'<span class="emotion-badge {css_class}">{emotion_type} ({score:.1f})</span>'


def main():
    st.title("🎭 OmniNPC — 认知架构 Agent 引擎 Demo")
    st.caption("Phase 2: 遗忘曲线 · 情绪打分 · 实体抽取 · 多 NPC")

    # ── 侧边栏配置 ──
    with st.sidebar:
        st.header("⚙️ 设置")

        character_id = st.selectbox(
            "选择 NPC 角色",
            list(CHARACTER_MAP.keys()),
            format_func=lambda x: CHARACTER_MAP.get(x, x),
        )

        session_id = st.text_input("会话 ID", value="demo_session")

        show_monologue = st.checkbox("🧠 显示内心独白", value=True)
        show_tool_calls = st.checkbox("🔧 显示工具调用", value=True)
        show_emotion = st.checkbox("💫 显示情绪打分", value=True)

        st.divider()

        if st.button("🗑️ 清空对话记录", key="btn_clear"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.header("⏳ 离线引擎测试")
        if st.button("⏰ 手动触发一次全局 Tick", key="btn_tick"):
            try:
                engine = get_engine()
                res = engine.force_tick()
                spread_count = len(res.get("rumor_spreads", []))
                st.success(f"Tick #{res.get('tick_id')} 执行完成！流言传播 {spread_count} 次")
            except Exception as e:
                st.error(f"Tick 失败: {e}")

        with st.expander("📢 制造流言"):
            rumor_text = st.text_input("流言内容", placeholder="玩家打碎了师傅的花瓶", key="input_rumor")
            if st.button("发布流言", key="btn_rumor"):
                if rumor_text:
                    engine = get_engine()
                    engine.rumor_spreader.create_rumor(
                        content=rumor_text,
                        source_npc=character_id,
                        original_event="玩家交互事件",
                    )
                    st.success("流言已制造！下一次 Tick 时将开始传播。")

        try:
            engine = get_engine()
            active_rumors = engine.rumor_spreader.active_rumors
            if active_rumors:
                st.caption(f"活跃流言数: {len(active_rumors)}")
                for r in active_rumors:
                    st.code(
                        f"内容：{r.content}\n"
                        f"来源：{r.source_npc}\n"
                        f"可信度：{r.credibility:.2f} | 传播：{r.spread_count} 次"
                    )
        except Exception:
            pass

        # 显示游戏状态
        st.header("📊 游戏状态")
        try:
            engine = get_engine()
            state = engine.get_game_state(session_id)
            st.json({
                "玩家位置": state.player.location,
                "持有道具": state.player.inventory,
                "关系": {
                    k: {"好感度": v.affection, "信任度": v.trust}
                    for k, v in state.relationships.items()
                },
                "世界事件": state.world_flags,
            })
        except Exception:
            st.info("等待首次对话后显示状态")

        # 显示知识图谱
        st.header("🕸️ 知识图谱")
        try:
            engine = get_engine()
            if character_id in engine._memory_managers:
                mm = engine._memory_managers[character_id]
                facts = mm.semantic.to_facts(character_id, depth=2)
                if facts:
                    for fact in facts[:10]:
                        st.text(f"• {fact}")
                else:
                    st.info("暂无图谱关系")
        except Exception:
            st.info("等待对话生成图谱")

    # ── 对话历史 ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    avatar_map = {
        "tsundere_sister": "🗡️",
        "gentle_healer": "🌿",
    }
    npc_avatar = avatar_map.get(character_id, "🤖")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar=npc_avatar):
                st.write(msg["content"])
                if show_monologue and msg.get("monologue"):
                    st.markdown(
                        f'<div class="inner-monologue">🧠 内心独白: {msg["monologue"]}</div>',
                        unsafe_allow_html=True,
                    )
                if show_emotion and msg.get("emotion"):
                    em = msg["emotion"]
                    badge = get_emotion_badge(em.get("emotion_score", 5), em.get("emotion_type", "平淡"))
                    imp_badge = f'<span class="emotion-badge emotion-mid">重要性 {em.get("importance", 5):.1f}</span>'
                    st.markdown(f"💫 {badge} {imp_badge}", unsafe_allow_html=True)
                if show_tool_calls and msg.get("tool_calls"):
                    with st.expander("🔧 工具调用详情"):
                        for tc in msg["tool_calls"]:
                            st.code(f"{tc['tool_name']}({tc['arguments']})\n→ {tc['result']}")

    # ── 用户输入 ──
    placeholder = {
        "tsundere_sister": "对凌霜说点什么...",
        "gentle_healer": "对柳轻吟说点什么...",
    }.get(character_id, "说点什么...")

    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.write(prompt)

        with st.chat_message("assistant", avatar=npc_avatar):
            with st.spinner("NPC 正在思考..."):
                try:
                    engine = get_engine()
                    response = engine.process_chat(
                        player_input=prompt,
                        character_id=character_id,
                        session_id=session_id,
                    )

                    # 显示台词
                    st.write(response.dialogue)

                    # 显示内心独白
                    if show_monologue and response.inner_monologue:
                        st.markdown(
                            f'<div class="inner-monologue">🧠 内心独白: {response.inner_monologue}</div>',
                            unsafe_allow_html=True,
                        )

                    # 显示情绪打分
                    emotion_data = None
                    if show_emotion and response.metadata and "emotion" in response.metadata:
                        emotion_data = response.metadata["emotion"]
                        badge = get_emotion_badge(
                            emotion_data.get("emotion_score", 5),
                            emotion_data.get("emotion_type", "平淡"),
                        )
                        imp_badge = f'<span class="emotion-badge emotion-mid">重要性 {emotion_data.get("importance", 5):.1f}</span>'
                        st.markdown(f"💫 {badge} {imp_badge}", unsafe_allow_html=True)

                    # 显示工具调用
                    if show_tool_calls and response.tool_calls:
                        with st.expander("🔧 工具调用详情"):
                            for tc in response.tool_calls:
                                st.code(
                                    f"{tc.tool_name}({tc.arguments})\n→ {tc.result}"
                                )

                    # 保存到对话历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.dialogue,
                        "monologue": response.inner_monologue,
                        "emotion": emotion_data,
                        "tool_calls": [tc.model_dump() for tc in response.tool_calls],
                    })

                except Exception as e:
                    st.error(f"❌ 引擎出错: {e}")
                    st.exception(e)

        st.rerun()


if __name__ == "__main__":
    main()
