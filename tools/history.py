import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config

Config.setup_directories()


def _format_timestamp(timestamp: str | None) -> str:
    if not timestamp:
        return "-"
    try:
        return datetime.fromisoformat(timestamp).strftime("%d %b %Y • %H:%M:%S")
    except ValueError:
        return timestamp


def _load_saved_sessions() -> List[Dict[str, Any]]:
    if not Config.SESSIONS_DIR.exists():
        return []

    sessions: List[Dict[str, Any]] = []
    for path in sorted(
        Config.SESSIONS_DIR.glob("session_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}

        messages = data.get("messages") or []
        sessions.append(
            {
                "label": (
                    f"📁 {path.stem.replace('session_', '')[:8]} • "
                    f"{_format_timestamp(data.get('updated_at'))} • {len(messages)} ข้อความ"
                ),
                "messages": messages,
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "file_path": path,
            }
        )

    return sessions


def _build_session_options() -> Dict[str, Dict[str, Any]]:
    options: Dict[str, Dict[str, Any]] = {}

    current_messages = st.session_state.get("messages", [])
    if current_messages:
        options["💬 เซสชันปัจจุบัน"] = {
            "messages": current_messages,
            "created_at": st.session_state.get("session_created_at"),
            "updated_at": st.session_state.get("last_saved_at"),
            "file_path": None,
        }

    for session in _load_saved_sessions():
        label = session.pop("label")
        options[label] = session

    return options


def _render_messages(messages: List[Dict[str, Any]]) -> None:
    if not messages:
        st.info("ยังไม่มีข้อความในประวัตินี้")
        return

    for message in messages:
        role = message.get("role", "assistant")
        display_name = "คุณ" if role == "user" else "ผู้ช่วย"
        tone = "user" if role == "user" else "assistant"

        with st.chat_message(tone):
            st.markdown(f"**{display_name}:** {message.get('content', '')}")
            if role != "user":
                retrieval_time = message.get("retrieval_time")
                generation_time = message.get("generation_time")
                total_time = message.get("total_time") or message.get("generation_time")

                timing_parts: List[str] = []
                if retrieval_time is not None:
                    timing_parts.append(f"🔎 ค้นหา: {retrieval_time:.2f}s")
                if generation_time is not None:
                    timing_parts.append(f"🤖 สร้างคำตอบ: {generation_time:.2f}s")
                if total_time is not None and len(timing_parts) >= 2:
                    timing_parts.append(f"⏱️ รวม: {total_time:.2f}s")
                elif total_time is not None and not timing_parts:
                    timing_parts.append(f"⏱️ เวลาในการตอบ: {total_time:.2f}s")

                if timing_parts:
                    st.caption(" • ".join(timing_parts))

            if role != "user" and message.get("sources"):
                with st.expander("ดูแหล่งข้อมูลอ้างอิง"):
                    for idx, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**เอกสาร {idx}**")
                        metadata = source.get("metadata") or {}
                        for key, value in metadata.items():
                            st.markdown(f"- {key}: {value}")
                        snippet = source.get("content")
                        if snippet:
                            st.markdown(f"> {snippet}")

        timestamp_text = _format_timestamp(message.get("timestamp"))
        if timestamp_text != "-":
            st.caption(timestamp_text)


st.title("ประวัติการสนทนา")

session_options = _build_session_options()

if not session_options:
    st.info("ยังไม่มีประวัติการสนทนา ลองกลับไปถามผู้ช่วยก่อนนะครับ")
    st.stop()

labels = list(session_options.keys())
default_index = 0
selected_label = st.selectbox("เลือกประวัติการสนทนา", labels, index=default_index)
selected_session = session_options[selected_label]

st.markdown(
    f"**สร้างเมื่อ:** {_format_timestamp(selected_session.get('created_at'))}"
)
st.markdown(
    f"**อัปเดตล่าสุด:** {_format_timestamp(selected_session.get('updated_at'))}"
)
st.markdown(f"**จำนวนข้อความ:** {len(selected_session.get('messages', []))}")

st.markdown("---")
_render_messages(selected_session.get("messages", []))
st.markdown("---")

selected_file = selected_session.get("file_path")

json_data = json.dumps(selected_session.get("messages", []), ensure_ascii=False, indent=2)
download_name = "chat_history_current.json"
if selected_file:
    download_name = f"{selected_file.stem}.json"
st.download_button(
    label="ดาวน์โหลดประวัติเป็น JSON",
    file_name=download_name,
    mime="application/json",
    data=json_data,
)
if selected_file:
    if st.button("ลบประวัตินี้", type="secondary"):
        selected_file.unlink(missing_ok=True)
        st.rerun()

