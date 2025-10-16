from __future__ import annotations

import json
import sys
import time
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import List

import chromadb
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config

try:
    from flashrank import Ranker, RerankRequest

    FLASHRANK_AVAILABLE = True
except ImportError:
    Ranker = None  # type: ignore
    RerankRequest = None  # type: ignore
    FLASHRANK_AVAILABLE = False


def _ensure_session_id() -> None:
    # Prefer a browser-stored user id passed in via the URL query param `user_id`.
    # This ensures each browser gets its own session file and cannot see others'.
    query_params = st.experimental_get_query_params()
    user_id = None
    if isinstance(query_params, dict):
        vals = query_params.get('user_id')
        if vals:
            user_id = vals[0]

    if "session_id" not in st.session_state:
        if user_id:
            st.session_state.session_id = user_id
        else:
            st.session_state.session_id = str(uuid.uuid4())


def _get_session_file() -> Path:
    # Kept for backward compatibility but we no longer rely on server files when
    # browser-based storage is available. Return the path if needed.
    return Config.SESSIONS_DIR / f"session_{st.session_state.session_id}.json"


def _load_session_history() -> None:
    # Prefer loading messages passed in via the query param `messages` which is
    # expected to be base64-encoded JSON placed there by client-side JS reading
    # localStorage. This keeps all histories in the browser memory for each user.
    try:
        params = st.experimental_get_query_params()
        encoded = None
        if isinstance(params, dict):
            vals = params.get('messages')
            if vals:
                encoded = vals[0]

        if encoded:
            try:
                decoded = json.loads(base64.b64decode(encoded).decode('utf-8'))
            except Exception:
                decoded = None

            if isinstance(decoded, dict):
                messages = decoded.get('messages')
                created_at = decoded.get('created_at')
                updated_at = decoded.get('updated_at')
            else:
                messages = None
                created_at = None
                updated_at = None

            if isinstance(messages, list):
                st.session_state.messages = messages
            if created_at:
                st.session_state.session_created_at = created_at
            if updated_at:
                st.session_state.last_saved_at = updated_at
            return
    except Exception:
        # Fall back to server-side file if client-side data is not available
        pass

    # Fallback: read the server-side session file if present
    session_file = _get_session_file()
    if session_file.exists():
        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}

        messages = data.get("messages")
        if isinstance(messages, list):
            st.session_state.messages = messages

        created_at = data.get("created_at")
        if created_at:
            st.session_state.session_created_at = created_at

        updated_at = data.get("updated_at")
        if updated_at:
            st.session_state.last_saved_at = updated_at


def _save_session_history() -> None:
    # Save messages to browser localStorage by rendering a small JS snippet via
    # components.html. The snippet will store the payload under the key
    # `thesis_history_<user_id>`. We encode payload as base64 to avoid issues
    # with special characters when embedding in HTML.
    created_at = st.session_state.get("session_created_at")
    if not created_at:
        created_at = datetime.now().isoformat()
        st.session_state.session_created_at = created_at

    payload = {
        "session_id": st.session_state.session_id,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "message_count": len(st.session_state.messages),
        "messages": st.session_state.messages,
    }

    try:
        import base64 as _base64

        encoded = _base64.b64encode(json.dumps(payload, ensure_ascii=False).encode('utf-8')).decode('ascii')
        user_key = f"thesis_history_{st.session_state.session_id}"
        js = f"""
        <script>
        try {{
          const k = '{user_key}';
          const v = '{encoded}';
          localStorage.setItem(k, v);
        }} catch(e) {{}}
        </script>
        """

        st.components.v1.html(js, height=0)
        st.session_state.last_saved_at = payload["updated_at"]
    except Exception:
        # Fallback to server-side file if browser storage fails
        session_file = _get_session_file()
        session_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        st.session_state.last_saved_at = payload["updated_at"]


def _delete_session_history() -> None:
    # Remove from browser localStorage if possible, otherwise remove server file
    try:
        user_key = f"thesis_history_{st.session_state.session_id}"
        js = f"""
        <script>
        try {{
          localStorage.removeItem('{user_key}');
        }} catch(e) {{}}
        </script>
        """
        st.components.v1.html(js, height=0)
    except Exception:
        session_file = _get_session_file()
        if session_file.exists():
            session_file.unlink(missing_ok=True)


def _format_display_time(timestamp: str | None) -> str:
    if not timestamp:
        return "-"
    try:
        return datetime.fromisoformat(timestamp).strftime("%d %b %Y • %H:%M")
    except ValueError:
        return timestamp


st.title("ระบบแชทถามตอบการท่องเที่ยวเชียงใหม่และวางแผนการท่องเที่ยว")
st.caption("สามารถสอบถามข้อมูลการท่องเที่ยวเชียงใหม่ และวางแผนการเดินทางได้อย่างครบถ้วนได้เลยครับ")
class TravelAssistantRAG:
    """RAG pipeline that retrieves from Chroma, reranks, and answers with Groq."""

    def __init__(self):
        Config.validate_config()
        Config.setup_directories()

        self.client = chromadb.CloudClient(
            api_key=Config.CHROMADB_API_KEY,
            tenant=Config.CHROMADB_TENANT,
            database=Config.CHROMADB_DATABASE,
        )

        self.embedding_function = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )

        self.vector_store = Chroma(
            client=self.client,
            collection_name=Config.COLLECTIONS[0],
            embedding_function=self.embedding_function,
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": Config.INITIAL_RETRIEVAL_K}
        )

        self.reranker = self._initialize_reranker()

        # Configure Gemini API
        genai.configure(api_key=Config.GOOGLE_API_KEY)

        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_output_tokens=Config.LLM_MAX_TOKENS,
            google_api_key=Config.GOOGLE_API_KEY,
        )

        # Initialize Gemini model for search
        self.search_model = genai.GenerativeModel('gemini-pro')

        self.prompt = ChatPromptTemplate.from_template(
            """
           # System Prompt: คุณเป็นผู้ช่วยด้านการท่องเที่ยวเชียงใหม่ที่เชี่ยวชาญ ให้คำตอบอย่างเป็นมิตร
โดยใช้ข้อมูลจากเอกสารอ้างอิงและข้อมูลเพิ่มเติมจาก Google Search (ถ้ามี)

## บทบาทหลัก
คุณคือนักเขียนคู่มือท่องเที่ยวมืออาชีพที่เชี่ยวชาญในการเปรียบเทียบสถานที่ท่องเที่ยว โดยนำเสนอข้อมูลอย่างละเอียด เป็นระบบ และใช้งานได้จริง

## หลักการในการเขียน

### 1. โครงสร้างเนื้อหา
- **เริ่มต้นด้วยภาพรวม**: ระบุความแตกต่างหลักระหว่างสถานที่ทั้งสองอย่างชัดเจน
- **แบ่งหมวดหมู่อย่างเป็นระบบ**: ใช้หัวข้อที่เหมาะสมตามลักษณะสถานที่
- **จบด้วยสรุปและคำแนะนำ**: ให้ข้อมูลที่เป็นประโยชน์สำหรับการวางแผนท่องเที่ยว

### 2. การใช้ตาราง
จัดรูปแบบข้อมูลเป็นตารางเพื่อความชัดเจน:
- **ตารางเปรียบเทียบ**: แสดงข้อมูลเคียงข้างกันให้เห็นความแตกต่างชัดเจน
- **ตารางรายละเอียด**: จัดหมวดหมู่ข้อมูลภายในสถานที่เดียว
- **ตารางคำแนะนำ**: สรุปข้อมูลสำคัญสำหรับการวางแผน

### 3. เนื้อหาที่ต้องครอบคลุม

#### ก. ข้อมูลพื้นฐาน
- เวลาเปิด-ปิด (ระบุวันและเวลาอย่างละเอียด)
- ราคาค่าเข้า (ถ้ามี)
- การเดินทาง
- หมายเหตุพิเศษ (เช่น ช่วงฤดูกาล, วันหยุด)

#### ข. กิจกรรมและจุดเด่น
- แบ่งตามประเภทกิจกรรมอย่างชัดเจน
- ระบุรายละเอียดเฉพาะที่น่าสนใจ
- เน้นจุดเด่นที่แตกต่างกันของแต่ละสถานที่

#### ค. คำแนะนำเชิงปฏิบัติ
- เวลาที่เหมาะสมในการเยือน
- การแต่งกาย
- สิ่งที่ควรเตรียม
- เคล็ดลับพิเศษ

### 4. น้ำเสียงและภาษา

**ใช้ภาษาที่:**
- เป็นมิตรและเข้าถึงง่าย
- ให้ข้อมูลครบถ้วนแต่กระชับ
- เป็นกลางและเป็นวัตถุประสงค์
- มีความเคารพต่อวัฒนธรรมและศาสนา
- ปิดท้ายด้วยการเชิญชวนให้ถามเพิ่มเติม

**หลีกเลี่ยง:**
- การใช้ภาษาที่เกินจริงหรือโฆษณาเกินไป
- ข้อมูลที่คลุมเครือหรือไม่แน่ชัด
- การตัดสินคุณค่าส่วนบุคคลมากเกินไป

### 5. การใช้องค์ประกอบเสริม

**ใช้ Blockquote (>) สำหรับ:**
- ข้อสรุปสำคัญ
- ข้อแตกต่างหลัก
- คำแนะนำเด่น

**ใช้ตัวหนา (bold) สำหรับ:**
- ชื่อสถานที่
- หัวข้อในตาราง
- คำศัพท์สำคัญ

**ใช้ตัวเอียง (italic) สำหรับ:**
- คำที่ต้องการเน้น
- ภาษาท้องถิ่น

### 6. รูปแบบการเปรียบเทียบ

**แบบ Side-by-Side (เคียงข้าง):**
```
| หัวข้อ | สถานที่ A | สถานที่ B |
```

**แบบ Category-based (แยกหมวดหมู่):**
```
### สถานที่ A
| ประเภท | รายละเอียด |

### สถานที่ B
| ประเภท | รายละเอียด |
```

## เทมเพลตโครงสร้าง

```markdown
# [ชื่อสถานที่ A] กับ [ชื่อสถานที่ B]

[บทนำสั้น ๆ อธิบายความแตกต่างหลัก]

---

## 1. เวลาเปิด-ปิด

[ตารางเปรียบเทียบเวลา]

> **สรุป** – [ข้อสังเกตสำคัญ]

---

## 2. กิจกรรมที่น่าสนใจ

### [สถานที่ A]
[ตารางรายละเอียดกิจกรรม]

### [สถานที่ B]
[ตารางรายละเอียดกิจกรรม]

> **ข้อแตกต่างสำคัญ** – [อธิบายสั้น ๆ]

---

## 3. คำแนะนำสำหรับการวางแผนไปเยือน

[ตารางคำแนะนำเปรียบเทียบ]

---

### สรุป

- **[สถานที่ A]**: [สรุปสั้น]
- **[สถานที่ B]**: [สรุปสั้น]

[คำปิดท้ายเชิญชวน]
```

## หมายเหตุพิเศษ

- ตรวจสอบความถูกต้องของข้อมูลเวลาและราคา
- คำนึงถึงความหลากหลายทางวัฒนธรรม
- เคารพในสถานที่ศักดิ์สิทธิ์
- ให้ข้อมูลที่เป็นปัจจุบันและใช้งานได้จริง
- ปรับเนื้อหาให้เหมาะกับกลุ่มเป้าหมาย (ครอบครัว, คู่รัก, นักเดินทางเดี่ยว)

---

## ตัวอย่างการใช้งาน

**Input:** "เปรียบเทียบ วัดพระธาตุดอยสุเทพ กับ ตลาดวโรรส"

**Output:** สร้างเนื้อหาเปรียบเทียบที่:
1. ครอบคลุมเวลาเปิด-ปิด
2. อธิบายกิจกรรมและจุดเด่น
3. ให้คำแนะนำเชิงปฏิบัติ
4. จัดรูปแบบเป็นตารางและหมวดหมู่ที่ชัดเจน
5. มีสรุปและคำเชิญชวนปิดท้าย
"""
        )

        self.output_parser = StrOutputParser()

    def _initialize_reranker(self):
        if not FLASHRANK_AVAILABLE:
            return None

        try:
            return Ranker(model_name=Config.FLASHRANK_MODEL, cache_dir="/tmp")
        except Exception:
            return None

    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        if not self.reranker:
            return docs[: Config.RETRIEVAL_K]

        passages = [
            {"id": idx, "text": doc.page_content, "meta": doc.metadata or {}}
            for idx, doc in enumerate(docs)
        ]

        try:
            response = self.reranker.rerank(
                RerankRequest(query=query, passages=passages)
            )
            ranked = sorted(response, key=lambda item: item["score"], reverse=True)
            selected_docs: List[Document] = []
            for item in ranked:
                original_doc = docs[item["id"]]
                selected_docs.append(original_doc)
                if len(selected_docs) >= Config.RETRIEVAL_K:
                    break
            return selected_docs
        except Exception:
            return docs[: Config.RETRIEVAL_K]

    def _google_search_supplement(self, question: str) -> str:
        """Use Gemini to search and provide supplementary information"""
        try:
            search_prompt = f"""ค้นหาข้อมูลเกี่ยวกับคำถามนี้และสรุปข้อมูลที่เป็นประโยชน์:

คำถาม: {question}

โปรดให้ข้อมูลที่เป็นปัจจุบัน ถูกต้อง และเกี่ยวข้องกับการท่องเที่ยวเชียงใหม่ (ถ้ามี)
หากไม่พบข้อมูลเพิ่มเติมหรือไม่เกี่ยวข้อง ให้ตอบว่า "ไม่พบข้อมูลเพิ่มเติม"

ข้อมูลที่ค้นหาได้:"""

            response = self.search_model.generate_content(search_prompt)
            search_result = response.text.strip()

            if search_result and "ไม่พบข้อมูลเพิ่มเติม" not in search_result:
                return search_result
            return ""
        except Exception as e:
            return ""

    def _retrieve_documents(self, question: str) -> tuple[List[Document], List[dict], str]:
        raw_docs = self.retriever.get_relevant_documents(question)
        reranked_docs = self._rerank_documents(question, raw_docs)

        sources = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in reranked_docs
        ]

        # Get supplementary information from Google Search via Gemini
        search_info = self._google_search_supplement(question)

        return reranked_docs, sources, search_info

    def answer(
        self, question: str
    ) -> tuple[str, float, List[dict], float, float]:
        retrieval_start = time.time()
        reranked_docs, sources, search_info = self._retrieve_documents(question)
        retrieval_time = time.time() - retrieval_start

        # Combine retrieved documents context
        context_parts = [doc.page_content for doc in reranked_docs]
        context_text = "\n\n".join(context_parts) if context_parts else "ไม่พบข้อมูลอ้างอิง"

        # Add Google Search information if available
        if search_info:
            context_text += f"\n\n--- ข้อมูลเพิ่มเติมจาก Google Search ---\n{search_info}"

        generation_start = time.time()
        try:
            response = (
                self.prompt
                | self.llm
                | self.output_parser
            ).invoke({
                "context": context_text,
                "question": question,
            })
        except Exception:
            response = (
                "ขออภัย ระบบไม่สามารถสร้างคำตอบได้ในขณะนี้ "
                "กรุณาลองใหม่อีกครั้ง หรือระบุคำถามเพิ่มเติม"
            )

        generation_time = time.time() - generation_start
        total_time = retrieval_time + generation_time

        return response, total_time, sources, retrieval_time, generation_time

@st.cache_resource(show_spinner=True)
def load_rag_pipeline() -> TravelAssistantRAG:
    return TravelAssistantRAG()


def init_session_state() -> None:
    _ensure_session_id()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_created_at" not in st.session_state:
        st.session_state.session_created_at = None

    if "last_saved_at" not in st.session_state:
        st.session_state.last_saved_at = None

    if "history_loaded" not in st.session_state:
        _load_session_history()
        st.session_state.history_loaded = True

    if not st.session_state.session_created_at:
        st.session_state.session_created_at = datetime.now().isoformat()


init_session_state()

try:
    rag_pipeline = load_rag_pipeline()
except Exception as exc:
    st.error(
        "ไม่สามารถเริ่มต้นระบบ RAG ได้ กรุณาตรวจสอบการตั้งค่า หรือดูรายละเอียดในไฟล์ config"
    )
    st.exception(exc)
    st.stop()


with st.sidebar:
    if  st.button("ล้างประวัติการสนทนา"):
        st.session_state.messages = []
        st.session_state.session_created_at = None
        _delete_session_history()
        st.rerun()


for message in st.session_state.messages:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(message.get("content", ""))
        if role == "assistant":
            retrieval_time = message.get("retrieval_time")
            generation_time = message.get("generation_time")
            total_time = message.get("total_time") or message.get("generation_time")

            parts: List[str] = []
            if generation_time is not None:
                parts.append(f"🤖 สร้างคำตอบ: {generation_time:.2f}s")
            if total_time is not None and len(parts) >= 2:
                parts.append(f"⏱️ รวม: {total_time:.2f}s")
            elif total_time is not None and not parts:
                parts.append(f"⏱️ เวลาในการตอบ: {total_time:.2f}s")

            if parts:
                st.caption(" • ".join(parts))


if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับการท่องเที่ยวเชียงใหม่ได้เลย..."):
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state.messages.append(user_message)
    _save_session_history()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("กำลังสืบค้นข้อมูลและสร้างคำตอบ..."):
            (
                answer,
                total_time,
                sources,
                retrieval_time,
                generation_time,
            ) = rag_pipeline.answer(prompt)
        placeholder.markdown(answer)

        timing_parts = [
            f"🔎 ค้นหา: {retrieval_time:.2f} วินาที",
            f"🤖 สร้างคำตอบ: {generation_time:.2f} วินาที",
            f"⏱️ รวม: {total_time:.2f} วินาที",
        ]
        st.caption(" • ".join(timing_parts))
        if sources:
            with st.expander("ดูแหล่งข้อมูลอ้างอิง"):
                for idx, source in enumerate(sources, start=1):
                    st.markdown(f"**เอกสาร {idx}**")
                    metadata = source.get("metadata") or {}
                    for key, value in metadata.items():
                        st.markdown(f"- {key}: {value}")
                    st.markdown(f"> {source.get('content', '')}")

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "generation_time": generation_time,
        "retrieval_time": retrieval_time,
        "total_time": total_time,
        "sources": sources,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state.messages.append(assistant_message)
    _save_session_history()

    st.rerun()
