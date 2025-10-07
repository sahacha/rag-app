import streamlit as st
st.set_page_config("Chiang Mai Q&A Travel & Planing", "https://e-cms.rmutl.ac.th/assets/upload/images/2017/06/post_thumbnail_2017060611262952915.jpg")
st.logo("https://e-cms.rmutl.ac.th/assets/upload/images/2017/06/post_thumbnail_2017060611262952915.jpg")
import warnings
warnings.filterwarnings("ignore")
chat_history = []


st.html(
    f"""
    <style>
    body {{http://localhost:8501/
        -webkit-font-smoothing: antialiased;
    }}
    </style>
    """
)

chat_rag = st.Page("chat_rag.py", title="Chiang Mai Q&A Travel & Planing")
chat_history = st.Page("tools/history.py", title="Chat History")
pg = st.navigation(
    [chat_rag, chat_history]
)
pg.run()
