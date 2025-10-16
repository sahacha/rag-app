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

# Ensure each browser has a persistent user_id stored in localStorage and
# that the app URL contains ?user_id=<id>. This ties server-side session files
# to a browser-specific id so users cannot see other users' histories.
st.components.v1.html(
        """
        <script>
        (function(){
            try {
                // Prefer crypto.randomUUID when available
                const makeId = () => (crypto && crypto.randomUUID) ? crypto.randomUUID() : ('u' + Date.now() + Math.floor(Math.random()*1000000));
                let id = localStorage.getItem('thesis_user_id');
                if (!id) {
                    id = makeId();
                    localStorage.setItem('thesis_user_id', id);
                }

                const params = new URLSearchParams(window.location.search);
                if (params.get('user_id') !== id) {
                    params.set('user_id', id);
                    const newUrl = window.location.pathname + '?' + params.toString();
                    // replaceState avoids adding history entries
                    window.history.replaceState({}, '', newUrl);
                }
            } catch (e) {
                // no-op
            }
        })();
        </script>
        """,
        height=0,
)

# If a saved history exists in localStorage for this user, copy it into the
# query param `messages` (base64) so the Python side can read and load it on
# first render. This keeps the canonical history in the browser.
st.components.v1.html(
        """
        <script>
        (function(){
            try {
                const id = localStorage.getItem('thesis_user_id');
                if (!id) return;
                const key = 'thesis_history_' + id;
                const stored = localStorage.getItem(key);
                if (!stored) return;

                const params = new URLSearchParams(window.location.search);
                // Only set messages param when it's missing to avoid growing the URL on every render
                if (!params.get('messages')) {
                    params.set('messages', stored);
                    params.set('user_id', id);
                    const newUrl = window.location.pathname + '?' + params.toString();
                    window.history.replaceState({}, '', newUrl);
                }
            } catch (e) {
                // no-op
            }
        })();
        </script>
        """,
        height=0,
)

chat_rag = st.Page("chat_rag.py", title="Chiang Mai Q&A Travel & Planing")
pg = st.navigation(
    [chat_rag, chat_history]
)
pg.run()
