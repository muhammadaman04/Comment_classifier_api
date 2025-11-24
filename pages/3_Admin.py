import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_storage import load_data, save_data

from config import PREDICT_URL as API_URL

# -----------------------------
# CONFIG
# -----------------------------
ADMIN_PASSWORD = "admin123"

if "data_loaded" not in st.session_state:
    st.session_state.posts, st.session_state.comments = load_data()
    st.session_state.data_loaded = True
else:
    st.session_state.posts, st.session_state.comments = load_data()

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

st.set_page_config(page_title="Admin Panel", page_icon="🔐", layout="wide")

# -----------------------------
# AUTH
# -----------------------------
if not st.session_state.admin_authenticated:
    st.title("🔐 Admin Login")
    password_input = st.text_input("Enter Admin Password:", type="password")

    if st.button("Login"):
        if password_input == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password!")

    st.stop()

# -----------------------------
# DASHBOARD ACTIONS
# -----------------------------
st.title("🔐 Admin Panel")
st.write("Manage posts and view comment classifications")

if st.button("🚪 Logout"):
    st.session_state.admin_authenticated = False
    st.rerun()

st.write("---")

# -----------------------------
# ADD NEW POST
# -----------------------------
st.subheader("➕ Add New Post")

if "post_success" in st.session_state and st.session_state["post_success"]:
    st.success("✅ Post added successfully!")
    st.session_state["post_success"] = False

post_text = st.text_area("Post Content:", height=100, key="new_post", value="")

if st.button("Add Post"):
    if post_text.strip():
        new_post = {
            "id": len(st.session_state.posts) + 1,
            "text": post_text.strip(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state.posts.append(new_post)
        save_data(st.session_state.posts, st.session_state.comments)
        st.session_state["post_success"] = True
        if "new_post" in st.session_state:
            del st.session_state["new_post"]
        st.rerun()
    else:
        st.warning("⚠️ Please enter post content")

st.write("---")

# -----------------------------
# VIEW POSTS & COMMENTS
# -----------------------------
st.subheader("📋 All Posts")

if not st.session_state.posts:
    st.info("No posts yet. Add a post above!")
else:
    for post in st.session_state.posts:
        with st.expander(f"📌 Post #{post['id']} - {post['timestamp']}", expanded=False):
            st.write(post["text"])

            post_comments = [c for c in st.session_state.comments if c["post_id"] == post["id"]]

            st.write("---")
            st.write(f"**Comments ({len(post_comments)}):**")

            if not post_comments:
                st.info("No comments yet for this post.")
            else:
                for idx, comment in enumerate(post_comments, 1):
                    st.write(f"**Comment #{idx}** - {comment['timestamp']}")
                    st.write(f"*{comment['text']}*")

                    classification = comment.get("classification", {})

                    if classification:
                        st.write("**Classification Report:**")
                        if classification.get("is_toxic"):
                            st.error("⚠️ **Toxic Comment**")
                        else:
                            st.success("✅ **Non-Toxic Comment**")

                        st.write(f"**Explanation:** {classification.get('explanation', 'N/A')}")

                        st.write("**Label-wise Predictions:**")
                        for pred in classification.get("predictions", []):
                            label = pred["label"].replace("_", " ").title()
                            prob = pred["probability"]
                            is_pos = pred["is_positive"]
                            if is_pos:
                                st.markdown(f"🔴 **{label}** — {prob:.4f}")
                            else:
                                st.markdown(f"🟢 {label} — {prob:.4f}")
                    st.write("---")

st.write("---")

# -----------------------------
# CLEAR ALL POSTS
# -----------------------------
st.subheader("🗑️ Clear All Posts")
st.warning("⚠️ This permanently deletes every post and comment.")

if st.button("🗑️ Clear All Posts", type="secondary"):
    st.session_state.posts = []
    st.session_state.comments = []
    save_data(st.session_state.posts, st.session_state.comments)
    st.success("✅ All posts and comments have been cleared!")
    st.rerun()

