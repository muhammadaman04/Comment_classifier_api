import streamlit as st
import requests
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path to import data_storage
sys.path.append(str(Path(__file__).parent.parent))
from data_storage import load_data, save_data

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://10.108.72.196:8000/predict"
ADMIN_PASSWORD = "admin123"  # Hardcoded password

# Load data from shared storage
if "data_loaded" not in st.session_state:
    st.session_state.posts, st.session_state.comments = load_data()
    st.session_state.data_loaded = True
else:
    # Reload data on each run to get latest from other sessions
    st.session_state.posts, st.session_state.comments = load_data()
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

st.set_page_config(page_title="Admin Panel", page_icon="🔐", layout="wide")

# -----------------------------
# PASSWORD PROTECTION
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
# ADMIN PANEL
# -----------------------------
st.title("🔐 Admin Panel")
st.write("Manage posts and view comment classifications")

# Logout button
if st.button("🚪 Logout"):
    st.session_state.admin_authenticated = False
    st.rerun()

st.write("---")

# -----------------------------
# ADD NEW POST
# -----------------------------
st.subheader("➕ Add New Post")

# Show success message if post was just added
if "post_success" in st.session_state and st.session_state["post_success"]:
    st.success("✅ Post added successfully!")
    st.session_state["post_success"] = False  # Clear after showing

post_text = st.text_area("Post Content:", height=100, key="new_post", value="")

if st.button("Add Post"):
    if post_text.strip():
        new_post = {
            "id": len(st.session_state.posts) + 1,
            "text": post_text.strip(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.posts.append(new_post)
        # Save to shared storage
        save_data(st.session_state.posts, st.session_state.comments)
        # Set success flag and clear input field
        st.session_state["post_success"] = True
        if "new_post" in st.session_state:
            del st.session_state["new_post"]
        st.rerun()
    else:
        st.warning("⚠️ Please enter post content")

# -----------------------------
# VIEW POSTS AND COMMENTS
# -----------------------------
st.subheader("📋 All Posts")

if not st.session_state.posts:
    st.info("No posts yet. Add a post above!")
else:
    for post in st.session_state.posts:
        with st.expander(f"📌 Post #{post['id']} - {post['timestamp']}", expanded=False):
            st.write(post['text'])
            
            # Get comments for this post
            post_comments = [c for c in st.session_state.comments if c['post_id'] == post['id']]
            
            st.write("---")
            st.write(f"**Comments ({len(post_comments)}):**")
            
            if not post_comments:
                st.info("No comments yet for this post.")
            else:
                for idx, comment in enumerate(post_comments, 1):
                    st.write(f"**Comment #{idx}** - {comment['timestamp']}")
                    st.write(f"*{comment['text']}*")
                    
                    # Display classification results
                    classification = comment.get('classification', {})
                    
                    if classification:
                        st.write("**Classification Report:**")
                        
                        # Overall prediction
                        if classification.get('is_toxic'):
                            st.error(f"⚠️ **Toxic Comment**")
                        else:
                            st.success(f"✅ **Non-Toxic Comment**")
                        
                        st.write(f"**Explanation:** {classification.get('explanation', 'N/A')}")
                        
                        st.write("**Label-wise Predictions:**")
                        predictions = classification.get('predictions', [])
                        for pred in predictions:
                            label = pred['label'].replace("_", " ").title()
                            prob = pred['probability']
                            is_pos = pred['is_positive']
                            
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
st.warning("⚠️ This will delete all posts and their associated comments. This action cannot be undone!")

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🗑️ Clear All Posts", type="secondary"):
        st.session_state.posts = []
        st.session_state.comments = []
        # Save empty data to shared storage
        save_data(st.session_state.posts, st.session_state.comments)
        st.success("✅ All posts and comments have been cleared!")
        st.rerun()

st.write("---")