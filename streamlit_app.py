# Main entry point - Feed page (default/home page)
import streamlit as st
import requests
from datetime import datetime

from data_storage import load_data, save_data
from config import PREDICT_URL as API_URL

# Load shared data on every run
if "data_loaded" not in st.session_state:
    st.session_state.posts, st.session_state.comments = load_data()
    st.session_state.data_loaded = True
else:
    st.session_state.posts, st.session_state.comments = load_data()

st.set_page_config(page_title="Feed", page_icon="📰", layout="wide")

# Minimal CSS for spacing and layout only (no colors)
st.markdown("""
<style>
    .post-content {
        font-size: 16px;
        line-height: 1.6;
        margin: 12px 0;
    }
    .post-timestamp {
        font-size: 14px;
        margin-bottom: 12px;
        opacity: 0.7;
    }
    .comment-section {
        border-top: 1px solid;
        border-color: inherit;
        margin-top: 16px;
        padding-top: 16px;
        opacity: 0.6;
    }
    .comment-item {
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid;
        border-color: inherit;
        opacity: 0.8;
    }
    .comment-text {
        font-size: 15px;
        margin: 4px 0;
    }
    .comment-time {
        font-size: 13px;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>📰 Feed</h1>", unsafe_allow_html=True)

# -----------------------------
# DISPLAY POSTS
# -----------------------------
if not st.session_state.posts:
    st.info("📭 No posts available yet. Check back later!")
else:
    # Create a centered column for feed (Twitter-like narrow feed)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        for post in reversed(st.session_state.posts):
            # Post Card Container
            with st.container():
                # Post Header
                st.markdown(f"""
                <div class="post-timestamp">
                    <strong>Post #{post['id']}</strong> · {post['timestamp']}
                </div>
                """, unsafe_allow_html=True)
                
                # Post Content
                st.markdown(f"""
                <div class="post-content">
                    {post['text']}
                </div>
                """, unsafe_allow_html=True)
                
                # Get comments for this post
                post_comments = [c for c in st.session_state.comments if c['post_id'] == post['id']]
                
                # Comments Section
                st.markdown(f"""
                <div class="comment-section">
                    <strong style="font-size: 14px;">💬 {len(post_comments)} Comments</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Display Comments
                if post_comments:
                    for comment in post_comments:
                        st.markdown(f"""
                        <div class="comment-item">
                            <div class="comment-text">{comment['text']}</div>
                            <div class="comment-time">{comment['timestamp']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="font-size: 14px; padding: 12px; text-align: center; opacity: 0.7;">
                        No comments yet. Be the first to comment!
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add Comment Section
                with st.expander("✍️ Add Comment", expanded=False):
                    success_key = f"success_{post['id']}"
                    if success_key in st.session_state and st.session_state[success_key]:
                        st.success("✅ Comment added successfully!")
                        st.session_state[success_key] = False
                    
                    comment_text = st.text_area(
                        "What's on your mind?",
                        height=100,
                        key=f"comment_{post['id']}",
                        value="",
                        label_visibility="collapsed"
                    )
                    
                    col_btn1, col_btn2 = st.columns([1, 4])
                    with col_btn1:
                        if st.button("Post", key=f"submit_{post['id']}", type="primary"):
                            if comment_text.strip():
                                with st.spinner("Processing comment..."):
                                    try:
                                        payload = {"comment": comment_text.strip()}
                                        response = requests.post(API_URL, json=payload)
                                        
                                        if response.status_code == 200:
                                            classification_result = response.json()
                                            
                                            new_comment = {
                                                "post_id": post['id'],
                                                "text": comment_text.strip(),
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                "classification": {
                                                    "is_toxic": classification_result.get("is_toxic"),
                                                    "explanation": classification_result.get("explanation"),
                                                    "predictions": classification_result.get("predictions", [])
                                                }
                                            }
                                            st.session_state.comments.append(new_comment)
                                            save_data(st.session_state.posts, st.session_state.comments)
                                            
                                            st.session_state[success_key] = True
                                            if f"comment_{post['id']}" in st.session_state:
                                                del st.session_state[f"comment_{post['id']}"]
                                            st.rerun()
                                        else:
                                            st.error("Failed to process comment. Please try again.")
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            else:
                                st.warning("⚠️ Please enter a comment")
                
                # Spacing between posts
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("<br>", unsafe_allow_html=True)
