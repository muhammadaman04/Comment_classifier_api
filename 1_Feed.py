import streamlit as st
import requests
from datetime import datetime
from pathlib import Path
from data_storage import load_data, save_data

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://10.108.72.196:8000/predict"

# Load data from shared storage
if "data_loaded" not in st.session_state:
    st.session_state.posts, st.session_state.comments = load_data()
    st.session_state.data_loaded = True
else:
    # Reload data on each run to get latest from other sessions
    st.session_state.posts, st.session_state.comments = load_data()

st.set_page_config(page_title="Feed", page_icon="📰", layout="wide")

st.title("📰 Feed")
st.write("Browse posts and add comments")

st.write("---")

# -----------------------------
# DISPLAY POSTS
# -----------------------------
if not st.session_state.posts:
    st.info("📭 No posts available yet. Check back later!")
else:
    # Display posts in reverse order (newest first)
    for post in reversed(st.session_state.posts):
        st.subheader(f"📌 Post #{post['id']}")
        st.caption(f"Posted on: {post['timestamp']}")
        st.write(post['text'])
        
        # Get comments for this post
        post_comments = [c for c in st.session_state.comments if c['post_id'] == post['id']]
        
        # Comments section (expandable)
        with st.expander(f"💬 Comments ({len(post_comments)})", expanded=False):
            if not post_comments:
                st.info("No comments yet. Be the first to comment!")
            else:
                for idx, comment in enumerate(post_comments, 1):
                    st.write(f"**Comment #{idx}** - {comment['timestamp']}")
                    st.write(comment['text'])
                    st.write("---")
        
        # Add comment section
        with st.expander("✍️ Add Comment", expanded=False):
            # Show success message if comment was just added
            success_key = f"success_{post['id']}"
            if success_key in st.session_state and st.session_state[success_key]:
                st.success("✅ Comment added successfully!")
                st.session_state[success_key] = False  # Clear after showing
            
            comment_text = st.text_area(
                f"Your comment on Post #{post['id']}:",
                height=100,
                key=f"comment_{post['id']}",
                value=""  # Ensure it starts empty
            )
            
            if st.button(f"Submit Comment", key=f"submit_{post['id']}"):
                if comment_text.strip():
                    # Classify the comment silently
                    with st.spinner("Processing comment..."):
                        try:
                            payload = {"comment": comment_text.strip()}
                            response = requests.post(API_URL, json=payload)
                            
                            if response.status_code == 200:
                                classification_result = response.json()
                                
                                # Store comment with classification data
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
                                # Save to shared storage
                                save_data(st.session_state.posts, st.session_state.comments)
                                # Set success flag and clear input field
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
        
        st.write("---")
