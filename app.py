import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import yake

# ========== CONFIG ==========
st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")

# ---------- HEADER ----------
st.title("🎬 YouTube Sentiment Analyzer")
st.markdown("Analyze comments from any YouTube video — sentiment, keywords, word clouds & audience insights.")

# ---------- INPUT ----------
video_url = st.text_input("Enter YouTube Video URL:", "")
num_comments = st.slider("Number of comments to analyze", 50, 500, 150)
analyze_btn = st.button("🚀 Run Analysis")

if analyze_btn and video_url:
    st.info("Scraping comments... please wait ⏳")
    downloader = YoutubeCommentDownloader()
    comments = []

    try:
        for comment in downloader.get_comments_from_url(video_url, sort_by=0):
            comments.append(comment['text'])
            if len(comments) >= num_comments:
                break
    except Exception as e:
        st.error(f"Error scraping comments: {e}")
        st.stop()

    if not comments:
        st.warning("No comments found! Try a different video.")
        st.stop()

    df = pd.DataFrame(comments, columns=["Comment"])
    st.success(f"✅ {len(df)} comments scraped!")

    # ---------- SENTIMENT ANALYSIS ----------
    st.info("Running Sentiment Analysis (DistilBERT)...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",truncation=True)
    results = sentiment_pipeline(df["Comment"].tolist())

    df["Label"] = [r["label"] for r in results]
    df["Score"] = [r["score"] for r in results]
    df["Label"] = df["Label"].map({"POSITIVE": "Positive", "NEGATIVE": "Negative"})

    st.success("✅ Sentiment analysis complete!")

    # ---------- CHARTS ----------
    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Label", data=df, palette=["#16a34a", "#dc2626"], ax=ax)
    plt.title("YouTube Comment Sentiment Distribution")
    st.pyplot(fig)

    # ---------- WORD CLOUDS ----------
    st.subheader("☁️ Word Clouds")

    stopwords = set(STOPWORDS).union({"video", "like", "subscribe", "please", "watch", "channel", "https"})
    pos_text = " ".join(df[df["Label"]=="Positive"]["Comment"])
    neg_text = " ".join(df[df["Label"]=="Negative"]["Comment"])
    all_text = " ".join(df["Comment"])

    wc_all = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords).generate(all_text)
    wc_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens", stopwords=stopwords).generate(pos_text)
    wc_neg = WordCloud(width=800, height=400, background_color="white", colormap="Reds", stopwords=stopwords).generate(neg_text)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(wc_all.to_array(), caption="Overall Comments")
    with col2:
        st.image(wc_pos.to_array(), caption="Positive Comments")
    with col3:
        st.image(wc_neg.to_array(), caption="Negative Comments")

    # ---------- KEYWORDS ----------
    st.subheader("🔑 Top Keywords (YAKE)")
    kw_extractor = yake.KeywordExtractor(n=1, top=15)
    keywords = kw_extractor.extract_keywords(all_text)
    kw_df = pd.DataFrame(keywords, columns=["Keyword","Score"])
    kw_df["Score"] = kw_df["Score"].round(4)
    st.dataframe(kw_df)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(y="Keyword", x="Score", data=kw_df, palette="mako", ax=ax)
    ax.set_title("Top Keywords by YAKE")
    st.pyplot(fig)

    # ---------- COMMENT PREVIEW ----------
    st.subheader("💬 Sample Comments")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 😃 Positive Comments")
        for c in df[df["Label"]=="Positive"]["Comment"].head(5):
            st.write(f"✅ {c}")
    with col2:
        st.markdown("### 😞 Negative Comments")
        for c in df[df["Label"]=="Negative"]["Comment"].head(5):
            st.write(f"❌ {c}")

    # ---------- AUTO SUMMARY ----------
    pos_pct = round((df["Label"]=="Positive").mean()*100,2)
    neg_pct = 100 - pos_pct

    summary = (
        f"**Audience Mood Summary:**\n"
        f"- Positive comments: {pos_pct}%\n"
        f"- Negative comments: {neg_pct}%\n"
        f"- The overall audience sentiment is "
        f"{'favorable 👍' if pos_pct > 60 else 'mixed 😐' if 40 <= pos_pct <= 60 else 'critical 👎'}.\n\n"
        f"Frequent keywords indicate main discussion themes."
    )
    st.info(summary)

    # ---------- EXPORT ----------
    st.download_button("📥 Download Full Data (CSV)", data=df.to_csv(index=False), file_name="youtube_sentiment_results.csv", mime="text/csv")
