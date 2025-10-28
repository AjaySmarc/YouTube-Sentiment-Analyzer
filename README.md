🎯 YouTube Sentiment Analyzer

A Streamlit-based web app that scrapes YouTube comments and performs real-time sentiment analysis using advanced Transformer models from Hugging Face.
Visualize audience opinions with word clouds, emotion breakdowns, and interactive charts to better understand viewer engagement.

🚀 Features

🔍 Fetch YouTube Comments using the youtube-comment-downloader library

🤖 Sentiment Analysis powered by DistilBERT from Hugging Face Transformers

☁️ Interactive Web App built with Streamlit

📊 Data Visualization using Matplotlib & Seaborn

☁️ Word Cloud Generation for most common words

⚙️ Keyword Extraction using YAKE for quick topic insights

🧠 Tech Stack
Category	Technology
Frontend	Streamlit
Backend	Python
NLP Model	DistilBERT (Hugging Face Transformers)
Visualization	Matplotlib, Seaborn, WordCloud
Data Processing	Pandas, NumPy
Comment Fetching	youtube-comment-downloader
⚙️ Installation & Setup
1️⃣ Clone this repository
git clone https://github.com/AjaySmarc/YouTube-Sentiment-Analyzer.git
cd YouTube-Sentiment-Analyzer

2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app.py


<img width="1913" height="806" alt="Screenshot 2025-10-28 121932" src="https://github.com/user-attachments/assets/3ddf2b21-8b39-4dfa-97cd-ba41ae93dcf3" />

<img width="1871" height="807" alt="Screenshot 2025-10-28 121949" src="https://github.com/user-attachments/assets/6e2785a0-90a4-4e5d-8076-078dcdf1992e" />

<img width="1863" height="817" alt="Screenshot 2025-10-28 122009" src="https://github.com/user-attachments/assets/dce90bc9-4fb2-4eb2-af01-06a2091f0098" />


🌐 Deployment

This project is deployed on Streamlit Cloud.
To deploy your own version:

Fork this repo

Connect it to Streamlit Cloud

Deploy directly from GitHub — Streamlit handles setup automatically


