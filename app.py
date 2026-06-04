import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Page Configuration & Header Setup
# ----------------------------------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analysis Portal",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Twitter Sentiment Analysis EDA Dashboard")
st.markdown("### SAP ID: `70176505` | Student Assignment Project")
st.write("An all-in-one exploratory data analysis dashboard tracking text sentiments.")
st.markdown("---")

# ----------------------------------------------------------
# Data Loading Strategy
# ----------------------------------------------------------
@st.cache_data
def load_data():
    # Placeholder structure mirroring the expected Twitter dataset schema
    # (Text column & Sentiment Labels)
    mock_dataset = {
        'tweet_text': [
            "This software update is absolute perfection! Super clean.",
            "Customer support was incredibly slow and unhelpful. Frustrating.",
            "The morning train departs at exactly 8:15 AM tomorrow.",
            "Fantastic weekend hanging out with family and old friends!",
            "Feeling quite discouraged and down about the news today.",
            "The delivery package arrived safely on my porch.",
            "Highly recommend trying this restaurant, food is extraordinary!",
            "It is currently overcast and raining outside.",
            "Abysmal quality; the device stopped working after two hours.",
            "Just finished reading the first chapter of my new textbook."
        ],
        'sentiment_label': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                           'neutral', 'positive', 'neutral', 'negative', 'neutral']
    }
    df = pd.DataFrame(mock_dataset)
    
    # Feature Engineering (Generates lengths for calculations)
    df['char_count'] = df['tweet_text'].astype(str).apply(len)
    df['sentiment_label'] = df['sentiment_label'].str.strip().str.lower()
    return df

df = load_data()

# ----------------------------------------------------------
# FILTER PANEL (Built inline using the sidebar)
# ----------------------------------------------------------
st.sidebar.header("🎯 Dashboard Control Panel")
sentiment_options = df['sentiment_label'].unique()

selected_sentiments = st.sidebar.multiselect(
    "Filter by Sentiment Class:",
    options=sentiment_options,
    default=sentiment_options
)

# Apply runtime dataframe scoping instantly
filtered_df = df[df['sentiment_label'].isin(selected_sentiments)]

# ----------------------------------------------------------
# CHARTS & VISUALS SECTION (Rendered directly dynamically)
# ----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Class Percentage Distribution")
    if not filtered_df.empty:
        distribution_df = filtered_df['sentiment_label'].value_counts().reset_index()
        distribution_df.columns = ['Sentiment', 'Total Records']
        
        # Plotly chart definition
        fig_pie = px.pie(
            distribution_df, 
            values='Total Records', 
            names='Sentiment', 
            hole=0.4,
            color='Sentiment',
            color_discrete_map={'positive': '#2ECC71', 'negative': '#E74C3C', 'neutral': '#3498DB'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Select at least one category to plot distributions.")

with col2:
    st.subheader("📈 Length Verification Boxplot")
    if not filtered_df.empty:
        # Plotly box chart definition
        fig_box = px.box(
            filtered_df, 
            x='sentiment_label', 
            y='char_count', 
            color='sentiment_label',
            labels={'sentiment_label': 'Sentiment Class', 'char_count': 'Character Count'},
            color_discrete_map={'positive': '#2ECC71', 'negative': '#E74C3C', 'neutral': '#3498DB'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Filter view is empty.")

st.markdown("---")

# ----------------------------------------------------------
# WORDCLOUD & DATA VIEWER
# ----------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("☁️ Keyword Cloud Map")
    if not filtered_df.empty:
        text_dump = " ".join(filtered_df['tweet_text'].astype(str).tolist())
        if text_dump.strip():
            wordcloud = WordCloud(width=600, height=350, background_color='white').generate(text_dump)
            fig_wc, ax = plt.subplots(figsize=(6, 3.5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)

with col4:
    st.subheader("🔍 Tabular Dataset Previewer")
    st.dataframe(filtered_df[['sentiment_label', 'char_count', 'tweet_text']], use_container_width=True, height=280)