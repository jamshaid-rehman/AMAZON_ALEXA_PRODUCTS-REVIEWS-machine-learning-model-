import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Amazon Alexa Sentiment Analysis",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF9900;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #232F3E;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9900;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('amazon_alexa.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['review_length'] = df['verified_reviews'].astype(str).apply(len)
    df['word_count'] = df['verified_reviews'].astype(str).apply(lambda x: len(x.split()))
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

@st.cache_resource
def load_model():
    model = joblib.load('alexa_sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Sidebar navigation
st.sidebar.title("🎤 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 EDA Dashboard", "🤖 Sentiment Prediction", "📈 Model Performance"]
)

# Load data
try:
    df = load_data()
    model, vectorizer = load_model()
except:
    st.error("⚠️ Please ensure the dataset and model files are in the same directory!")
    st.stop()

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🎤 Amazon Alexa Reviews Sentiment Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📋 Project Overview
    This project performs comprehensive **Exploratory Data Analysis (EDA)** and **Sentiment Classification** 
    on Amazon Alexa product reviews using Machine Learning.
    
    **Dataset:** Amazon Alexa Reviews  
    **Records:** 3,150 customer reviews  
    **Features:** Rating, Date, Variation, Verified Reviews, Feedback
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Reviews", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        positive_pct = (df['feedback'].sum() / len(df)) * 100
        st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Product Variations", df['variation'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Rating", f"{df['rating'].mean():.2f} ⭐")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Project Objectives")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Completed Tasks:**
        - 15+ Exploratory Data Analyses
        - Data Preprocessing & Cleaning
        - TF-IDF Vectorization
        - Random Forest Classification Model
        - Interactive Streamlit Dashboard
        """)
    
    with col2:
        st.markdown("""
        **🔍 Key Features:**
        - Real-time Sentiment Prediction
        - Interactive Visualizations
        - Comprehensive Statistical Analysis
        - Model Performance Metrics
        - Word Cloud Analysis
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================================
# EDA DASHBOARD
# ============================================================================
elif page == "📊 EDA Dashboard":
    st.markdown('<p class="main-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Basic Statistics", 
        "🎯 Target Analysis", 
        "📦 Product Variations",
        "📅 Temporal Analysis",
        "💬 Text Analysis"
    ])
    
    # TAB 1: Basic Statistics
    with tab1:
        st.markdown("## 1️⃣ Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Numerical Features")
            st.dataframe(df[['rating', 'review_length', 'word_count']].describe())
        
        with col2:
            st.markdown("### Missing Values Analysis")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing': df.isnull().sum(),
                'Percentage': (df.isnull().sum() / len(df)) * 100
            })
            st.dataframe(missing_df)
        
        st.markdown("### 2️⃣ Rating Distribution")
        fig = px.histogram(df, x='rating', color='feedback', 
                          title='Rating Distribution by Feedback',
                          labels={'rating': 'Rating', 'count': 'Count'},
                          color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 3️⃣ Correlation Heatmap")
        corr_data = df[['rating', 'feedback', 'review_length', 'word_count']].corr()
        fig = px.imshow(corr_data, text_auto=True, color_continuous_scale='RdBu_r',
                       title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Target Analysis
    with tab2:
        st.markdown("## 4️⃣ Feedback (Target) Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feedback_counts = df['feedback'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative'],
                values=feedback_counts.values,
                hole=0.4,
                marker_colors=['#4ECDC4', '#FF6B6B']
            )])
            fig.update_layout(title='Feedback Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Feedback Statistics")
            st.write(f"**Positive Reviews:** {feedback_counts[1]} ({(feedback_counts[1]/len(df)*100):.1f}%)")
            st.write(f"**Negative Reviews:** {feedback_counts[0]} ({(feedback_counts[0]/len(df)*100):.1f}%)")
            st.write(f"**Total Reviews:** {len(df)}")
        
        st.markdown("### 5️⃣ Rating Statistics by Feedback")
        rating_stats = df.groupby('feedback')['rating'].agg(['mean', 'median', 'std', 'count'])
        rating_stats.index = ['Negative', 'Positive']
        st.dataframe(rating_stats.style.highlight_max(axis=0))
        
        st.markdown("### 6️⃣ Box Plot: Rating by Feedback")
        fig = px.box(df, x='feedback', y='rating', color='feedback',
                    title='Rating Distribution by Feedback Type',
                    labels={'feedback': 'Feedback (0=Negative, 1=Positive)', 'rating': 'Rating'},
                    color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'})
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Product Variations
    with tab3:
        st.markdown("## 7️⃣ Product Variation Analysis")
        
        variation_counts = df['variation'].value_counts()
        fig = px.bar(x=variation_counts.index, y=variation_counts.values,
                    title='Reviews by Product Variation',
                    labels={'x': 'Product Variation', 'y': 'Number of Reviews'},
                    color=variation_counts.values,
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 8️⃣ Feedback Distribution by Product Variation")
        variation_feedback = pd.crosstab(df['variation'], df['feedback'], normalize='index') * 100
        fig = px.bar(variation_feedback, barmode='group',
                    title='Feedback Distribution (%) by Product Variation',
                    labels={'value': 'Percentage', 'variation': 'Product Variation'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 9️⃣ Average Rating by Variation")
        avg_rating = df.groupby('variation')['rating'].mean().sort_values(ascending=False)
        fig = px.bar(x=avg_rating.index, y=avg_rating.values,
                    title='Average Rating by Product Variation',
                    labels={'x': 'Product Variation', 'y': 'Average Rating'},
                    color=avg_rating.values,
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Temporal Analysis
    with tab4:
        st.markdown("## 🔟 Time Series Analysis")
        
        df_clean = df.dropna(subset=['date'])
        
        st.markdown("### Reviews Over Time (Monthly)")
        monthly_reviews = df_clean.groupby(df_clean['date'].dt.to_period('M')).size()
        monthly_reviews.index = monthly_reviews.index.to_timestamp()
        
        fig = px.line(x=monthly_reviews.index, y=monthly_reviews.values,
                     title='Number of Reviews Over Time',
                     labels={'x': 'Date', 'y': 'Number of Reviews'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 1️⃣1️⃣ Reviews by Year")
        yearly = df_clean['year'].value_counts().sort_index()
        fig = px.bar(x=yearly.index, y=yearly.values,
                    title='Reviews Distribution by Year',
                    labels={'x': 'Year', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 1️⃣2️⃣ Average Rating Trend Over Time")
        monthly_rating = df_clean.groupby(df_clean['date'].dt.to_period('M'))['rating'].mean()
        monthly_rating.index = monthly_rating.index.to_timestamp()
        
        fig = px.line(x=monthly_rating.index, y=monthly_rating.values,
                     title='Average Rating Trend Over Time',
                     labels={'x': 'Date', 'y': 'Average Rating'})
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: Text Analysis
    with tab5:
        st.markdown("## 1️⃣3️⃣ Review Length Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='review_length', color='feedback',
                             title='Review Length Distribution',
                             labels={'review_length': 'Character Count'},
                             color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='word_count', color='feedback',
                             title='Word Count Distribution',
                             labels={'word_count': 'Word Count'},
                             color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 1️⃣4️⃣ Review Statistics by Feedback")
        review_stats = df.groupby('feedback')[['review_length', 'word_count']].mean()
        review_stats.index = ['Negative', 'Positive']
        st.dataframe(review_stats.style.highlight_max(axis=0))
        
        st.markdown("### 1️⃣5️⃣ Word Cloud Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Reviews**")
            positive_text = ' '.join(df[df['feedback']==1]['verified_reviews'].astype(str))
            wordcloud = WordCloud(width=400, height=300, background_color='white',
                                colormap='Greens').generate(positive_text)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Negative Reviews**")
            negative_text = ' '.join(df[df['feedback']==0]['verified_reviews'].astype(str))
            wordcloud = WordCloud(width=400, height=300, background_color='white',
                                colormap='Reds').generate(negative_text)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

# ============================================================================
# SENTIMENT PREDICTION
# ============================================================================
elif page == "🤖 Sentiment Prediction":
    st.markdown('<p class="main-header">🤖 Real-Time Sentiment Prediction</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### ✍️ Enter a Review
    Type or paste an Amazon Alexa product review below to predict its sentiment.
    """)
    
    user_input = st.text_area(
        "Your Review:",
        placeholder="Example: I love this Alexa! It works great and the sound quality is amazing.",
        height=150
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button("🔮 Predict Sentiment", use_container_width=True)
    
    if predict_button and user_input:
        with st.spinner("Analyzing sentiment..."):
            # Vectorize the input
            input_vectorized = vectorizer.transform([user_input])
            
            # Make prediction
            prediction = model.predict(input_vectorized)[0]
            probability = model.predict_proba(input_vectorized)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
                st.metric("Sentiment", sentiment)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                confidence = max(probability) * 100
                st.metric("Confidence", f"{confidence:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                word_count = len(user_input.split())
                st.metric("Word Count", word_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Probability distribution
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Positive'],
                'Probability': probability
            })
            
            fig = px.bar(prob_df, x='Sentiment', y='Probability',
                        color='Sentiment',
                        color_discrete_map={'Negative': '#FF6B6B', 'Positive': '#4ECDC4'},
                        title='Prediction Probabilities')
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    
    elif predict_button:
        st.warning("⚠️ Please enter a review to analyze.")
    
    # Example reviews
    st.markdown("---")
    st.markdown("### 💡 Try These Example Reviews")
    
    examples = {
        "Positive Example 1": "This Alexa device is absolutely amazing! The sound quality is superb and it responds quickly to all my commands.",
        "Positive Example 2": "Best purchase ever! Love how it integrates with my smart home devices.",
        "Negative Example 1": "Very disappointed. The device keeps disconnecting from WiFi and the sound quality is poor.",
        "Negative Example 2": "Doesn't understand my commands properly. Very frustrating experience."
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Positive Example 1"):
            st.code(examples["Positive Example 1"])
        if st.button("📝 Positive Example 2"):
            st.code(examples["Positive Example 2"])
    
    with col2:
        if st.button("📝 Negative Example 1"):
            st.code(examples["Negative Example 1"])
        if st.button("📝 Negative Example 2"):
            st.code(examples["Negative Example 2"])

# ============================================================================
# MODEL PERFORMANCE
# ============================================================================
elif page == "📈 Model Performance":
    st.markdown('<p class="main-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    # Load processed data to show metrics
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.metrics import roc_curve, auc
        
        X = df['verified_reviews'].astype(str)
        y = df['feedback']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        y_pred_proba = model.predict_proba(X_test_tfidf)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        st.markdown("## 📊 Overall Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{accuracy*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Training Samples", len(X_train))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Testing Samples", len(X_test))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Type", "Random Forest")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("## 🎯 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, text_auto=True,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Negative', 'Positive'],
                       y=['Negative', 'Positive'],
                       color_continuous_scale='Blues',
                       title='Confusion Matrix')
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.markdown("## 📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
        
        # Feature Importance
        st.markdown("## 🔝 Top Important Words")
        feature_importance = pd.DataFrame({
            'Word': vectorizer.get_feature_names_out(),
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(20)
        
        fig = px.bar(feature_importance, x='Importance', y='Word',
                    orientation='h',
                    title='Top 20 Most Important Words for Prediction',
                    color='Importance',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading model performance: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📚 IDS F24 Project | Created with Streamlit & Machine Learning</p>
    <p>Dataset: Amazon Alexa Reviews | Model: Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)