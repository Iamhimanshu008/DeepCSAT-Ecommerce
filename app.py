import streamlit as st
import pandas as pd
import joblib
from utils import plot_csat_distribution, plot_avg_resolution_vs_csat, plot_channel_vs_csat

# ----------- Page Setup -----------
st.set_page_config(page_title="DeepCSAT – Ecommerce", layout="wide")
st.title("🛍️ DeepCSAT – E-commerce CSAT Predictor")
st.markdown("<style>h1{font-size: 36px;}</style>", unsafe_allow_html=True)

# ----------- Load Model Safely -----------
try:
    model = joblib.load("model/model.pkl")
    preprocessor = joblib.load("model/preprocessor.pkl")
except FileNotFoundError:
    st.error("❌ Model files not found. Please run `train_model.py` first.")
    st.stop()

# ----------- Sidebar Upload + Branding -----------
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    st.markdown("---")
    st.markdown("🧠 Built by **Himanshu Shekhar**")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/shekharr-himanshu/)")

# ----------- Save Uploaded Data -----------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

# ----------- Tab Layout -----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Home", "📊 EDA", "🔮 Predict", "💬 Feedback"])

# ----------- 🏠 Home Tab -----------
with tab1:
    st.markdown("## 👋 Welcome to DeepCSAT")
    st.markdown("""<div style='font-size:18px;'>
    This ML-powered app predicts <b>Customer Satisfaction (CSAT)</b> using support data from e-commerce platforms.
    <br><br>
    <b>🚀 What You Can Do:</b>
    <ul>
        <li>Upload your support ticket CSV</li>
        <li>Analyze customer experience patterns</li>
        <li>Predict future customer satisfaction</li>
    </ul>
    </div>""", unsafe_allow_html=True)

# ----------- 📊 EDA Tab -----------
with tab2:
    st.header("📊 Exploratory Data Analysis")

    if "df" in st.session_state:
        df = st.session_state["df"]
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        # Add option to filter data based on column
        channel_filter = st.selectbox("Filter by Channel", options=[
                                      "All"] + list(df['channel_name'].unique()))
        if channel_filter != "All":
            df = df[df['channel_name'] == channel_filter]

        # Show visual charts
        st.subheader("📈 CSAT Score Distribution")
        fig = plot_csat_distribution(df)
        if fig:
            st.pyplot(fig)

        st.subheader("⏱️ Avg Resolution Time vs CSAT")
        fig = plot_avg_resolution_vs_csat(df)
        if fig:
            st.pyplot(fig)
        else:
            st.warning(
                "Missing columns: 'resolution_time' or 'connected_handling_time'")

        st.subheader("📡 CSAT by Support Channel")
        fig = plot_channel_vs_csat(df)
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Missing column: 'channel_name'")

        # Summary Stats
        st.subheader("📊 Summary Stats")
        st.markdown(f"- Total Records: `{len(df)}`")
        if 'CSAT Score' in df.columns:
            avg = df['CSAT Score'].mean()
            st.markdown(f"- Average CSAT Score: `{avg:.2f}`")
            st.markdown(
                f"- % Satisfied (CSAT ≥ 4): `{(df['CSAT Score'] >= 4).mean() * 100:.1f}%`")
    else:
        st.info("⬅️ Please upload a CSV file from the sidebar.")

# ----------- 🔮 Prediction Tab -----------
with tab3:
    st.header("🔮 Predict Customer Satisfaction")

    if "df" in st.session_state:
        df = st.session_state["df"]
        st.subheader("🧾 Input Preview")
        st.dataframe(df.head())

        # Live Prediction Form
        st.subheader("📝 Live Prediction Form")
        channel_name = st.selectbox(
            "Select Channel", df['channel_name'].unique())
        category = st.text_input("Category")
        sub_category = st.text_input("Sub-category")
        agent_name = st.text_input("Agent Name")
        supervisor = st.text_input("Supervisor")
        manager = st.text_input("Manager")
        tenure_bucket = st.selectbox(
            "Tenure Bucket", df['Tenure Bucket'].unique())
        agent_shift = st.selectbox("Agent Shift", df['Agent Shift'].unique())

        live_input_data = {
            'channel_name': [channel_name],
            'category': [category],
            'Sub-category': [sub_category],
            'Agent_name': [agent_name],
            'Supervisor': [supervisor],
            'Manager': [manager],
            'Tenure Bucket': [tenure_bucket],
            'Agent Shift': [agent_shift]
        }

        live_input_df = pd.DataFrame(live_input_data)

        if st.button("🚀 Run Live Prediction"):
            try:
                live_input_processed = preprocessor.transform(live_input_df)
                live_prediction = model.predict(live_input_processed)

                st.success("✅ Live Prediction Complete!")
                st.write(
                    f"Predicted CSAT: {'🟢 Satisfied' if live_prediction[0] == 1 else '🔴 Not Satisfied'}")
            except Exception as e:
                st.error(f"❌ Error during live prediction: {e}")

        # Run Prediction on Uploaded Data
        if st.button("🚀 Run Prediction on Uploaded Data"):
            try:
                X = df[['channel_name', 'category', 'Sub-category', 'Agent_name',
                        'Supervisor', 'Manager', 'Tenure Bucket', 'Agent Shift']]
                X_processed = preprocessor.transform(X)
                predictions = model.predict(X_processed)

                df['Predicted CSAT'] = predictions
                df['Predicted Label'] = df['Predicted CSAT'].map(
                    {1: '🟢 Satisfied', 0: '🔴 Not Satisfied'})

                st.success("✅ Prediction Complete!")
                st.subheader("🔝 Preview of Predictions (Top 5)")
                st.dataframe(df[['Predicted CSAT', 'Predicted Label']].head())

                st.subheader("📥 Download Full Results")
                st.download_button("📤 Download CSV", df.to_csv(
                    index=False), file_name="csat_predictions.csv")

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
    else:
        st.info("⬅️ Upload your data first to predict.")

# ----------- 💬 Feedback Tab -----------
with tab4:
    st.markdown("## 📝 We Value Your Feedback")

    # Section for name (mandatory)
    st.subheader("👤 Your Name")
    name = st.text_input("Please enter your name:")

    # Section for email (mandatory)
    st.subheader("📧 Your Email")
    email = st.text_input("Please enter your email:")

    # Section for rating the app
    st.subheader("🌟 Rate Your Experience")
    rating = st.slider("Rate the app (1 = Worst, 5 = Best)", 1, 5, 3)

    # Section for additional feedback and suggestions
    st.subheader("💬 Suggestions for Improvement")
    suggestions = st.text_area(
        "Please provide any suggestions or feedback for improving this app:")

    # Section for submitting feedback
    if st.button("🚀 Submit Feedback"):
        if name and email and suggestions:
            # Save feedback to a file
            feedback_data = f"Name: {name}\nEmail: {email}\nRating: {rating} ⭐\nSuggestions: {suggestions}\n\n"

            # Write feedback to a text file using UTF-8 encoding
            with open("feedback.txt", "a", encoding="utf-8") as f:
                f.write(feedback_data)

            st.success("✅ Thank you for your feedback!")
            st.markdown(f"**Name:** {name}")
            st.markdown(f"**Email:** {email}")
            st.markdown(f"**Your Rating:** {rating} ⭐")
            st.markdown(f"**Suggestions:** {suggestions}")
        else:
            st.warning(
                "❌ Please provide your name, email, and suggestions before submitting.")
