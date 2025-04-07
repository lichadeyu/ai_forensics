import streamlit as st
import pandas as pd
from datetime import datetime
import re
import plotly.express as px
import ollama

st.set_page_config(page_title="Mental Health & Intent Analysis", layout="wide")

st.title("📊 AI-Driven Mental Health & Intent Investigator")
st.markdown("Analyze user behavior from 5 data sources to determine mental state and potential criminal intent.")

# Sidebar navigation
section = st.sidebar.radio("Navigate", [
    "Upload Data",
    "App Usage",
    "Search History",
    "Call Logs",
    "Web History",
    "Chats",
    "Final Verdict"
])

# Session State for storing uploaded data
if "data" not in st.session_state:
    st.session_state.data = {}

# Section: Upload Data
if section == "Upload Data":
    st.header("📂 Upload CSV Files for Analysis")

    uploaded_files = {
        "app_usage": st.file_uploader("Upload App Usage CSV", type="csv"),
        "search_history": st.file_uploader("Upload Search History CSV", type="csv"),
        "call_logs": st.file_uploader("Upload Call Logs CSV", type="csv"),
        "web_history": st.file_uploader("Upload Web History CSV", type="csv"),
        "chats": st.file_uploader("Upload Chats CSV", type="csv")
    }

    for key, file in uploaded_files.items():
        if file is not None:
            st.session_state.data[key] = pd.read_csv(file)
            st.success(f"Uploaded ✅ {key.replace('_', ' ').title()}")

# Section: Call Logs Analysis
elif section == "Call Logs":
    st.header("📞 Call Logs Analysis")

    if "call_logs" in st.session_state.data:
        df = st.session_state.data["call_logs"]

        # Clean and convert Timestamp
        if 'Start Time' not in df.columns:
            df["Timestamp"] = df["Timestamp"].str.replace(r"\(UTC\+\d+\)", "", regex=True).str.strip()
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
            df['Start Time'] = df['Timestamp']
        df['Start Time'] = pd.to_datetime(df['Start Time'], errors="coerce")

        # Date range picker
        start_date, end_date = st.date_input(
            "Select Date Range",
            [df['Start Time'].min().date(), df['Start Time'].max().date()],
            min_value=df['Start Time'].min().date(),
            max_value=df['Start Time'].max().date()
        )

        # Filter by date
        mask = (df['Start Time'].dt.date >= start_date) & (df['Start Time'].dt.date <= end_date)
        filtered_df = df.loc[mask]

        # Show DataFrame
        st.subheader("Filtered Call Logs")
        st.dataframe(filtered_df)

        # 1. Calls per day
        filtered_df['Date'] = filtered_df['Start Time'].dt.date
        calls_per_day = filtered_df['Date'].value_counts().reset_index()
        calls_per_day.columns = ['Date', 'Number of Calls']
        st.plotly_chart(px.bar(calls_per_day, x='Date', y='Number of Calls', title='Number of Calls per Day'))

        # 2. Day vs. Night
        filtered_df['Hour'] = filtered_df['Start Time'].dt.hour
        filtered_df['Time of Day'] = filtered_df['Hour'].apply(lambda x: 'Day' if 6 <= x < 18 else 'Night')
        time_of_day_counts = filtered_df['Time of Day'].value_counts().reset_index()
        time_of_day_counts.columns = ['Time of Day', 'Number of Calls']
        st.plotly_chart(px.bar(time_of_day_counts, x='Time of Day', y='Number of Calls', title='Day vs. Night Calls'))

        # 3. Status distribution
        if 'Status' in filtered_df.columns:
            status_counts = filtered_df['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            st.plotly_chart(px.pie(status_counts, names='Status', values='Count', title='Call Status Distribution'))

        # 4. Top contacts
        N = st.number_input('Enter the number of top contacts to display:', min_value=1, value=5)
        top_contacts = filtered_df['Parties'].value_counts().head(N).reset_index()
        top_contacts.columns = ['Contact', 'Number of Calls']
        st.plotly_chart(px.bar(top_contacts, x='Contact', y='Number of Calls', title=f'Top {N} Frequently Contacted Persons'))

        # 5. Video call frequency
        if 'Video call' in filtered_df.columns:
            video_calls = filtered_df[filtered_df['Video call'].str.lower() == 'yes']
            video_call_counts = video_calls['Parties'].value_counts().reset_index()
            video_call_counts.columns = ['Contact', 'Number of Video Calls']
            st.plotly_chart(px.bar(video_call_counts, x='Contact', y='Number of Video Calls', title='Video Call Frequency'))

        # LLM analysis
        prompt_template = """
        You are a digital forensic psychologist.

        Analyze the following call behavior of a WhatsApp contact and assess:
        1. The person's mental health condition.
        2. Whether this call pattern might indicate criminal intent or suspicious activity.
        3. Give reasons for your assessment.

        ### Call Behavior Summary:
        {summary}

        Respond in this format:
        - PhoneNumber:
        - Mental Health Insight:
        - Criminal Intent Risk:
        - Explanation:
        """

        def analyze_call_behavior(summary):
            prompt = prompt_template.format(summary=summary)
            response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
            return response['message']['content']

        summaries = filtered_df.apply(lambda row: f"PhoneNumber: {row['Parties']}, Duration: {row['Duration']}, Status: {row['Status']}, Video Call: {row['Video call']}", axis=1)
        results = [analyze_call_behavior(summary) for summary in summaries]

        st.subheader("🧠 LLM Analysis Results")
        for result in results:
            st.text(result)

    else:
        st.warning("⚠️ Please upload the Call Logs CSV first from 'Upload Data'.")

# Generic placeholder for other sections
elif section in ["App Usage", "Search History", "Web History", "Chats"]:
    st.header(f"📁 {section} Analysis")

    data_key = section.lower().replace(" ", "_")
    if data_key in st.session_state.data:
        df = st.session_state.data[data_key]
        st.subheader("📌 Sample Data")
        st.dataframe(df.head())
        st.info("📈 Plots and LLM verdict will be displayed here soon.")
    else:
        st.warning("⚠️ Please upload this CSV first from 'Upload Data'.")

# Final Verdict Section
elif section == "Final Verdict":
    st.header("🧠 Final Verdict")
    st.info("🤖 LLM-generated summary of user's mental health and criminal intent will appear here.")
