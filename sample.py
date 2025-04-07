import streamlit as st
import pandas as pd
from datetime import datetime
import re
import plotly.express as px
import ollama

# Sample DataFrame loading (replace with actual data loading)
df = pd.read_csv('cleaned_call_logs.csv')

# Ensure 'Start Time' column exists
if 'Start Time' not in df.columns:
    # Extract 'Start Time' from 'Timestamp' if necessary


# Step 1: Clean and convert Timestamp
    df["Timestamp"] = df["Timestamp"].str.replace(r"\(UTC\+\d+\)", "", regex=True).str.strip()

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], 
        format="%d-%m-%Y %H:%M:%S", 
        errors="coerce"
    )
    df['Start Time'] = df['Timestamp']

# Convert 'Start Time' to datetime
df['Start Time'] = pd.to_datetime(df['Start Time'], format='%d-%m-%Y %H:%M:%S')

# Sidebar navigation
section = st.sidebar.radio("Navigate", ["Upload Data", "Call Logs", "Other Sections..."])

if section == "Call Logs":
    st.header("📞 Call Logs Analysis")

    # Date range picker
    start_date, end_date = st.date_input(
        "Select Date Range",
        [df['Start Time'].min().date(), df['Start Time'].max().date()],
        min_value=df['Start Time'].min().date(),
        max_value=df['Start Time'].max().date()
    )

    # Filter data based on date range
    if start_date and end_date:
        mask = (df['Start Time'].dt.date >= start_date) & (df['Start Time'].dt.date <= end_date)
        filtered_df = df.loc[mask]
    else:
        filtered_df = df

    # Display filtered data
    st.subheader("Filtered Call Logs")
    st.dataframe(filtered_df)

    # Placeholder for future visualizations and LLM verdict
    st.info("📈 Visualizations and LLM verdict will be displayed here in future steps.")
    filtered_df['Date'] = filtered_df['Start Time'].dt.date
    calls_per_day = filtered_df['Date'].value_counts().reset_index()
    calls_per_day.columns = ['Date', 'Number of Calls']
    fig1 = px.bar(calls_per_day, x='Date', y='Number of Calls', title='Number of Calls per Day')
    st.plotly_chart(fig1)

    # 2. Count of Day vs. Night Calls
    filtered_df['Hour'] = filtered_df['Start Time'].dt.hour
    filtered_df['Time of Day'] = filtered_df['Hour'].apply(lambda x: 'Day' if 6 <= x < 18 else 'Night')
    time_of_day_counts = filtered_df['Time of Day'].value_counts().reset_index()
    time_of_day_counts.columns = ['Time of Day', 'Number of Calls']
    fig2 = px.bar(time_of_day_counts, x='Time of Day', y='Number of Calls', title='Day vs. Night Calls')
    st.plotly_chart(fig2)

    # 3. Call Status Distribution
    status_counts = filtered_df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig3 = px.pie(status_counts, names='Status', values='Count', title='Call Status Distribution')
    st.plotly_chart(fig3)

    # 4. Top N Frequently Contacted Persons
    N = st.number_input('Enter the number of top contacts to display:', min_value=1, value=5)
    top_contacts = filtered_df['Parties'].value_counts().head(N).reset_index()
    top_contacts.columns = ['Contact', 'Number of Calls']
    fig4 = px.bar(top_contacts, x='Contact', y='Number of Calls', title=f'Top {N} Frequently Contacted Persons')
    st.plotly_chart(fig4)

    # 5. Video Call Analysis
    video_calls = filtered_df[filtered_df['Video call'] == 'Yes']
    video_call_counts = video_calls['Parties'].value_counts().reset_index()
    video_call_counts.columns = ['Contact', 'Number of Video Calls']
    fig5 = px.bar(video_call_counts, x='Contact', y='Number of Video Calls', title='Video Call Frequency')
    st.plotly_chart(fig5)

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

    # Assuming 'filtered_df' is your DataFrame after applying date filters

def analyze_call_behavior(summary):
    prompt = prompt_template.format(summary=summary)
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Generate summaries for analysis
summaries = filtered_df.apply(lambda row: f"PhoneNumber: {row['Parties']}, Duration: {row['Duration']}, Status: {row['Status']}, Video Call: {row['Video call']}", axis=1)

# Analyze each summary
results = [analyze_call_behavior(summary) for summary in summaries]

# Display results in Streamlit
st.subheader("LLM Analysis Results")
for result in results:
    st.text(result)

