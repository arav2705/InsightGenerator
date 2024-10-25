import streamlit as st
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AzureOpenAI
from google.cloud import bigquery
import os
import ssl
from io import BytesIO
from fpdf import FPDF
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
import time
import pathlib
import json
import ijson
from streamlit_float import *
st.set_page_config(
    page_title="Feedback & App Rating Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",  
)
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;  
        padding: 20px;
        color: #000000;  
    }
    [data-testid="stSidebar"] {
        background-color: #f8c200 !important; 
        padding: 10px;
    }
    .stTabs {
        background-color: #1b1b1b; 
        color: white;
        padding: 10px;
    }
    .stTabs div[role="tab"] {
        border: 2px solid #f8c200;
        margin-right: 10px;
        padding: 10px;
        border-radius: 5px;
        color: #f8c200;
        font-size: 18px;
        font-weight: bold;
    }
    .stTabs div[aria-selected="true"] {
        background-color: #f8c200; 
        color: black;
    }
    .stButton button {
        background-color: #f8c200 !important;
        color: black !important;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .css-1x8cf1d {
        border: 2px dashed #f8c200 !important;
        color: black !important;
        background-color: #ffffff !important;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px 0;
    }
    footer {
        position: fixed;
        bottom: 0;
        width: 88%;
        background-color: #1b1b1b;  
        color: #ffffff;  
        text-align: center;
        padding: 10px 0;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="logo-container">
        <img src="https://1000logos.net/wp-content/uploads/2021/12/Norton-Logo.png" alt="Norton Logo" width="150">
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <footer>
        Norton Lifecycle - Powered by AI
    </footer>
    """,
    unsafe_allow_html=True,
)
tab0,tab1,tab3,tab4 = st.tabs(["Visualizations","Visualization Insight Generation", "Feedback Categorization", "User Prompt for Feedback"])
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
endpoint = os.environ["AZURE_OPENAI_BASE"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
print(f"Endpoint: {endpoint}, API Key: {api_key}")
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=endpoint,
    api_key=api_key
)
if "azure_openai_model" not in st.session_state:
    st.session_state["azure_openai_model"] = "gpt-4o"
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Insights Report', align='C', ln=True)
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True, align='L')
        self.ln(10)
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        cleaned_body = self.clean_text(body)
        self.multi_cell(0, 10, cleaned_body)
        self.ln()
    def clean_text(self, text):
        """Replace unsupported Unicode characters with ASCII equivalents."""
        text = text.replace(u'\u2019', "'")  
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text
    def add_figure(self, fig):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            fig.savefig(temp_file.name, format='png')
            temp_file.close()  
            self.image(temp_file.name, w=180)  
        self.ln(10)
    def add_insight(self, title, body):
        self.chapter_title(title)
        self.chapter_body(body)
def generate_pdf(all_insights, visualizations):
    pdf = PDF()
    pdf.add_page()
    for vis_name, vis_data in visualizations.items():
        pdf.chapter_title(vis_name)
        fig, ax = plt.subplots()
        if isinstance(vis_data, pd.Series):
            vis_data.plot(ax=ax, kind='line', title=vis_name)
        elif vis_data.ndim == 2:
            sns.heatmap(vis_data, ax=ax, cmap='YlGnBu')
            ax.set_title(vis_name)
        else:
            vis_data.plot(ax=ax, kind='bar', title=vis_name)
        pdf.add_figure(fig)
        plt.close(fig)
        insight_text = all_insights.get(vis_name, "")
        pdf.add_insight(f"Insights for {vis_name}", insight_text)
    return pdf.output(dest='S').encode('latin1')
class PDF2(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Feedback Analysis Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def add_summary(self, summary):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Detailed Explanations', 0, 1, 'L')
        self.set_font('Arial', '', 10)
        for category, explanation in summary.items():
            self.cell(0, 10, f'Category: {category}', 0, 1, 'L')
            explanation_str = str(explanation)  
            cleaned_explanation = self.clean_text(explanation_str)
            self.multi_cell(0, 10, cleaned_explanation)
            self.ln()
    def clean_text(self, text):
        """Replace unsupported Unicode characters with ASCII equivalents."""
        text = text.replace(u'\u2019', "'")  
        text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
        return text
def generate_feedback_pdf(categories_summary, file_path="feedback_analysis.pdf"):
    pdf = PDF2()
    pdf.add_page()
    pdf.add_summary(categories_summary)
    pdf.output(file_path)
    return file_path
clients = bigquery.Client(project='ppp-cdo-rep-ext-6c')
downloads_path = str(pathlib.Path.home() / "Downloads")
output_csv_path = os.path.join(downloads_path, 'rating_feedback_results.csv')
query = """
WITH feedback_data AS (
  SELECT DISTINCT
    identity.hwid AS hwid,
    platform.time_zone,
    TIMESTAMP_SECONDS(CAST(event.time / 1000 AS INT64)) AS event_timestamp_utc,
    FORMAT_DATETIME('%F %T', DATETIME(TIMESTAMP_SECONDS(CAST(event.time / 1000 AS INT64)), "UTC") +
      INTERVAL CAST(platform.time_zone AS INT64) MINUTE) AS event_timestamp_local,
    DATE(year, month, day) AS date_ymd,
    feedback.text_feedback AS text_feedback,
    feedback.score AS score,
    feedback.origin AS origin,
    geo.country AS country,
    geo.region AS region,
    geo.city AS city,
    license.gen_license.skup AS skup,
    product.version_app AS version_app,
    platform.version AS version,
    platform.architecture AS architecture,
    license.type AS type,
    installation.aiid AS aiid,
    license.gen_license.psn AS psn,
    event.time AS event_time 
  FROM
    `ppp-bds-data-lake-56.v4_ext.lifecycle_rating_feedback`
  WHERE
    os = 'osx'
    AND product.tenant_id = 'de527324-05e3-46eb-a2a7-43ab29c1aff7'
    AND (
        (DATE(year, month, day) BETWEEN '2024-03-01' AND '2024-09-09' AND NOT (geo.isp LIKE '%AVAST%' OR geo.isp LIKE '%Norton%'))
        OR
        (DATE(year, month, day) > '2024-09-09')
      )
),
uninstall_feedback_data AS (
  SELECT DISTINCT
    identity.hwid AS hwid,
    COALESCE(response.value,"None") AS feedback_value,
    COALESCE(section.text_feedback, "None") AS uninstall_text_feedback,
    event.time AS uninstall_event_time -- Add uninstall event time for ordering
  FROM 
    `ppp-bds-data-lake-56.v4_ext.lifecycle_uninstall_feedback`,
    UNNEST(feedback.sections) AS section,
    UNNEST(section.responses) AS response
  WHERE 
    os = 'osx'
    AND product.tenant_id = 'de527324-05e3-46eb-a2a7-43ab29c1aff7'
    AND (
        (DATE(year, month, day) BETWEEN '2024-03-01' AND '2024-09-09' AND NOT (geo.isp LIKE '%AVAST%' OR geo.isp LIKE '%Norton%'))
        OR
        (DATE(year, month, day) > '2024-09-09')
      )
),
ranked_feedback AS (
  SELECT 
    fd.*,
    ROW_NUMBER() OVER (
      PARTITION BY fd.hwid 
      ORDER BY 
        CASE WHEN fd.text_feedback IS NOT NULL THEN 1 ELSE 0 END DESC, 
        fd.event_time DESC
    ) AS rn
  FROM 
    feedback_data fd
),
ranked_uninstall_feedback AS (
  SELECT 
    ufd.hwid,
    ufd.feedback_value,
    ufd.uninstall_text_feedback,
    ROW_NUMBER() OVER (
      PARTITION BY ufd.hwid 
      ORDER BY 
        CASE WHEN ufd.uninstall_text_feedback IS NOT NULL THEN 1 ELSE 0 END DESC, 
        ufd.uninstall_event_time DESC
    ) AS rn
  FROM 
    uninstall_feedback_data ufd
)

SELECT
  rfd.hwid AS hardware_id,
  rfd.time_zone AS time_zone,
  rfd.event_timestamp_utc AS event_timestamp_utc,
  rfd.event_timestamp_local AS event_timestamp_local,
  rfd.date_ymd AS date_ymd,
  rfd.text_feedback AS text_feedback,
  rfd.score AS score,
  rfd.origin AS origin,
  rfd.country AS country,
  rfd.region AS region,
  rfd.city AS city,
  rfd.version_app AS version_app,
  rfd.version AS version,
  rfd.architecture AS architecture,
  rfd.aiid AS aiid,
  ufd.uninstall_text_feedback AS uninstall_text_feedback,
  ufd.feedback_value AS uninstall_feedback_value
FROM 
  ranked_feedback rfd
LEFT JOIN 
  ranked_uninstall_feedback ufd
ON 
  rfd.hwid = ufd.hwid
WHERE 
  rfd.rn = 1 
  AND (ufd.rn = 1 OR ufd.rn IS NULL) 
ORDER BY 
  rfd.date_ymd DESC;
"""
query_job = clients.query(query)
results = query_job.result()
df = results.to_dataframe()
df.to_csv(output_csv_path, index=False)
dataset_summary = (
    "### Dataset Summary\n\n"
    "**Columns:**\n"
    "- **identity.hwid**: Unique hardware identifier for each device.\n"
    "- **event_timestamp**: Timestamp of the event, converted to a human-readable format.\n"
    "- **date_ymd**: Date of the event in YYYY-MM-DD format.\n"
    "- **text_feedback**: User-provided feedback text.\n"
    "- **score**: Numeric score associated with the feedback.\n"
    "- **origin**: Source of the feedback (e.g., platform, device).\n"
    "- **country**: Country where the feedback originated.\n"
    "- **region**: Region within the country of the feedback origin.\n"
    "- **city**: City where the feedback originated.\n"
    "- **skup**: License group or type.\n"
    "- **version_app**: Version of the application used.\n"
    "- **version**: Operating System platform version.\n"
    "- **architecture**: Architecture of the platform (e.g., x86, ARM).\n"
    "- **type**: Type of license.\n"
    "- **aiid**: Application installation ID.\n"
    "- **psn**: Product serial number.\n"
    "- **uninstall_text_feedback**: Feedback provided by the user during uninstallation, or 'None' if not available.\n"
    "- **uninstall_feedback_value**: The value or reason selected by the user during the uninstallation process.\n\n"
    "### Key Statistics and Patterns\n\n"
    "- **Feedback Analysis**: The `text_feedback` and `uninstall_text_feedback` fields contain qualitative feedback for sentiment analysis and theme extraction, including uninstallation reasons.\n"
    "- **Temporal Patterns**: Use `event_timestamp` and `date_ymd` to analyze trends over time, including feedback patterns before uninstallation.\n"
    "- **Geographic Insights**: Analyze `country`, `region`, and `city` for location-based trends and potential correlations with uninstall feedback.\n"
    "- **Product and License Metrics**: Evaluate `version_app`, `version`, `architecture`, and `type` for product usage, license management, and uninstallation patterns.\n\n"
)
MAX_TOKENS = 8000 
def process_entire_json(file_path):
    """Process the entire JSON file and return all records in a list."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        parser = ijson.items(file, '', multiple_values=True)
        for obj in parser:
            if isinstance(obj, dict):
                data.append(obj) 
    return data
def summarize_json_data(json_data):
    """Summarize each record in the JSON data."""
    summary_list = []
    for record in json_data:
        hwid = record.get("hwid", "N/A")
        lifecycle_actions = record.get("lifecycle_actions", [])
        summary = f"HWID: {hwid}\n"
        action_summaries = []
        for action in lifecycle_actions:
            action_name = action.get("action_name", "Unknown Action")
            code = action.get("code", "Unknown Code")
            event_time = action.get("event_time", "Unknown Time")
            country = action.get("country", "Unknown Country")
            version_app = action.get("version_app", "Unknown Version")
            action_summary = (
                f"- {action_name} with Error Code {code} at {event_time} in {country}, App Version: {version_app}"
            )
            action_summaries.append(action_summary)
        summary += "\n".join(action_summaries)
        summary_list.append(summary)
    return "\n\n".join(summary_list)
def aggregate_summaries(json_data):
    """Generate a summary of the entire dataset and ensure it fits the model's token limit."""
    aggregated_summary = summarize_json_data(json_data)
    token_limit = MAX_TOKENS * 4  
    return aggregated_summary[:token_limit]  
def get_final_summary(aggregated_summary,user_question):
    """Create a prompt for the AI model with the summarized data and the user's question."""
    prompt_summary = aggregated_summary  
    prompt = (
        f"Here is the summarized version of the entire JSON data:\n{prompt_summary}\n"
        f"Question: {user_question}. "
        f"Analyze the summarized data for patterns, trends, and feedback correlations "
        f"(e.g., app version, country, fresh or migrated users, etc.). "
        f"Provide a detailed, well-reasoned answer with key insights."
    )
    
    response = client.chat.completions.create(
        model=st.session_state["azure_openai_model"],
        messages=[
            {"role": "system", "content": "You are an insight generator for Norton CyberSecurity App that answers questions based on summarized JSON data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        n=1
    )
    return response.choices[0].message.content.strip()
def stream_response(full_response):
    for chunk in full_response:
        yield chunk
def set_chat_styles():
    st.markdown(
        """
        <style>
        .user-message {
            background-color: #145374;
            color: white;
            border-radius: 10px;
            padding: 10px;
            margin: 5px;
            text-align: left;
        }
        .bot-response {
            background-color: #2C3E50;
            color: white;
            border-radius: 10px;
            padding: 10px;
            margin: 5px;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
@st.cache_data
def analyze_feedback(row, _client, feedback_column):
    install_type = "FRESH" if row['aiid'] == 'mmm_prw_tst_007_498_c' else "MIGRATED" if row['aiid'] == 'mmm_n36_mig_000_888_m' else "Unknown"
    feedback_text = row[feedback_column]
    relevant_columns = ['aiid', 'version_app', 'version', 'architecture', 'city', 'region', 'country']
    column_values = ', '.join([f"{col}: {row[col]}" for col in relevant_columns if col in row and pd.notnull(row[col])])
    prompt = (
        f"Feedback: '{feedback_text}'. Install type: {install_type}."
        f"Context: {column_values}. "
        f"Analyze this user feedback in detail, focusing on the key themes raised."
        f"Categorize the feedback into one of the following categories: Performance, VPN, UI/UX, Security, Installation, Licensing, Firewall, Cost or Other."
        f"If multiple categories apply, choose the one that seems most relevant based on the feedback content. "
        f"Additionally, assess the sentiment of the feedback as positive, neutral, or negative."
        f"Only give 2 words in your response: The feature category and the sentiment category."
    )
    response = _client.chat.completions.create(
        model=st.session_state["azure_openai_model"],
        messages=[
            {"role": "system", "content": "You are tasked with analyzing customer feedback and their data of Norton Cybersecurity and categorizing them."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=14,
        temperature=0,
        n=1
    )
    result = response.choices[0].message.content.strip()
    result_parts = result.split()
    category = result_parts[0].replace(',', '').capitalize()  
    sentiment = result_parts[1].capitalize()
    return category, sentiment
@st.cache_data
def preprocess_feedback(feedback_text):
    if not feedback_text or pd.isnull(feedback_text) or feedback_text.strip().lower() == "unknown":
        return "Unknown"
    return feedback_text.strip()
if 'feedback_analysis' not in st.session_state:
    st.session_state['feedback_analysis'] = {}
if 'categories_summary' not in st.session_state:
    st.session_state['categories_summary'] = {}
if 'visual_qna_questions' not in st.session_state:
    st.session_state['visual_qna_questions'] = []
if 'visual_qna_responses' not in st.session_state:
    st.session_state['visual_qna_responses'] = {}
@st.cache_data
def get_visual_insights_response(question, combined_insights):
    user_prompt = (
        f"Question: {question}. Provide an answer based on the following insights:\n{combined_insights}."
    )
    response = client.chat.completions.create(
        model=st.session_state["azure_openai_model"],
        messages=[
            {"role": "system", "content": "You are a goal provider for a cybersecurity team and you answer questions based on customer data from visualizations."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        n=1
    )
    return response.choices[0].message.content.strip()
def stream_response(response_text):
    sentences = response_text.split(". ")
    streamed_response = ""
    for sentence in sentences:
        streamed_response += sentence + ". "
        yield streamed_response.strip()  
        time.sleep(0.2)  
if 'all_insights' not in st.session_state:
    st.session_state['all_insights'] = {}
if 'final_summary' not in st.session_state:
    st.session_state['final_summary'] = None
if 'pdf_file' not in st.session_state:
    st.session_state['pdf_file'] = None
@st.cache_data
def plot_plotly_heatmap(data, title):
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='YlGnBu',
        zmin=data.values.min(),
        zmax=data.values.max(),
        text=data.round(2).astype(str),
        hoverinfo="text"
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Category',
        yaxis_title='Category',
        xaxis_tickangle=45
    )
    return fig
@st.cache_data
def plot_plotly_line(data, title):
    fig = px.line(
        data,
        title=title,
        labels={'index': 'Date', 'value': 'Count'},
        template='plotly_dark' 
    )
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Count',
        xaxis_tickangle=45
    )
    return fig
@st.cache_data
def plot_plotly_bar(data, title):
    fig = px.bar(
        data,
        title=title,
        labels={'index': 'Category', 'value': 'Count'},
        template='plotly_white' 
    )
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title='Count',
        xaxis_tickangle=45
    )
    return fig
@st.cache_data
def get_visualization_details(vis_data):
    if isinstance(vis_data, pd.Series):
        summary_stats = vis_data.describe().to_string()
        key_metrics = f"Min: {vis_data.min()}, Max: {vis_data.max()}, Std: {vis_data.std()}"
        data_string = f"Summary Statistics:\n{summary_stats}\n\nKey Metrics:\n{key_metrics}"
    else:
        row_means = vis_data.mean(axis=1).to_string()
        col_means = vis_data.mean(axis=0).to_string()
        data_string = (f"Row Means:\n{row_means}\n\nColumn Means:\n{col_means}\n\n"
                    f"Full Data:\n{vis_data.to_string()}")
    return data_string
with tab0:
    st.header("Automatic Insights and Visualization 🤖")
    df = pd.read_csv(output_csv_path)
    st.write("Dataset Preview:")
    st.dataframe(df)
    df = df.fillna('Unknown')
    df['event_timestamp_local'] = pd.to_datetime(df['event_timestamp_local']).dt.tz_localize(None)
    def filter_df(df, filters):
        filtered_df = df.copy()
        for column, values in filters.items():
            if values:
                filtered_df = filtered_df[filtered_df[column].isin(values)]
        return filtered_df
    columns = ['country', 'version_app', 'version','aiid']
    unique_values = {col: sorted(df[col].dropna().unique()) for col in columns}
    filters = {col: [] for col in columns}
    for col in columns:
        select_all = st.checkbox(f"Select all {col}", key=f"select_all_{col}")
        if select_all:
            filters[col] = unique_values[col]
        else:
            selected_values = st.multiselect(f"Select {col}", options=unique_values[col], default=filters[col], key=f"filter_{col}")
            filters[col] = selected_values
    filtered_df = filter_df(df, filters)
    start_date = st.date_input("Start Date", df['event_timestamp_local'].min().date())
    end_date = st.date_input("End Date", df['event_timestamp_local'].max().date())
    filtered_df = filtered_df[(filtered_df['event_timestamp_local'] >= pd.Timestamp(start_date)) &
                            (filtered_df['event_timestamp_local'] <= pd.Timestamp(end_date))]
    st.write("Time-Based Analysis")
    feedback_counts_by_date = filtered_df.groupby('date_ymd').size()
    st.write("Number of Feedbacks Over Time")
    st.line_chart(feedback_counts_by_date)
    feedback_score_by_date = filtered_df.groupby(['date_ymd', 'score']).size().unstack().fillna(0)
    st.write("Feedback Scores Over Time")
    st.line_chart(feedback_score_by_date)
    filtered_df['hour_of_day'] = filtered_df['event_timestamp_local'].dt.hour
    feedback_by_hour = filtered_df.groupby('hour_of_day').size()
    st.write("Feedback Trends by Hour of the Day")
    st.bar_chart(feedback_by_hour)
    filtered_df['day_of_week'] = filtered_df['event_timestamp_local'].dt.day_name()
    feedback_by_day = filtered_df.groupby('day_of_week').size()
    st.write("Feedback Trends by Day of the Week")
    st.bar_chart(feedback_by_day)
    feedback_score_by_country = filtered_df.groupby(['country', 'score']).size().unstack().fillna(0)
    st.write("Feedback Scores by Country")
    st.bar_chart(feedback_score_by_country)
    feedback_score_by_version_app = filtered_df.groupby(['version_app', 'score']).size().unstack().fillna(0)
    st.write("Feedback Scores by Version App")
    st.bar_chart(feedback_score_by_version_app)
    st.write("Feedback Scores by Cookie")
    def map_aiid_to_label(aiid_value):
        if aiid_value == 'mmm_prw_tst_007_498_c':
            return 'Fresh Install'
        elif aiid_value == 'mmm_n36_mig_000_888_m':
            return 'Migrated'
        else:
            return 'Other'  
    filtered_df['cookie_label'] = filtered_df['aiid'].apply(map_aiid_to_label)
    feedback_score_by_cookie = filtered_df.groupby(['cookie_label', 'score']).size().unstack().fillna(0)
    st.bar_chart(feedback_score_by_cookie)    
with tab1:
    st.subheader("Insights from Visualizations")
    visualizations = {
        "Number of Feedbacks Over Time": feedback_counts_by_date,
        "Feedback Scores Over Time": feedback_score_by_date,
        "Feedback Trends by Hour of the Day": feedback_by_hour,
        "Feedback Trends by Day of the Week": feedback_by_day,
        "Feedback Scores by Country": feedback_score_by_country,
        "Feedback Scores by Version App": feedback_score_by_version_app,
        "Feedback Scores by Cookie": feedback_score_by_cookie
    }
    installation_type_context = (
        "FRESH installs = new installation without data. If aiid == 'mmm_prw_tst_007_498_c', it's a FRESH install. "
        "MIGRATED installs = updated software retaining old data. If aiid == 'mmm_n36_mig_000_888_m', it's a MIGRATED install."
    )
    for i, (vis_name, vis_data) in enumerate(visualizations.items()):
        st.write(f"Generating insights for: {vis_name}")
        if isinstance(vis_data, pd.Series):
            fig = plot_plotly_line(vis_data, vis_name)
        elif vis_data.ndim == 2:  
            fig = plot_plotly_heatmap(vis_data, vis_name)
        else:
            fig = plot_plotly_bar(vis_data, vis_name)
        
        st.plotly_chart(fig)
        if vis_name not in st.session_state.get('all_insights', {}):
            vis_data_string = get_visualization_details(vis_data)
            if i == len(visualizations) - 1:
                visualization_prompt = (
                    f"### Dataset Context: {filtered_df}"
                    f"You can also refer to this dataset summary so you know what each column refers to: "
                    f"{dataset_summary}\n\n"
                    f"### Visualization: {vis_name}\n"
                    f"Here is the data from the visualization:\n{vis_data_string}\n\n"
                    
                    f"### Analysis Task:\n"
                    f"1. **Analyze Data**: Provide a thorough analysis of the data in the visualization. Identify patterns, trends, and any anomalies present in the data.\n"
                    f"2. **Interpret Trends**: Discuss possible causes for observed trends in the data, and reference columns such as `country`, `region`, `app_version`, `score`, or `hour_of_day`.\n"
                    f"3. **Installation Context**: Consider the installation types (FRESH vs. MIGRATED) while explaining possible reasons for trends.\n"
                    f"4. **Provide Insights**: Offer actionable insights on the implications of the trends for customer experience, software performance, and feedback quality.\n"
                    f"5. **Cross-Reference Data**: Examine correlations between geography, app version, feedback score, and other factors. Discuss how these might be impacting the trends seen in the visualization.\n\n"
                    
                    f"### FRESH vs. MIGRATED Installations Context:\n{installation_type_context}\n"
                    f"Provide a detailed and data-driven response."
                )
            else:
                visualization_prompt = (
                    f"### Dataset Context: {filtered_df}"
                    f"You can also refer to this dataset summary so you know what each column refers to: "
                    f"{dataset_summary}\n\n"
                    f"### Visualization: {vis_name}\n"
                    f"Here is the data from the visualization:\n{vis_data_string}\n\n"
                    
                    f"### Analysis Task:\n"
                    f"1. **Analyze Data**: Provide a detailed analysis of the data presented in this visualization. Identify key patterns, anomalies, and any trends.\n"
                    f"2. **Interpret Trends**: Explain possible causes for trends in the data, taking into account columns like `country`, `region`, `app_version`, `score`, etc.\n"
                    f"3. **Insights**: Provide actionable insights that can be derived from these trends and patterns, focusing on their impact on customer experience.\n"
                    f"4. **Cross-Reference**: Look for correlations between various factors such as app version, geography, and feedback score. Discuss the broader implications of these relationships.\n\n"
                    
                    f"### FRESH vs. MIGRATED Installations Context:\n{installation_type_context}\n"
                    f"Provide a well-reasoned and structured response based on the data."
                )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data analyst and insight generator for Norton CyberSecurity, tasked with analyzing customer feedback data."},
                    {"role": "user", "content": visualization_prompt}
                ],
                temperature=0.2,
                n=1
            )
            visual_insight = response.choices[0].message.content.strip()
            st.session_state['all_insights'][vis_name] = visual_insight
        else:
            visual_insight = st.session_state['all_insights'][vis_name]
        
        with st.expander(f"Insights for {vis_name}"):
            st.write(visual_insight)

    if st.session_state.get('final_summary') is None:
        all_insights_combined = "\n\n".join(f"**{name}**: {insight}" for name, insight in st.session_state['all_insights'].items())
        
        final_summary_prompt = (
            f"### Dataset Context: {filtered_df}"
            f"You can also refer to this dataset summary so you know what each column refers to: "
            f"{dataset_summary}\n\n"
            f"### Comprehensive Summary of Insights:\n"
            f"Based on the following detailed insights from multiple visualizations, provide a comprehensive summary that:\n\n"
            f"1. **Identifies Cross-Visualization Patterns**: Look for recurring themes or patterns across different visualizations, such as geographical trends, app version performance, and feedback score trends.\n"
            f"2. **Discusses Correlations and Dependencies**: Analyze how factors like country, region, app version, and installation type (FRESH or MIGRATED) correlate with feedback scores. Are certain regions or app versions linked to higher scores or more negative feedback?\n"
            f"3. **Highlights Anomalies**: Identify any anomalies or inconsistencies across the visualizations and suggest potential reasons based on the data provided.\n"
            f"4. **Actionable Insights**: Provide actionable insights based on the cross-visualization trends, considering their impact on customer satisfaction, product performance, and regional or version-specific issues.\n\n"
            
            f"### Insights From Each Visualization:\n{all_insights_combined}\n\n"
            f"### Context of FRESH and MIGRATED Installs:\n{installation_type_context}\n\n"
            f"### Dataset Summary:\n{dataset_summary}\n"
        )
        
        final_summary_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data analyst for Norton CyberSecurity providing a comprehensive overview of customer feedback data."},
                {"role": "user", "content": final_summary_prompt}
            ],
            temperature=0.2,
            n=1
        )
        
        st.session_state['final_summary'] = final_summary_response.choices[0].message.content
    with st.expander("Comprehensive Summary of Insights Across Visualizations"):
        st.write(st.session_state['final_summary'])
    if st.session_state.get('pdf_file') is None:
        pdf = generate_pdf(st.session_state['all_insights'], visualizations)
        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_file.write(pdf)
        pdf_file.flush()
        st.session_state['pdf_file'] = pdf_file.name
    st.download_button(
        label="Download Insights as PDF", 
        data=open(st.session_state['pdf_file'], "rb").read(), 
        file_name="insights_report.pdf", 
        mime="application/pdf"
    )
# with tab2:
#     user_question = st.text_input("Ask a question about the visualizations:")
#     if user_question:
#         response_placeholder = st.empty()
#         response_placeholder.write("Generating response...")
#         if user_question in st.session_state['visual_qna_responses']:
#             chat_response = st.session_state['visual_qna_responses'][user_question]
#         else:
#             all_insights_combined = "\n\n".join(f"**{name}**: {insight}" for name, insight in st.session_state['all_insights'].items())
#             full_response = get_visual_insights_response(user_question, all_insights_combined)
#             st.session_state['visual_qna_responses'][user_question] = full_response
#             chat_response = ""
#             for chunk in stream_response(full_response):
#                 chat_response = chunk
#                 response_placeholder.write(chat_response)
#         response_placeholder.write(chat_response)
with tab3:
    df['text_feedback'] = df['text_feedback'].apply(preprocess_feedback)
    feedback_column = 'text_feedback'
    if 'categories' not in st.session_state:
        st.session_state['categories'] = []
    if 'sentiments' not in st.session_state:
        st.session_state['sentiments'] = []
    if 'results_df' not in st.session_state:
        df_sample = df
        categories = []
        sentiments = []
        for index, row in df_sample.iterrows():
            category, sentiment = analyze_feedback(row, client, feedback_column)
            categories.append(category)
            sentiments.append(sentiment)
        results_df = df_sample.copy()
        results_df['Feature Category'] = categories
        results_df['Sentiment'] = sentiments
        st.session_state['results_df'] = results_df
    else:
        results_df = st.session_state['results_df']
    st.write("Feedback Categorization and Sentiment Analysis:")
    st.write(results_df)
    category_counts = results_df['Feature Category'].value_counts()
    st.write("Feature Category Counts:")
    st.write(category_counts)
    sentiment_counts = results_df['Sentiment'].value_counts()
    st.write("Sentiment Category Counts:")
    st.write(sentiment_counts)
    if 'category_tables' not in st.session_state:
        st.session_state['category_tables'] = {}
    for category in results_df['Feature Category'].unique():
        category_rows = results_df[results_df['Feature Category'] == category]
        relevant_columns = [
            'aiid', 'version_app', 'version', 'architecture', 'city', 
            'region', 'country', 'uninstall_text_feedback', 'uninstall_feedback_value'
        ]
        category_table = category_rows[['text_feedback', 'Sentiment', 'Feature Category', *relevant_columns]]
        st.session_state['category_tables'][category] = category_table
        st.write(f"Category Feedback Table for '{category}':")
        st.dataframe(category_table)
    @st.cache_data
    def get_category_summary(category, feedback_rows, feedback_column):
        category_tables = st.session_state['category_tables']
        explanation_prompt = (
            f"Analyse the following Category Feedback Tables for all the feedback categories':\n"
            f"{category_tables}\n\n"
            f"## Category Summary for: {category}\n\n"
            f"You can also refer to this dataset summary so you know what each column refers to: "
            f"{dataset_summary}\n\n"
            f"### Overview:\n"
            f"Here is a summary of feedback categorized as '{category}'.\n\n"
            f"### Intersecting Trends:\n"
            f"Analyze the feedback for any trends that may intersect with other categories, such as VPN and Firewall, "
            f"or if there are patterns observed across regions or device types. Compare with the tables from other categories.\n\n"
            f"### Instructions:\n"
            f"Your task is to generate a detailed summary and analysis of the feedback provided. Follow the guidelines below:\n\n"
            f"1. **General Overview**: Begin by providing a high-level overview of the feedback for the '{category}' category.\n"
            f"2. **Key Issues and Concerns**: Identify and list the main issues and concerns raised by users. Highlight any recurring themes in the feedback. Mention any commonly reported bugs, performance issues, or feature requests.\n"
            f"3. **Sentiment**: Note any recurring positive, negative, or neutral sentiments across different feedback entries. Provide counts or percentages of sentiments where possible (e.g., 60% positive, 30% negative, 10% neutral).\n"
            f"4. **Contextual Insights**: Dive deeper into contextual insights from the feedback:\n"
            f"   - **Version-Specific Insights**: Identify any trends related to specific app versions or architecture (e.g., issues that occur more frequently in certain app versions or only on specific device architectures).\n"
            f"   - **Regional Variations**: Identify and analyze any significant regional variations in feedback, such as common issues in certain countries, cities, or regions.\n"
            f"   - **Device/Architecture Issues**: Highlight any issues that may be unique to specific device types (e.g., phones, tablets, different operating systems) or architectures (e.g., x86, ARM).\n"
            f"   - **Uninstall Feedback**: Examine if uninstall feedback and reasons correlate with specific patterns (e.g., regional differences, version issues, etc.).\n"
            f"5. **Patterns and Trends**: Note any observable patterns in feedback over time. Mention whether issues are new, ongoing, or have been resolved in recent updates. If you have access to previous summaries, compare the current issues with past feedback to identify new or recurring problems.\n"
            f"6. **Notable Comments**: Select and analyze notable feedback entries that stand out (either positively or negatively). Provide detailed insights about these comments and their potential implications.\n"
            f"7. **Sentiment vs. Contextual Factors**: Explore any correlations between feedback sentiment and contextual factors such as version, region, or device type (e.g., negative feedback from specific regions or devices).\n"
            f"8. **Localized Feedback Clusters**: Mention if you notice localized clusters of negative feedback (e.g., certain regions or cities with disproportionately high negative sentiment).\n\n"
            f"### Summary of Key Insights:\n"
            f"- **Main Issues**: Summarize the top concerns raised in this category and back them up with feedback quotes.\n"
            f"- **Recurring Themes**: Highlight recurring feedback themes across various contexts and provide 2-3 quotes for each theme.\n"
            f"- **Regional/Version Trends**: Note any significant regional differences or app version issues, along with relevant quotes.\n"
            f"- **Uninstall Feedback Trends**: Include trends or reasons linked to uninstall feedback if relevant, supported by quotes.\n"
            f"- **Potential Reasons**: Provide possible explanations for the issues raised in the feedback, supported by user quotes (e.g., compatibility issues, misunderstood features, bugs in recent updates).\n\n"
            f"### Conclusion:\n"
            f"Based on the above, provide a final conclusion and, if possible, offer recommendations for addressing the key issues raised by users. Suggest potential fixes, improvements, or further investigation areas to improve user satisfaction in the {category} category."
        )

        response = client.chat.completions.create(
            model=st.session_state["azure_openai_model"],
            messages=[
                {"role": "system", "content": "You are a smart insight generator and feedback summarizer for customer feedback of a Norton CyberSecurity App."},
                {"role": "user", "content": explanation_prompt}
            ],
            max_tokens=4000,
            temperature=0.2,
            n=1
        )
        return response.choices[0].message.content.strip()

    @st.cache_data
    def plot_stacked_bar_chart(results_df):
        category_sentiment_counts = pd.crosstab(results_df['Feature Category'], results_df['Sentiment'])
        st.bar_chart(category_sentiment_counts)
    plot_stacked_bar_chart(results_df)
    st.download_button("Download Analysis", df.to_csv(index=False), file_name="analysis_with_categories.csv")
    if 'categories_summary' not in st.session_state:
        st.session_state['categories_summary'] = {}
    for category in results_df['Feature Category'].unique():
        if category not in st.session_state['categories_summary']:
            category_rows = results_df[results_df['Feature Category'] == category]
            st.session_state['categories_summary'][category] = get_category_summary(category, category_rows, 'text_feedback')
    st.write("Detailed Explanations for Each Feedback Category:")
    for category, explanation in st.session_state['categories_summary'].items():
        with st.expander(f"Category: {category}"):
            st.write(explanation)
    if 'pdf_file_path' not in st.session_state:
        pdf_file_path = generate_feedback_pdf(st.session_state['categories_summary'])
        st.session_state['pdf_file_path'] = pdf_file_path
    with open(st.session_state['pdf_file_path'], "rb") as pdf_file:
        st.download_button(
            label="Download Feedback Analysis PDF",
            data=pdf_file,
            file_name="feedback_analysis.pdf",
            mime="application/pdf",
            key="pdf_download"
        )
float_init(theme=True, include_unstable_primary=False)

def chat_content():
    st.session_state['chat_history'].append(st.session_state['content'])

with tab4:
    st.title("Feedback Insight Chat")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'categories_analysis' not in st.session_state or not st.session_state['categories_analysis']:
        st.write("No feedback summaries available. Please visit the 'Feedback Categorization' tab first.")
    else:
        if 'category_tables' not in st.session_state:
            st.session_state['category_tables'] = {}
        if 'load_state' not in st.session_state:    
            st.session_state['load_state'] = False  
        if not st.session_state['load_state']:  
            for category in st.session_state['categories_analysis'].keys():
                category_rows = df[df['Feature Category'] == category]
                st.session_state['category_tables'][category] = category_rows[['text_feedback', 'Sentiment', 'Feature Category', 'aiid', 'version_app', 'version', 'architecture', 'city', 'region', 'country']]
            st.session_state['load_state'] = True  

        col1, = st.columns(1)
        with col1:
            with st.container(border=True):
                if 'messages' not in st.session_state:
                    st.session_state.messages = []

                with st.container():
                    st.chat_input(key='content', on_submit=chat_content) 
                    button_b_pos = "3rem"
                    button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
                    float_parent(css=button_css)
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                prompt = st.session_state.get('content')
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    summary_info = "\n".join([f"Category: {category}. Summary: {summary}" 
                                              for category, summary in st.session_state['categories_analysis'].items()])
                    
                    system_message = {
                        "role": "system", 
                        "content": (
                            "You are a highly intelligent assistant specializing in analyzing customer feedback data for Norton CyberSecurity. "
                            "Your role is to answer questions thoroughly and accurately by analyzing the feedback category tables and generated summaries. "
                            "Prioritize relevant tables for answers, and always examine the tables first when asked specific questions. "
                            "If asked for specific data, insights, or quotes, retrieve them directly from the category tables and avoid summarizing unnecessarily. "
                            "When responding to general, unrelated, or casual questions (such as greetings or non-feedback topics), reply conversationally without referencing feedback data."
                        )
                    }

                    user_message = {
                    "role": "user", 
                    "content": (
                        f"This is the session's conversation history so far: {st.session_state.messages}\n\n"
                        f"### User Question:\n"
                        f"{prompt}\n\n"
                        f"### Response Instructions:\n"
                        f"- **For Specific Data Requests**: If the user is asking for a list, specific rows, or data points (e.g., 'Show me 20 feedback entries' or 'List feedback from city X'), "
                        f"retrieve that data directly from the category tables and present it clearly. Do not summarize unless explicitly asked.\n"
                        f"- **For Trends or Insights**: Thoroughly analyze the category tables first. Look for patterns, common trends, or correlations, and explain the insights in a concise way. "
                        f"Then review any generated summaries to add additional context. Base your answer primarily on the tables, then refer to the summaries only for added context.\n"
                        f"- **For Mixed Questions**: If a question involves both specific data and trends (e.g., 'List entries for category X and analyze trends'), respond by first providing the requested data entries, then "
                        f"follow with a deeper analysis of trends.\n\n"
                        f"### Feedback Data:\n"
                        f"- **Category Tables**: Use the following tables to retrieve data or analyze trends:\n"
                        f"{st.session_state['category_tables'].items()}\n\n"
                        f"- **Generated Summaries**: Refer to these only for context when necessary:\n"
                        f"{summary_info}\n\n"
                        f"Installation type context based on aiid - If the user asks about fresh or migrated installs refer to the aiid and this context: {installation_type_context}\n"
                        f"### Dataset Summary (for context):\n"
                        f"{dataset_summary}\n\n"
                        f"### Answer Structure:\n"
                        f"- **Straightforward Response**: If the user asks for specific feedback entries or data, provide it directly and accurately. Avoid unnecessary summaries.\n"
                        f"- **In-depth Analysis**: When asked for insights or trends, first analyze the feedback tables, then cross-reference the summaries if needed. Ensure your response is thoughtful and evidence-backed."
                    )
                    }

                    with st.chat_message("assistant"):
                        stream = client.chat.completions.create(
                            model=st.session_state["azure_openai_model"],
                            messages=[
                                {"role": "system", "content": system_message["content"]}, 
                                {"role": "user", "content": user_message["content"]}
                            ],
                            stream=True,
                            temperature=0.2
                        )
                        response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        # with tab5:
        #     st.header("User Q&A Based on Large JSON File")
        #     json_file_path = r'/Users/aravind.vijayaraghav/Downloads/COMPLETE_CONSUMER_LIFECYCLE.json'
        #     st.write("Processing large JSON file...")
        #     json_data = process_entire_json(json_file_path) 
        #     aggregated_summary = aggregate_summaries(json_data)  
        #     user_question = st.text_input("Ask a question about the JSON file content:")
        #     if user_question:
        #         st.write("Generating final response...")
        #         @st.cache_data
        #         def get_final_response(user_question, aggregated_summary):
        #             return get_final_summary(aggregated_summary, user_question)
        #         response_placeholder = st.empty()
        #         final_response = get_final_response(user_question, aggregated_summary)
        #         response_placeholder.write(final_response)
