
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-4o"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("What is up?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         )
#         response = st.write_stream(stream)
#     st.session_state.messages.append({"role": "assistant", "content": response})

from openai import OpenAI
import asyncio
import streamlit as st
import openai
import streamlit as st
import pandas as pd
import tempfile
import openai
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
installation_type_context = (
        "FRESH installs = new installation without data. If aiid == 'mmm_prw_tst_007_498_c' or Unknown, it's a FRESH install."
        "MIGRATED installs = updated software retaining old data. If aiid == 'mmm_n36_mig_000_888_m', it's a MIGRATED install."
)
st.set_page_config(layout="wide")
endpoint = os.environ["AZURE_OPENAI_BASE"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
client = AzureOpenAI(
    api_version="2024-09-01-preview",
    azure_endpoint=endpoint,
    api_key=api_key
)

st.title("Feedback Insight Chat")
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
@st.cache_data
def prepare_category_tables(df):
    category_tables = {}
    for category in df['Feature Category'].unique():
        category_rows = df[df['Feature Category'] == category]
        relevant_columns = [
            'date_ymd', 'event_timestamp_local', 'score', 'aiid', 'version_app', 'version', 'architecture',
            'city', 'region', 'country', 'uninstall_text_feedback', 'uninstall_feedback_value'
        ]
        category_table = category_rows[['text_feedback', 'Sentiment', 'Feature Category', *relevant_columns]]
        category_tables[category] = category_table
    return category_tables
@st.cache_data
def load_uploaded_csv():
    csv_path = os.path.join(downloads_folder, "uploaded_feedback_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error("CSV file not found in Downloads folder.")
        return None
@st.cache_data
def load_summaries_from_txt():
    summaries_path = os.path.join(downloads_folder, "generated_feedback_summaries.txt")
    summaries_dict = {}
    if os.path.exists(summaries_path):
        with open(summaries_path, "r") as file:
            content = file.read().split("\n" + "=" * 40 + "\n\n")
            for section in content:
                if "Category:" in section:
                    category_line, summary = section.split("\nSummary:\n", 1)
                    category = category_line.split("Category: ")[1].strip()
                    summaries_dict[category] = summary.strip()
        return summaries_dict
    else:
        st.error("Summaries file not found in Downloads folder.")
        return None


if "df" not in st.session_state:
    st.session_state["df"] = load_uploaded_csv()
if "categories_analysis" not in st.session_state:
    st.session_state["categories_analysis"] = load_summaries_from_txt()
if st.session_state["df"] is not None and st.session_state["categories_analysis"] is not None:
    df = st.session_state["df"]
    if "unique_values" not in st.session_state:
        def create_unique_column_values(df):
            unique_values = {}
            for col in df.columns:
                unique_values[col] = df[col].unique().tolist()
            st.session_state.unique_values = unique_values
        create_unique_column_values(st.session_state["df"])
    st.sidebar.title("Apply Filters")
    filters = {}
    filters['Feature Category'] = st.sidebar.multiselect("Select Features", st.session_state.unique_values["Feature Category"])
    filters['country'] = st.sidebar.multiselect("Select Country", st.session_state.unique_values["country"])
    filters['aiid'] = st.sidebar.multiselect("Select Fresh or Migrated?", st.session_state.unique_values["aiid"])
    filters['version_app'] = st.sidebar.multiselect("Select Product Versions", st.session_state.unique_values["version_app"])
    filtered_df = st.session_state["df"]
    for column, values in filters.items():
        if values:  
            filtered_df = filtered_df[filtered_df[column].isin(values)]

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    filtered_df = st.session_state["df"]
    for column, values in filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[column].isin(values)]

    if "category_tables" not in st.session_state:
        st.session_state["category_tables"] = prepare_category_tables(df)
    for category in filtered_df['Feature Category'].unique():
        st.session_state["category_tables"][category] = filtered_df[filtered_df['Feature Category'] == category][
            ['text_feedback', 'Sentiment', 'Feature Category', 'score', 'date_ymd', 'event_timestamp_local', 'aiid',
                'version_app', 'version', 'architecture', 'city', 'region', 'country', 'uninstall_text_feedback',
                'uninstall_feedback_value']
        ]
    st.session_state["load_state"] = True
    summary_info = "\n".join(
        [f"Category: {category}. Summary: {summary}" for category, summary in st.session_state["categories_analysis"].items()]
    )

    system_message = {
        "role": "system",
        "content": (
            "You are a highly intelligent assistant specializing in analyzing customer feedback data for Norton CyberSecurity. "
            "Your primary role is to answer questions thoroughly and accurately by analyzing the feedback category tables and generated summaries. "
            f"Focus exclusively on data within the specified filters: {filters}. "
            "If any filter results in insufficient or no data, clearly state this and provide insights based on available data. "
            "Always read and analyze all rows of the relevant filtered data before generating your response. Avoid fabricating information.\n\n"
            f"### Response Instructions:\n"
            "- **For Feedback-Related Questions**: Thoroughly analyze the filtered feedback data and generated summaries. "
            "Substantiate your claims and insights with proper citation of the user review(s) that you used for inference. "
            "Translate the cited reviews into English and include the translated review(s) alongside the citation.\n"
            "- **For General Questions**: If the user prompt is unrelated to feedback analysis, respond conversationally and politely, as a general AI assistant would.\n"
            "- **For Mixed Questions**: If the prompt involves both general conversation and feedback-related queries, prioritize addressing the feedback-related portion and then respond conversationally to the rest."
        )
    }
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        user_message = {
            "role": "user",
            "content": (
                f"This is the session's conversation history so far: {st.session_state.messages}\n\n"
                f"### Filters Applied:\n{filters}\n\n"
                f"### User Question:\n{prompt}\n\n"
                f"### Feedback Data:\n{st.session_state['category_tables']}\n\n"
                f"### Instructions:\n"
                "- Stick to the filtered dataset strictly and analyze all rows. "
                "If any filter has no data, state this explicitly and provide insights based on available data.\n\n"
                "- Substantiate your claims with proper citation of user reviews and translate them into English when cited.\n"
                "- If the user's prompt is unrelated to feedback analysis, reply as a conversational assistant would, maintaining a polite and friendly tone."
            )
        }
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"], 
                messages=[system_message, user_message],
                stream=True,
                temperature=0.7
            )
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.write("Please make sure the required files are in the Downloads folder to proceed.")


# st.set_page_config(layout="wide")
# endpoint = os.environ["AZURE_OPENAI_BASE"]
# api_key = os.environ["AZURE_OPENAI_API_KEY"]
# client = AzureOpenAI(
#     api_version="2024-09-01-preview",
#     azure_endpoint=endpoint,
#     api_key=api_key
# )
# st.markdown(
#     '''
#     <style>
#     .logo-container {
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         padding: 20px 0;
#     }
#     </style>
#     ''',
#     unsafe_allow_html=True,
# )
# st.markdown(
#     """
#     <div class="logo-container">
#         <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTa_iQ5H9Fpa8iIjTWniSE9_KOHuiz6kSfGDg&s"  width="250">
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
# dataset_summary = (
#     "### Dataset Summary\n\n"
#     "**Columns:**\n"
#     "- **identity.hwid**: Unique hardware identifier for each device.\n"
#     "- **event_timestamp**: Timestamp of the event, converted to a human-readable format.\n"
#     "- **date_ymd**: Date of the event in YYYY-MM-DD format.\n"
#     "- **text_feedback**: User-provided feedback text.\n"
#     "- **score**: Numeric score associated with the feedback.\n"
#     "- **origin**: Source of the feedback (e.g., platform, device).\n"
#     "- **country**: Country where the feedback originated.\n"
#     "- **region**: Region within the country of the feedback origin.\n"
#     "- **city**: City where the feedback originated.\n"
#     "- **skup**: License group or type.\n"
#     "- **version_app**: Version of the application used.\n"
#     "- **version**: Operating System platform version.\n"
#     "- **architecture**: Architecture of the platform (e.g., x86, ARM).\n"
#     "- **type**: Type of license.\n"
#     "- **aiid**: Application installation ID.\n"
#     "- **psn**: Product serial number.\n"
#     "- **uninstall_text_feedback**: Feedback provided by the user during uninstallation, or 'None' if not available.\n"
#     "- **uninstall_feedback_value**: The value or reason selected by the user during the uninstallation process.\n\n"
#     "### Key Statistics and Patterns\n\n"
#     "- **Feedback Analysis**: The `text_feedback` and `uninstall_text_feedback` fields contain qualitative feedback for sentiment analysis and theme extraction, including uninstallation reasons.\n"
#     "- **Temporal Patterns**: Use `event_timestamp` and `date_ymd` to analyze trends over time, including feedback patterns before uninstallation.\n"
#     "- **Geographic Insights**: Analyze `country`, `region`, and `city` for location-based trends and potential correlations with uninstall feedback.\n"
#     "- **Product and License Metrics**: Evaluate `version_app`, `version`, `architecture`, and `type` for product usage, license management, and uninstallation patterns.\n\n"
# )
# installation_type_context = (
#         "FRESH installs = new installation without data. If aiid == 'mmm_prw_tst_007_498_c' or Unknown, it's a FRESH install."
#         "MIGRATED installs = updated software retaining old data. If aiid == 'mmm_n36_mig_000_888_m', it's a MIGRATED install."
# )
# st.title("Feedback Insight Chat")
# downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
# @st.cache_data
# def load_uploaded_csv():
#     csv_path = os.path.join(downloads_folder, "uploaded_feedback_data.csv")
#     if os.path.exists(csv_path):
#         return pd.read_csv(csv_path)
#     else:
#         st.error("CSV file not found in Downloads folder.")
#         return None
# @st.cache_data
# def load_summaries_from_txt():
#     summaries_path = os.path.join(downloads_folder, "generated_feedback_summaries.txt")
#     summaries_dict = {}
#     if os.path.exists(summaries_path):
#         with open(summaries_path, "r") as file:
#             content = file.read().split("\n" + "=" * 40 + "\n\n")
#             for section in content:
#                 if "Category:" in section:
#                     category_line, summary = section.split("\nSummary:\n", 1)
#                     category = category_line.split("Category: ")[1].strip()
#                     summaries_dict[category] = summary.strip()
#         return summaries_dict
#     else:
#         st.error("Summaries file not found in Downloads folder.")
#         return None

# @st.cache_data
# def prepare_category_tables(df):
#     category_tables = {}
#     for category in df['Feature Category'].unique():
#         category_rows = df[df['Feature Category'] == category]
#         relevant_columns = [
#             'date_ymd', 'event_timestamp_local', 'score', 'aiid', 'version_app', 'version', 'architecture',
#             'city', 'region', 'country', 'uninstall_text_feedback', 'uninstall_feedback_value'
#         ]
#         category_table = category_rows[['text_feedback', 'Sentiment', 'Feature Category', *relevant_columns]]
#         category_tables[category] = category_table
#     return category_tables

# if "df" not in st.session_state:
#     st.session_state["df"] = load_uploaded_csv()
# if "categories_analysis" not in st.session_state:
#     st.session_state["categories_analysis"] = load_summaries_from_txt()
# if st.session_state["df"] is not None and st.session_state["categories_analysis"] is not None:
#     df = st.session_state["df"]
#     if "category_tables" not in st.session_state:
#         st.session_state["category_tables"] = prepare_category_tables(df)
#     if "openai_model" not in st.session_state:
#         st.session_state["openai_model"] = "gpt-4o"
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
#     if "response_options" not in st.session_state:
#         st.session_state["response_options"] = []
#     if "response_ready" not in st.session_state:
#         st.session_state["response_ready"] = False

#     if not st.session_state.get("load_state"):
#         for category in st.session_state["df"]['Feature Category'].unique():
#             st.session_state["category_tables"][category] = st.session_state["df"][
#                 st.session_state["df"]['Feature Category'] == category
#             ][[
#                 'text_feedback', 'Sentiment', 'Feature Category', 'score', 'date_ymd', 'event_timestamp_local',
#                 'aiid', 'version_app', 'version', 'architecture', 'city', 'region', 'country',
#                 'uninstall_text_feedback', 'uninstall_feedback_value'
#             ]]
#         st.session_state["load_state"] = True
#     summary_info = "\n".join(
#             [f"Category: {category}. Summary: {summary}" for category, summary in
#             st.session_state["categories_analysis"].items()]
#         )
#     def hide_buttons():
#         st.markdown(
#             """
#             <style>
#             button[data-testid="stBaseButton-secondary"] {
#                 display: none;
#             }
#             </style>
#             """,
#             unsafe_allow_html=True,
#         )

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"], unsafe_allow_html=True)

#     if 'chat_input_value' not in st.session_state:
#         st.session_state.chat_input_value = ""

#     system_message = {
#             "role": "system",
#             "content": (
#                 "You are a highly intelligent assistant specializing in analyzing customer feedback data for Norton CyberSecurity. "
#                 "Answer user queries based on category tables, generated summaries, and context. "
#                 "Prioritize specific data requests or trends as instructed."
#             )
#         }
#     if prompt := st.chat_input("Type your question here..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
#         user_message = {
#                 "role": "user", 
#                 "content": (
#                     f"This is the session's conversation history so far: {st.session_state.messages}\n\n"
#                     f"### User Question:\n"
#                     f"{prompt}\n\n"
#                     f"### Response Instructions:\n"
#                     f"- **For Specific Data Requests**: If the user is asking for a list, specific rows, or data points (e.g., 'Show me 20 feedback entries' or 'List feedback from city X'), "
#                     f"retrieve that data directly from the category tables and present it clearly. Do not summarize unless explicitly asked.\n"
#                     f"- **For Trends or Insights**: Thoroughly analyze the category tables first. Look for patterns, common trends, or correlations, and explain the insights in a concise way. "
#                     f"Then review any generated summaries to add additional context. Base your answer primarily on the tables, then refer to the summaries only for added context.\n"
#                     f"- **For Mixed Questions**: If a question involves both specific data and trends (e.g., 'List entries for category X and analyze trends'), respond by first providing the requested data entries, then "
#                     f"follow with a deeper analysis of trends.\n\n"
#                     f"### Feedback Data:\n"
#                     f"- **Category Tables**: Use the following tables to retrieve data or analyze trends:\n"
#                     f"{st.session_state['category_tables'].items()}\n\n"
#                     f"- **Generated Summaries**: Refer to these only for context when necessary:\n"
#                     f"{summary_info}\n\n"
#                     f"Installation type context based on aiid - If the user asks about fresh or migrated installs refer to the aiid and this context: {installation_type_context}\n"
#                     f"### Dataset Summary (for context):\n"
#                     f"{dataset_summary}\n\n"
#                     f"### Answer Structure:\n"
#                     f"- **Straightforward Response**: If the user asks for specific feedback entries or data, provide it directly and accurately. Avoid unnecessary summaries.\n"
#                     f"- **In-depth Analysis**: When asked for insights or trends, first analyze the feedback tables, then cross-reference the summaries if needed. Ensure your response is thoughtful and evidence-backed."
#                 )
#             }

#         with st.chat_message("assistant"):
#             stream = client.chat.completions.create(
#                 model=st.session_state["openai_model"],
#                 messages=[system_message, user_message],
#                 stream=True,
#                 temperature=0.1
#             )
#             response = st.write_stream(stream)

#         st.session_state.messages.append({"role": "assistant", "content": response})
#         st.session_state.response_ready = True
#         st.rerun()  

#     if st.session_state.response_ready:
#         response_generation_message = {
#             "role": "system",
#             "content": (
#                 "You are reading an AI chatbot's last message to the user, and your job is to come up with relevant prompts/questions based on what the AI responded with. "
#                 "Generate concise, natural, user-like follow-up response options relevant to the last assistant response. "
#                 "Simulate realistic user behavior by framing prompts that align with common ways users interact with the chatbot:\n"
#                 "1. Specific Data Requests: Ask for specific rows, data points, or feedback entries.\n"
#                 "2. Trends or Insights Requests: Ask about patterns, trends, or correlations in the data.\n"
#                 "3. Mixed Requests: Combine specific data retrieval with trend analysis.\n"
#                 "4. Exploratory or Hypothetical Requests: Pose speculative or exploratory questions.\n"
#                 "5. Clarifications and Follow-Ups: Ask for additional details or elaborations.\n"
#                 "Return the prompts as a numbered list, one option per line. Ensure they are concise and realistic."
#             )
#         }

#         response_prompt = {
#             "role": "user",
#             "content": (
#                 f"The last assistant response was: {st.session_state.messages[-1]['content']}\n\n"
#                 f"### Dataset Summary (for context):\n"
#                 f"{dataset_summary}\n\n"
#                 f"Installation type context based on AIID: {installation_type_context}\n\n"
#                 f"### User Interaction Context:\n"
#                 f"Users typically ask the assistant questions regarding customer feedback data or issues related to a specific feature category. "
#                 f"Examples include requesting specific feedback entries, analyzing trends in the data, exploring potential correlations, or seeking elaborations on insights provided by the assistant.\n\n"
#                 f"### Relevant Feedback Data Columns:\n"
#                 f"'text_feedback', 'Sentiment', 'Feature Category', 'score', 'date_ymd', 'event_timestamp_local', 'aiid', 'version_app', 'version', "
#                 f"'architecture', 'city', 'region', 'country', 'uninstall_text_feedback', 'uninstall_feedback_value'\n\n"
#                 f"### Classified Feature Categories:\n"
#                 f"Performance, VPN, UI/UX, Security, Installation, Licensing, Firewall, Cost, or Other\n\n"
#                 f"### Classified Sentiment Categories:\n"
#                 f"Positive, Neutral, or Negative\n\n"
#                 f"### Instructions:\n"
#                 f"Based on the last assistant response, the feedback category tables, and the dataset context, generate 3 to 4 concise follow-up human response options relevant to the prior chat. "
#                 f"Frame the options in a way that aligns with user-like behavior. For example:\n"
#                 f"- If the user asked about a certain feature's issues, suggest asking about another feature or trend.\n"
#                 f"- If the user requested insights, propose drilling deeper into a specific category or sentiment trend."
#             )
#         }
#         option_stream = client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=[response_generation_message, response_prompt],
#             stream=False,
#             max_tokens=100,
#             temperature=0.1
#         )
#         raw_response = option_stream.choices[0].message.content.strip()
#         response_options = [
#             line.strip()[3:].strip()
#             for line in raw_response.split("\n")
#             if line.strip() and line.strip()[0].isdigit() and line.strip()[1] == "."
#         ]
#         st.session_state.response_options = response_options if response_options else ["No valid options generated."]
#         st.session_state.response_ready = False

#     if st.session_state.response_options:
#         button_container = st.container()
#         with button_container:
#             for idx, option in enumerate(st.session_state.response_options):
#                 if st.button(option, key=f"response_option_{idx}"):
#                     hide_buttons()
#                     st.session_state.messages.append({"role": "user", "content": option})
#                     with st.chat_message("user"):
#                         st.markdown(f'<div class="user-message">{option}</div>', unsafe_allow_html=True)
#                     option_message = {
#                             "role": "user",
#                             "content": (
#                                 f"This is the session's conversation history so far: {st.session_state.messages}\n\n"
#                                 f"### The user selected one of the AI response options for their prompt/question:\n"
#                                 f"{option}\n\n"
#                                 f"### Response Instructions:\n"
#                                 f"- **For Specific Data Requests**: If the user is asking for a list, specific rows, or data points (e.g., 'Show me 20 feedback entries' or 'List feedback from city X'), "
#                                 f"retrieve that data directly from the category tables and present it clearly. Do not summarize unless explicitly asked.\n"
#                                 f"- **For Trends or Insights**: Thoroughly analyze the category tables first. Look for patterns, common trends, or correlations, and explain the insights in a concise way. "
#                                 f"Then review any generated summaries to add additional context. Base your answer primarily on the tables, then refer to the summaries only for added context.\n"
#                                 f"- **For Mixed Questions**: If a question involves both specific data and trends (e.g., 'List entries for category X and analyze trends'), respond by first providing the requested data entries, then "
#                                 f"follow with a deeper analysis of trends.\n\n"
#                                 f"### Feedback Data:\n"
#                                 f"- **Category Tables**: Use the following tables to retrieve data or analyze trends:\n"
#                                 f"{st.session_state['category_tables'].items()}\n\n"
#                                 f"- **Generated Summaries**: Refer to these only for context when necessary:\n"
#                                 f"{summary_info}\n\n"
#                                 f"Installation type context based on aiid - If the user asks about fresh or migrated installs refer to the aiid and this context: {installation_type_context}\n"
#                                 f"### Dataset Summary (for context):\n"
#                                 f"{dataset_summary}\n\n"
#                                 f"### Answer Structure:\n"
#                                 f"- **Straightforward Response**: If the user asks for specific feedback entries or data, provide it directly and accurately. Avoid unnecessary summaries.\n"
#                                 f"- **In-depth Analysis**: When asked for insights or trends, first analyze the feedback tables, then cross-reference the summaries if needed. Ensure your response is thoughtful and evidence-backed."
#                             )
#                         }
#                     with st.chat_message("assistant"):
#                         stream = client.chat.completions.create(
#                             model=st.session_state["openai_model"],
#                             messages=[system_message, option_message],
#                             stream=True,
#                             temperature=0.1
#                         )
#                         option_response = st.write_stream(stream)
#                     st.session_state.messages.append({"role": "assistant", "content": option_response})
#                     st.session_state.response_ready = True
#                     st.rerun()
# else:
#     st.write("Please make sure the required files are in the Downloads folder to proceed.")
