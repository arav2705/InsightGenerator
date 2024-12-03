import streamlit as st
import pandas as pd
import tempfile
import openai
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AzureOpenAI
from google.cloud import bigquery
import pandas.tseries.offsets as offsets
# import os
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
import subprocess
import time
import pathlib
import json
from streamlit_float import *
import ijson
st.set_page_config(
    page_title="Feedback & App Rating Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",  
)
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
st.markdown(
    '''
    <style>
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px 0;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="logo-container">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTa_iQ5H9Fpa8iIjTWniSE9_KOHuiz6kSfGDg&s"  width="150">
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .dataframe tbody tr td {
        font-size: 18px;  /* Adjust as needed */
    }
    .dataframe thead tr th {
        font-size: 20px;  /* Adjust as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.markdown(
#     """
#     <footer>
#         Norton Lifecycle - Powered by AI
#     </footer>
#     """,
#     unsafe_allow_html=True,
# )
installation_type_context = (
        "FRESH installs = new installation without data. If aiid == 'mmm_prw_tst_007_498_c' or Unknown, it's a FRESH install."
        "MIGRATED installs = updated software retaining old data. If aiid == 'mmm_n36_mig_000_888_m', it's a MIGRATED install."
)
tab0,tab1,tab3= st.tabs(["Visualizations","Visualization Insight Generation", "Feedback Categorization"])
ssl._create_default_https_context = ssl._create_unverified_context
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['REQUESTS_CA_BUNDLE'] = ''
endpoint = st.secrets["AZURE_OPENAI_BASE"]
api_key = st.secrets["AZURE_OPENAI_API_KEY"]
client = AzureOpenAI(
    api_version="2024-09-01-preview",
    azure_endpoint=endpoint,
    api_key=api_key
)
def map_aiid_to_label(aiid_value):
    if not aiid_value or pd.isna(aiid_value) or aiid_value == '' or aiid_value == 'NULL' or aiid_value == 'Unknown':
        return 'Fresh Installs'
    elif aiid_value == 'mmm_prw_tst_007_498_c':
        return 'Fresh Installs'
    elif aiid_value == 'mmm_n36_mig_000_888_m':
        return 'Migrated Installs'
    else:
        return 'Other'
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
@st.cache_data
def generate_pdf(all_insights, visualizations):
    pdf = PDF()
    pdf.add_page()
    for vis_name, vis_data in visualizations.items():
        pdf.chapter_title(vis_name)
        fig, ax = plt.subplots()
        
        if isinstance(vis_data, pd.Series):
            vis_data.plot(ax=ax, kind='line', title=vis_name)
        elif vis_data.ndim == 2:
            if vis_data.applymap(lambda x: isinstance(x, (int, float))).all().all():
                sns.heatmap(vis_data, ax=ax, cmap='YlGnBu')
                ax.set_title(vis_name)
            else:
                ax.text(0.5, 0.5, "Non-numeric data; heatmap skipped.", ha='center', va='center')
                ax.set_title(f"{vis_name} (Non-numeric data)")
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
@st.cache_data
def generate_feedback_pdf(categories_summary, file_path="feedback_analysis.pdf"):
    pdf = PDF2()
    pdf.add_page()
    pdf.add_summary(categories_summary)
    pdf.output(file_path)
    return file_path
@st.cache_data
def get_feature_analysis(df, _client):
    ai_analysis = []
    
    for version_app in df['version_app'].unique():  
        version_df = df[df['version_app'] == version_app]
        feature_avg_ratings = version_df.groupby('Feature Category')['score'].mean().reset_index()
        feature_avg_ratings.columns = ['Feature', 'Mean App Rating']
        
        for _, row in feature_avg_ratings.iterrows():
            feature = row['Feature']
            mean_rating = row['Mean App Rating']
            feature_feedback = version_df[version_df['Feature Category'] == feature]['text_feedback'].tolist()
            uninstall_feedback = version_df[version_df['Feature Category'] == feature]['uninstall_text_feedback'].tolist()
            uninstall_values = version_df[version_df['Feature Category'] == feature]['uninstall_feedback_value'].tolist()
            installs = version_df[version_df['Feature Category'] == feature]['aiid'].apply(map_aiid_to_label)
            fresh_installs = installs[installs == 'Fresh Installs'].count()
            migrated_installs = installs[installs == 'Migrated Installs'].count()

            sentiment_counts = version_df[version_df['Feature Category'] == feature]['Sentiment'].value_counts()
            positive_count = sentiment_counts.get('Positive', 0)
            neutral_count = sentiment_counts.get('Neutral', 0)
            negative_count = sentiment_counts.get('Negative', 0)

            summary_prompt = (
                "DON'T PROVIDE A HEADING FOR THIS RESPONSE, JUST GENERATE THE RESPONSE\n\n"
                f"Provide an in-depth analysis for feedback on '{feature}' in app version {version_app} of Norton CyberSecurity. If you are not able to see much feedback texts, please say so and end the summary, don't make up anything.\n\n"
                "Analyze both positive and negative feedback, and summarize notable positive and negative comments if applicable. "
                "Do not use any headings.\n\n"
                "Additionally, include an analysis of uninstall feedback specific to this feature, covering general themes in "
                "'uninstall_text_feedback' and 'uninstall_feedback_value'.\n\n"
                "At the end of your summary, quote some notable feedback comments (both positive and negative) that significantly impacted your summarizing process. Make sure these comments are translated to ENGLISH.\n\n"
                f"Feature Feedback:\n" + "\n".join(map(str, feature_feedback)) + "\n\n"
                f"Uninstall Feedback:\n" + "\n".join(map(str, uninstall_feedback)) + "\n\n"
                f"Uninstall Feedback Values:\n" + "\n".join(map(str, uninstall_values))
            )

            response = _client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an expert analyzer of customer feedback for Norton CyberSecurity. Generate concise, insightful "
                            "summaries for app feedback and uninstall themes. Avoid using any headings in your responses, and structure the analysis as a chatbot response."
                        )
                    },
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.1,
                n=1
            )
            ai_summary = response.choices[0].message.content.strip()

            ai_analysis.append({
                'version_app': version_app,  
                'Feature': feature, 
                'Mean App Rating': mean_rating, 
                'Fresh Installs': fresh_installs,
                'Migrated Installs': migrated_installs,
                'Positive': positive_count,
                'Neutral': neutral_count,
                'Negative': negative_count,
                'AI Summary': ai_summary,
            })
    
    return pd.DataFrame(ai_analysis)

# downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

# @st.cache_data
# def save_uploaded_csv(df):
#     csv_path = os.path.join(downloads_folder, "uploaded_feedback_data.csv")
#     df.to_csv(csv_path, index=False)
#     return csv_path

# @st.cache_data
# def save_summaries_to_txt(summaries_dict):
#     summaries_path = os.path.join(downloads_folder, "generated_feedback_summaries.txt")
#     with open(summaries_path, "w") as file:
#         for category, summary in summaries_dict.items():
#             file.write(f"Category: {category}\n")
#             file.write(f"Summary:\n{summary}\n")
#             file.write("\n" + "="*40 + "\n\n")  
#     return summaries_path

with tab3:
    uploaded_file = st.file_uploader("Upload pre-categorized feedback CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Feature Category' not in df.columns or 'Sentiment' not in df.columns:
            st.error("The uploaded CSV must contain 'Feature Category' and 'Sentiment' columns.")
        else:
            # csv_file_path = save_uploaded_csv(df)
            st.session_state['results_df'] = df  
            st.write("Feedback Categorization and Sentiment Analysis:")
            st.write(df)
            category_counts = df['Feature Category'].value_counts()
            st.write("Feature Category Counts:")
            st.write(category_counts)
            sentiment_counts = df['Sentiment'].value_counts()
            st.write("Sentiment Category Counts:")
            st.write(sentiment_counts)
            mean_ratings = df.groupby('Feature Category')['score'].mean()
            st.write("Mean App Ratings by Feature Category")
            st.bar_chart(mean_ratings)
            st.write("Trend of Mean App Ratings by Feature Category Over Time")
            df['date_ymd'] = pd.to_datetime(df['date_ymd'])
            df['week'] = df['date_ymd'].dt.to_period('W').apply(lambda r: r.start_time)
            df = df.sort_values(by=['week', 'Feature Category'])
            trend_data = (
                df.groupby(['Feature Category', 'week'])
                .apply(lambda x: pd.Series({
                    'Cumulative Mean Rating': x['score'].expanding().mean().iloc[-1]
                }))
                .reset_index()
            )
            trend_pivot = trend_data.pivot(index='week', columns='Feature Category', values='Cumulative Mean Rating')
            st.line_chart(trend_pivot)

            last_4_weeks = trend_pivot.iloc[-4:] 
            st.write("Cumulative Weekly Trend of Mean App Ratings by Feature Category (Last 3 Weeks)")
            st.line_chart(last_4_weeks)

            df['date_ymd'] = pd.to_datetime(df['date_ymd'])
            df['week'] = df['date_ymd'] - pd.to_timedelta((df['date_ymd'].dt.dayofweek - 2) % 7, unit='d')
            df = df.sort_values(by=['week', 'Feature Category'])
            trend_data = (
                df.groupby(['Feature Category', 'week'])
                .apply(lambda x: pd.Series({
                    'Cumulative Mean Rating': x['score'].expanding().mean().iloc[-1]
                }))
                .reset_index()
            )
            trend_pivot = trend_data.pivot(index='week', columns='Feature Category', values='Cumulative Mean Rating')
            st.write("Weekly Basis - Mean App Ratings by Feature Category")
            st.line_chart(trend_pivot)
            last_3_weeks = trend_pivot.iloc[-4:]
            st.write("Weekly Basis - Mean App Ratings by Feature Category (Last 3 Weeks)")
            st.line_chart(last_3_weeks)
            df['date_ymd'] = pd.to_datetime(df['date_ymd'])
            df['week'] = df['date_ymd'] - pd.to_timedelta((df['date_ymd'].dt.dayofweek - 2) % 7, unit='d')
            df = df.sort_values(by=['week', 'Feature Category'])
            df['Cumulative Mean Rating'] = df.groupby('Feature Category')['score'].expanding().mean().reset_index(level=0, drop=True)
            trend_data = df.groupby(['week', 'Feature Category']).agg({'Cumulative Mean Rating': 'last'}).reset_index()
            trend_pivot = trend_data.pivot(index='week', columns='Feature Category', values='Cumulative Mean Rating')
            st.write("Cumulative Weekly Trend of Mean App Ratings by Feature Category")
            st.line_chart(trend_pivot)
            last_3_weeks = trend_pivot.iloc[-4:]
            st.write("Cumulative Weekly Trend of Mean App Ratings by Feature Category (Last 3 Weeks)")
            st.line_chart(last_3_weeks)

            def calculate_feature_category_stats(filtered_df):
                filtered_df['cookie_label'] = filtered_df['aiid'].apply(map_aiid_to_label)
                category_stats = (
                    filtered_df.groupby('Feature Category').apply(
                        lambda group: pd.Series({
                            "Average Rating Score": group['score'].mean(),
                            "Avg Fresh Score": group.loc[group['cookie_label'] == 'Fresh Installs', 'score'].mean() if not group.loc[group['cookie_label'] == 'Fresh Installs', 'score'].empty else None,
                            "Avg Migrated Score": group.loc[group['cookie_label'] == 'Migrated Installs', 'score'].mean() if not group.loc[group['cookie_label'] == 'Migrated Installs', 'score'].empty else None,
                            "User Count": group['score'].size,
                            "Fresh Install Count": (group['cookie_label'] == 'Fresh Installs').sum(),
                            "Migrated Install Count": (group['cookie_label'] == 'Migrated Installs').sum(),
                        })
                    ).reset_index()
                )
                category_stats.columns = [
                    'Feature Category', 
                    'Average Rating Score', 
                    'Avg Fresh Score', 
                    'Avg Migrated Score', 
                    'User Count', 
                    'Fresh Install Count', 
                    'Migrated Install Count'
                ]
                category_stats = category_stats.sort_values(by='Feature Category')
                return category_stats
            df['date_ymd'] = pd.to_datetime(df['date_ymd'])
            last_4_months_start = df['date_ymd'].max() - pd.DateOffset(months=4)
            last_4_months_df = df[df['date_ymd'] >= last_4_months_start]
            last_3_weeks_start = df['date_ymd'].max() - pd.DateOffset(weeks=3)
            last_3_weeks_df = df[df['date_ymd'] >= last_3_weeks_start]
            overall_stats = calculate_feature_category_stats(df)
            st.write("Feature Category Statistics (Overall)")
            st.dataframe(overall_stats)
            last_4_months_stats = calculate_feature_category_stats(last_4_months_df)
            st.write("Feature Category Statistics (Last 4 Months)")
            st.dataframe(last_4_months_stats)
            last_3_weeks_stats = calculate_feature_category_stats(last_3_weeks_df)
            st.write("Feature Category Statistics (Last 3 Weeks)")
            st.dataframe(last_3_weeks_stats)

            df['date_ymd'] = pd.to_datetime(df['date_ymd'])
            unique_feature_categories = sorted(df['Feature Category'].dropna().unique())
            select_all_feature_categories = st.checkbox("Select all Feature Categories", key="select_all_feature_categories")
            if select_all_feature_categories:
                selected_feature_categories = unique_feature_categories
            else:
                selected_feature_categories = st.multiselect(
                    "Select Feature Categories",
                    options=unique_feature_categories,
                    default=[],
                    key="filter_feature_categories"
                )
            filtered_df = (
                df[df['Feature Category'].isin(selected_feature_categories)]
                if selected_feature_categories else df
            )
            filtered_df['week'] = filtered_df['date_ymd'] - pd.to_timedelta(
                (filtered_df['date_ymd'].dt.dayofweek - 2) % 7, unit='d'
            )
            filtered_df = filtered_df.sort_values(by=['week', 'Feature Category'])
            trend_data = (
                filtered_df.groupby(['Feature Category', 'week'])
                .apply(lambda x: pd.Series({
                    'Cumulative Mean Rating': x['score'].expanding().mean().iloc[-1]
                }))
                .reset_index()
            )
            trend_pivot = trend_data.pivot(index='week', columns='Feature Category', values='Cumulative Mean Rating')
            st.write("Weekly Basis - Mean App Ratings by Feature Category")
            st.line_chart(trend_pivot)
            last_3_weeks = trend_pivot.iloc[-4:]
            st.write("Weekly Basis - Mean App Ratings by Feature Category (Last 3 Weeks)")
            st.line_chart(last_3_weeks)
            filtered_df['Cumulative Mean Rating'] = (
                filtered_df.groupby('Feature Category')['score']
                .expanding().mean().reset_index(level=0, drop=True)
            )
            trend_data_cumulative = (
                filtered_df.groupby(['week', 'Feature Category'])
                .agg({'Cumulative Mean Rating': 'last'})
                .reset_index()
            )
            trend_pivot_cumulative = trend_data_cumulative.pivot(
                index='week', columns='Feature Category', values='Cumulative Mean Rating'
            )
            st.write("Cumulative Weekly Trend of Mean App Ratings by Feature Category")
            st.line_chart(trend_pivot_cumulative)
            last_3_weeks_cumulative = trend_pivot_cumulative.iloc[-4:]
            st.write("Cumulative Weekly Trend of Mean App Ratings by Feature Category (Last 3 Weeks)")
            st.line_chart(last_3_weeks_cumulative)
            # feature_analysis = get_feature_analysis(df, client)
            # sorted_versions = sorted(df['version_app'].unique(), reverse=True)
            # for version_app in sorted_versions: 
            #     st.write(f"### App Version: {version_app}")
            #     version_analysis = feature_analysis[feature_analysis['version_app'] == version_app][
            #         ['Feature', 'Mean App Rating', 'Fresh Installs', 'Migrated Installs', 'Positive', 'Neutral', 'Negative', 'AI Summary']
            #     ]
            #     fresh_installs_sum = version_analysis['Fresh Installs'].sum()
            #     migrated_installs_sum = version_analysis['Migrated Installs'].sum()
            #     positive_sum = version_analysis['Positive'].sum()
            #     neutral_sum = version_analysis['Neutral'].sum()
            #     negative_sum = version_analysis['Negative'].sum()
            #     summary_row = pd.DataFrame({
            #         'Feature': ['Total'],
            #         'Mean App Rating': [""],
            #         'Fresh Installs': [fresh_installs_sum],
            #         'Migrated Installs': [migrated_installs_sum],
            #         'Positive': [positive_sum],
            #         'Neutral': [neutral_sum],
            #         'Negative': [negative_sum],
            #         'AI Summary': [""]
            #     })
            #     version_analysis = pd.concat([version_analysis, summary_row], ignore_index=True)
            #     st.dataframe(version_analysis)
            relevant_columns = [
                            'date_ymd', 'event_timestamp_local', 'score', 'aiid', 'version_app', 'version',
                            'architecture', 'city', 'region', 'country', 'uninstall_text_feedback', 'uninstall_feedback_value'
            ]
            if 'category_tables' not in st.session_state:
                @st.cache_data
                def generate_category_tables(df):
                    category_tables = {}
                    for category in df['Feature Category'].unique():
                        category_rows = df[df['Feature Category'] == category]
                        relevant_columns = [
                            'date_ymd', 'event_timestamp_local', 'score', 'aiid', 'version_app', 'version',
                            'architecture', 'city', 'region', 'country', 'uninstall_text_feedback', 'uninstall_feedback_value'
                        ]
                        category_table = category_rows[['text_feedback', 'Sentiment', 'Feature Category', *relevant_columns]]
                        category_tables[category] = category_table
                    return category_tables
                st.session_state['category_tables'] = generate_category_tables(df)

            for category, table in st.session_state['category_tables'].items():
                st.write(f"Category Feedback Table for '{category}':")
                st.dataframe(table)

            @st.cache_data
            def get_category_analysis(category, feedback_rows, feedback_column):
                category_tables = st.session_state['category_tables']
                relevant_columns_display = ", ".join(relevant_columns)
                explanation_prompt = (
                    f"## Detailed Analysis for: {category}\n\n"
                    f"### Overview:\n"
                    f"Conduct an in-depth analysis of all feedback categorized as '{category}'. Focus on identifying "
                    f"trends, providing feedback quotes, and offering insights based on the available data across the following relevant columns: "
                    f"{relevant_columns_display}.\n\n"
                    f"### Category Feedback Data:\n"
                    f"{category_tables}\n\n"
                    f"### Contextual Insights and Trends:\n"
                    f"Use the relevant columns to uncover meaningful insights. Analyze patterns across app versions, regions, architectures, "
                    f"and uninstall feedback, considering how these factors influence user sentiment and behavior. Be sure to include "
                    f"substantive quotes from the feedback to justify your insights and provide real-world context.\n"
                    f"Also, explore possible correlations or overlaps with other categories (e.g., VPN, Firewall).\n\n"
                    f"### Instructions:\n"
                    f"1. **Key Insights**: Provide a detailed summary of the key issues raised for '{category}'. Focus on identifying trends, repeating patterns, or common concerns across the relevant data.\n"
                    f"2. **Sentiment Analysis**: Dive into the sentiment distribution for this category, using feedback quotes to highlight significant sentiment trends (e.g., clusters of negative feedback in specific regions or versions).\n"
                    f"3. **Version-Specific Trends**: Highlight any issues tied to specific app versions or architectures, and relate these to sentiment trends. Use quotes from users to substantiate the claims.\n"
                    f"4. **Regional Insights**: Analyze any regional variations in feedback and include device-specific patterns where applicable. Compare these trends with those in other categories.\n"
                    f"5. **Installation type Insights**: {installation_type_context} .Analyze the feedback based on the installation type (FRESH or MIGRATED) and provide insights on the differences in feedback between these types.\n"
                    f"5. **Uninstall Feedback Trends**: Investigate if uninstall feedback is correlated with specific features, regions, or app versions. Analyze uninstall feedback for patterns or critical issues.\n"
                    f"6. **Inter-category Trends**: Identify overlaps or divergences in feedback between '{category}' and other categories. For example, are similar trends present in feedback for VPN or Firewall? Compare trends using feedback quotes where applicable.\n"
                    f"7. **Actionable Insights**: Summarize findings and provide actionable recommendations for improving user experience based on the detailed analysis. "
                    f"Be sure to focus on critical areas of concern, patterns of dissatisfaction, and common feedback themes.\n\n"
                    f"### Final Summary:\n"
                    f"Summarize the findings for the {category} category and provide recommendations based on trends observed across the relevant columns."
                )

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert analyst for going through customer feedback along with their customer data and generating detailed insights based on all their data."},
                        {"role": "user", "content": explanation_prompt}
                    ],
                    temperature=0.1,
                    n=1
                )
                return response.choices[0].message.content


            @st.cache_data
            def plot_stacked_bar_chart(results_df):
                category_sentiment_counts = pd.crosstab(results_df['Feature Category'], results_df['Sentiment'])
                st.bar_chart(category_sentiment_counts)

            plot_stacked_bar_chart(df)
            # st.download_button("Download Analysis", df.to_csv(index=False), file_name="analysis_with_categories.csv")

            @st.cache_data
            def generate_categories_analysis(df):
                categories_analysis = {}
                for category in df['Feature Category'].unique():
                    category_rows = df[df['Feature Category'] == category]
                    categories_analysis[category] = get_category_analysis(category, category_rows, 'text_feedback')
                return categories_analysis

            
            if 'categories_analysis' not in st.session_state:
                st.session_state['categories_analysis'] = generate_categories_analysis(df)

            st.write("Detailed Analysis for Each Feedback Category:")
            for category, analysis in st.session_state['categories_analysis'].items():
                with st.expander(f"Category: {category}"):
                    st.write(analysis)

            # summaries_path = save_summaries_to_txt(st.session_state['categories_analysis'])

            if 'pdf_file_path' not in st.session_state:
                pdf_file_path = generate_feedback_pdf(st.session_state['categories_analysis'])
                st.session_state['pdf_file_path'] = pdf_file_path

            # with open(st.session_state['pdf_file_path'], "rb") as pdf_file:
            #     st.download_button(
            #         label="Download Feedback Analysis PDF",
            #         data=pdf_file,
            #         file_name="feedback_analysis.pdf",
            #         mime="application/pdf",
            #         key="pdf_download"
            #     )

            file_to_run = "Another_AI_Test.py"

            if st.button("Open Feedback Insight Chat"):
                try:
                    subprocess.Popen(["streamlit", "run", file_to_run])
                    st.success("Feedback Insight Chat is opening...")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

            # if st.button("Open Feedback Insight Chat"):
            #    subprocess.Popen(["streamlit", "run", "/Users/aravind.vijayaraghav/Documents/Data_AI_ML/Another_AI_Test.py"])
            #    st.rerun()
    else:
        st.write("Please upload a pre-categorized feedback CSV to proceed.")

clients = bigquery.Client(project='ppp-cdo-rep-ext-6c')
downloads_path = str(pathlib.Path.home() / "Downloads")
# output_csv_path = os.path.join(downloads_path, 'rating_feedback_results.csv')
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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an insight generator for Norton CyberSecurity App that answers questions based on summarized JSON data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.1,
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
    install_type = (
        "FRESH" if not row['aiid'] or pd.isna(row['aiid']) or row['aiid'] == '' 
        else "FRESH" if row['aiid'] == 'mmm_prw_tst_007_498_c' 
        else "MIGRATED" if row['aiid'] == 'mmm_n36_mig_000_888_m' 
        else "Unknown"
    )
    feedback_text = row[feedback_column]
    relevant_columns = ['uninstall_text_feedback','uninstall_feedback_value','aiid', 'version_app', 'version', 'architecture', 'city', 'region', 'country']
    column_values = ', '.join([f"{col}: {row[col]}" for col in relevant_columns if col in row and pd.notnull(row[col])])
    prompt = (
        f"Install type: {install_type}. Feedback: '{feedback_text}'. "
        f"Context: {column_values}. "
        f"Analyze this user feedback in detail, focusing on the key themes raised."
        f"Categorize the feedback into one of the following categories: Performance, VPN, UI/UX, Security, Installation, Licensing, Firewall, Cost"
        f"If multiple categories apply, choose the one that seems most relevant based on the feedback content. "
        f"Additionally, assess the sentiment of the feedback as positive, neutral, or negative."
    )
    response = _client.chat.completions.create(
        model=st.session_state["azure_openai_model"],
        messages=[
            {"role": "system", "content": "You are tasked with analyzing user feedback in the context of software installation and product usage."},
            {"role": "system", "content": "You will categorize the feedback into one of the predefined categories: Performance, VPN, UI/UX, Security, Installation, Licensing, Firewall, or Other. When multiple categories apply, select the most relevant."},
            {"role": "system", "content": "Only give 2 words in your response: The feature category and the sentiment category."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a goal provider for a cybersecurity team and you answer questions based on customer data from visualizations."},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1500,
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
    data_numeric = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=data_numeric.values,
        x=data_numeric.columns,
        y=data_numeric.index,
        colorscale='YlGnBu',
        zmin=data_numeric.values.min(),
        zmax=data_numeric.values.max(),
        text=data_numeric.round(2).astype(str),
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
    elif isinstance(vis_data, pd.DataFrame):
        overall_summary = vis_data.describe().to_string()
        data_string = (
            f"Overall Summary Statistics:\n{overall_summary}\n\n"
            f"Full Data:\n{vis_data.head().to_string()}\n\n(Note: Displaying top rows)"
        )
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
    filtered_df['cookie_label'] = filtered_df['aiid'].apply(map_aiid_to_label)
    st.write("Average Rating Score, User Count, and Install Counts by App Version")
    version_stats = (
        filtered_df.groupby('version_app').apply(
            lambda group: pd.Series({
                "Average Rating Score": group['score'].mean(),
                "Avg Fresh Score": group.loc[group['cookie_label'] == 'Fresh Installs', 'score'].mean() if not group.loc[group['cookie_label'] == 'Fresh Installs', 'score'].empty else None,
                "Avg Migrated Score": group.loc[group['cookie_label'] == 'Migrated Installs', 'score'].mean() if not group.loc[group['cookie_label'] == 'Migrated Installs', 'score'].empty else None,
                "User Count": group['score'].size,
                "Fresh Install Count": (group['cookie_label'] == 'Fresh Installs').sum(),
                "Migrated Install Count": (group['cookie_label'] == 'Migrated Installs').sum(),
            })
        ).reset_index()
    )
    version_stats.columns = [
        'App Version', 
        'Average Rating Score', 
        'Avg Fresh Score', 
        'Avg Migrated Score', 
        'User Count', 
        'Fresh Install Count', 
        'Migrated Install Count'
    ]
    version_stats = version_stats.sort_values(by='App Version', ascending=False)
    st.dataframe(version_stats)

    df['date_ymd'] = pd.to_datetime(df['date_ymd'])
    last_4_months_start = df['date_ymd'].max() - pd.DateOffset(months=4)
    last_4_months_df = df[df['date_ymd'] >= last_4_months_start]
    last_3_weeks_start = df['date_ymd'].max() - pd.DateOffset(weeks=3)
    last_3_weeks_df = df[df['date_ymd'] >= last_3_weeks_start]
    def calculate_version_stats(filtered_df):
        filtered_df['cookie_label'] = filtered_df['aiid'].apply(map_aiid_to_label)
        version_stats = (
            filtered_df.groupby('version_app').apply(
                lambda group: pd.Series({
                    "Average Rating Score": group['score'].mean(),
                    "Avg Fresh Score": group.loc[group['cookie_label'] == 'Fresh Installs', 'score'].mean() if not group.loc[group['cookie_label'] == 'Fresh Installs', 'score'].empty else None,
                    "Avg Migrated Score": group.loc[group['cookie_label'] == 'Migrated Installs', 'score'].mean() if not group.loc[group['cookie_label'] == 'Migrated Installs', 'score'].empty else None,
                    "User Count": group['score'].size,
                    "Fresh Install Count": (group['cookie_label'] == 'Fresh Installs').sum(),
                    "Migrated Install Count": (group['cookie_label'] == 'Migrated Installs').sum(),
                })
            ).reset_index()
        )
        version_stats.columns = [
            'App Version', 
            'Average Rating Score', 
            'Avg Fresh Score', 
            'Avg Migrated Score', 
            'User Count', 
            'Fresh Install Count', 
            'Migrated Install Count'
        ]
        version_stats = version_stats.sort_values(by='App Version', ascending=False)
        return version_stats
    last_4_months_stats = calculate_version_stats(last_4_months_df)
    st.write("App Version Statistics (Last 4 Months)")
    st.dataframe(last_4_months_stats)
    last_3_weeks_stats = calculate_version_stats(last_3_weeks_df)
    st.write("App Version Statistics (Last 3 Weeks)")
    st.dataframe(last_3_weeks_stats)
    
    filtered_df['date_ymd'] = pd.to_datetime(filtered_df['date_ymd'])
    filtered_df['week'] = filtered_df['date_ymd'] - pd.to_timedelta((filtered_df['date_ymd'].dt.dayofweek - 2) % 7, unit='d')
    weekly_version_stats = (
        filtered_df.groupby(['version_app', 'week']).apply(
            lambda group: pd.Series({
                "Weekly Average Rating Score": group['score'].mean()
            })
        ).reset_index()
    )
    weekly_pivot = weekly_version_stats.pivot(index='week', columns='version_app', values='Weekly Average Rating Score')
    st.write("Weekly Trend of Average Rating Score by App Version")
    st.line_chart(weekly_pivot)
    latest_weeks = weekly_version_stats['week'].drop_duplicates().nlargest(4)
    last_3_weeks_stats = weekly_version_stats[weekly_version_stats['week'].isin(latest_weeks)]
    weekly_pivot_last_3_weeks = last_3_weeks_stats.pivot(index='week', columns='version_app', values='Weekly Average Rating Score')
    st.write("Weekly Trend of Average Rating Score by App Version (Last 3 Weeks)")
    st.line_chart(weekly_pivot_last_3_weeks)
    filtered_df['date_ymd'] = pd.to_datetime(filtered_df['date_ymd'])
    filtered_df['week'] = filtered_df['date_ymd'] - pd.to_timedelta((filtered_df['date_ymd'].dt.dayofweek - 2) % 7, unit='d')
    filtered_df = filtered_df.sort_values(by=['week', 'version_app'])
    filtered_df['Cumulative Mean Rating'] = filtered_df.groupby('version_app')['score'].expanding().mean().reset_index(level=0, drop=True)
    weekly_version_stats = filtered_df.groupby(['week', 'version_app']).agg({'Cumulative Mean Rating': 'last'}).reset_index()
    weekly_pivot = weekly_version_stats.pivot(index='week', columns='version_app', values='Cumulative Mean Rating')
    st.write("Cumulative Weekly Trend of Average Rating Score by App Version")
    st.line_chart(weekly_pivot)
    latest_weeks = weekly_version_stats['week'].drop_duplicates().nlargest(4)
    last_4_weeks_stats = weekly_version_stats[weekly_version_stats['week'].isin(latest_weeks)]
    weekly_pivot_last_3_weeks = last_4_weeks_stats.pivot(index='week', columns='version_app', values='Cumulative Mean Rating')
    st.write("Cumulative Weekly Trend of Average Rating Score by App Version (Last 3 Weeks)")
    st.line_chart(weekly_pivot_last_3_weeks)


    select_all_version_app = st.checkbox("Select all App Versions", key="select_all_version_app_unique_key")
    unique_version_apps = sorted(filtered_df['version_app'].dropna().unique())
    if select_all_version_app:
        selected_version_apps = unique_version_apps
    else:
        selected_version_apps = st.multiselect(
            "Select App Versions", 
            options=unique_version_apps, 
            default=[], 
            key="filter_version_app_unique_key"
        )
    filtered_versions_df = (
        filtered_df[filtered_df['version_app'].isin(selected_version_apps)] 
        if selected_version_apps else filtered_df
    )
    filtered_versions_df['date_ymd'] = pd.to_datetime(filtered_versions_df['date_ymd'])
    filtered_versions_df['week'] = filtered_versions_df['date_ymd'] - pd.to_timedelta((filtered_versions_df['date_ymd'].dt.dayofweek - 2) % 7, unit='d')
    weekly_version_stats = (
        filtered_versions_df.groupby(['version_app', 'week'])
        .agg({'score': 'mean'})
        .rename(columns={'score': 'Weekly Average Rating Score'})
        .reset_index()
    )
    weekly_pivot = weekly_version_stats.pivot(index='week', columns='version_app', values='Weekly Average Rating Score')
    st.write("Weekly Trend of Average Rating Score by App Version")
    st.line_chart(weekly_pivot)
    latest_weeks = weekly_version_stats['week'].drop_duplicates().nlargest(4)
    last_3_weeks_stats = weekly_version_stats[weekly_version_stats['week'].isin(latest_weeks)]
    weekly_pivot_last_3_weeks = last_3_weeks_stats.pivot(index='week', columns='version_app', values='Weekly Average Rating Score')
    st.write("Weekly Trend of Average Rating Score by App Version (Last 4 Weeks)")
    st.line_chart(weekly_pivot_last_3_weeks)
    filtered_versions_df = filtered_versions_df.sort_values(by=['week', 'version_app'])
    filtered_versions_df['Cumulative Mean Rating'] = filtered_versions_df.groupby('version_app')['score'].expanding().mean().reset_index(level=0, drop=True)
    cumulative_version_stats = (
        filtered_versions_df.groupby(['week', 'version_app'])
        .agg({'Cumulative Mean Rating': 'last'})
        .reset_index()
    )
    cumulative_pivot = cumulative_version_stats.pivot(index='week', columns='version_app', values='Cumulative Mean Rating')
    st.write("Cumulative Weekly Trend of Average Rating Score by App Version")
    st.line_chart(cumulative_pivot)
    latest_cumulative_weeks = cumulative_version_stats['week'].drop_duplicates().nlargest(4)
    last_3_weeks_cumulative_stats = cumulative_version_stats[cumulative_version_stats['week'].isin(latest_cumulative_weeks)]
    cumulative_pivot_last_3_weeks = last_3_weeks_cumulative_stats.pivot(index='week', columns='version_app', values='Cumulative Mean Rating')
    st.write("Cumulative Weekly Trend of Average Rating Score by App Version (Last 3 Weeks)")
    st.line_chart(cumulative_pivot_last_3_weeks)
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
    filtered_df['cookie_label'] = filtered_df['aiid'].apply(map_aiid_to_label)
    feedback_score_by_cookie = filtered_df.groupby(['cookie_label', 'score']).size().unstack().fillna(0)
    st.bar_chart(feedback_score_by_cookie) 

with tab1:
    st.subheader("Insights from Visualizations")
    visualizations = {
        "Average Rating Score and User Count by App Version": version_stats,
        "Number of Feedbacks Over Time": feedback_counts_by_date,
        "Feedback Scores Over Time": feedback_score_by_date,
        "Feedback Trends by Hour of the Day": feedback_by_hour,
        "Feedback Trends by Day of the Week": feedback_by_day,
        "Feedback Scores by Country": feedback_score_by_country,
        "Feedback Scores by Version App": feedback_score_by_version_app,
        "Feedback Scores by Cookie": feedback_score_by_cookie
    }
    installation_type_context = (
        "FRESH installs = new installation without data. This includes cases where the 'aiid' is Unknown or empty, or if "
        "aiid == 'mmm_prw_tst_007_498_c'. "
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
                model=st.session_state["azure_openai_model"],
                messages=[
                    {"role": "system", "content": "You are a data analyst and insight generator for Norton CyberSecurity, tasked with analyzing customer feedback data."},
                    {"role": "user", "content": visualization_prompt}
                ],
                temperature=0.5,
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
            max_tokens=4000,
            temperature=0.1,
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
