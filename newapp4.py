import streamlit as st
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import openai

# AstraDB connection setup using st.secrets
def get_astradb_session():
    cloud_config = {
        'secure_connect_bundle': st.secrets["astra"]["secure_connect_bundle"]
    }
    auth_provider = PlainTextAuthProvider(st.secrets["astra"]["client_id"], st.secrets["astra"]["client_secret"])
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect(st.secrets["astra"]["keyspace"])
    return session

def infer_schema_from_df(df):
    dtype_map = {
        'int64': 'int',
        'float64': 'float',
        'object': 'text',
        'bool': 'boolean',
        'datetime64[ns]': 'timestamp'
    }
    schema = ", ".join([f"{col} {dtype_map[str(dtype)]}" for col, dtype in zip(df.columns, df.dtypes)])
    return schema

def create_table(session, table_name, schema):
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema}, PRIMARY KEY ({df.columns[0]}))"
    session.execute(query)

def insert_data(session, table_name, df):
    for _, row in df.iterrows():
        columns = ", ".join(row.index)
        values = ", ".join([f"'{str(val)}'" if isinstance(val, str) else str(val) for val in row.values])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
        session.execute(query)

# Function to normalize field names
def normalize_field_names(columns):
    return [col.lower().replace(" ", "_") for col in columns]

# Function to remove commas from numbers and convert to numeric types
def clean_numeric_fields(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').replace('nan', '')
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

# Function to mask selected fields
def mask_fields(df, columns_to_mask):
    for col in columns_to_mask:
        df[col] = df[col].apply(lambda x: '*' * len(str(x)))
    return df

# Function to handle NaN values
def handle_nan_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna('', inplace=True)
        else:
            df[col].fillna(0, inplace=True)
    return df

# Function to generate CQL from text using GPT-4
def generate_cql_from_text(prompt, table_name, schema):
    openai.api_key = st.secrets["openai"]["api_key"]
    schema_description = f"The table '{table_name}' has the following schema: {schema}."
    full_prompt = f"{schema_description} Convert the following text to a CQL query: {prompt}"
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a CQL query generator."},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Function to generate insights from text using GPT-4
def generate_insights_from_text(prompt, df):
    openai.api_key = st.secrets["openai"]["api_key"]
    data_summary = df.describe().to_csv(index=False)
    full_prompt = f"The following is the summary of the dataset:\n{data_summary}\n\n{prompt}"
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

st.title('CSV Loader to AstraDB with Field Normalization, Masking, Schema Inference, and Text-to-CQL')

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Normalize column names
    df.columns = normalize_field_names(df.columns)
    
    # Clean numeric fields
    df = clean_numeric_fields(df)
    
    # Handle NaN values
    df = handle_nan_values(df)
    
    st.write("Data Preview")
    st.dataframe(df)
    
    st.write("Select fields to include:")
    selected_columns = st.multiselect("Select columns to include", df.columns)
    
    st.write("Select fields to mask:")
    mask_columns = st.multiselect("Select columns to mask", df.columns)
    
    if selected_columns:
        df = df[selected_columns]
        st.write("You selected the following columns:")
        st.write(selected_columns)
        
    if mask_columns:
        st.write("You selected the following columns to mask:")
        st.write(mask_columns)
        
        # Mask the selected columns
        df = mask_fields(df, mask_columns)
        
        st.write("Data Preview with Masked Fields")
        st.dataframe(df)
    
    table_name = st.text_input("Enter Table Name")
    
    if st.button("Load Data into AstraDB"):
        if table_name:
            session = get_astradb_session()
            schema = infer_schema_from_df(df)
            create_table(session, table_name, schema)
            insert_data(session, table_name, df)
            st.success(f"Data loaded into AstraDB table `{table_name}` successfully.")

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Execute Query", "Visualize Data", "Generate Reports", "Generate Insights"])
    
    with tab1:
        st.write("### Text-to-CQL Query")
        user_query = st.text_input("Enter your query in natural language")
        
        if st.button("Generate CQL"):
            if user_query and table_name:
                session = get_astradb_session()
                schema = infer_schema_from_df(df)
                cql_query = generate_cql_from_text(user_query, table_name, schema)
                st.write("Generated CQL Query:")
                st.code(cql_query)
                
                try:
                    result = session.execute(cql_query)
                    result_df = pd.DataFrame(result.all())
                    st.write("Query Result:")
                    st.dataframe(result_df)
                except Exception as e:
                    st.error(f"Error executing CQL query: {e}")

    with tab2:
        st.write("### Visualize Data")
        
        if table_name:
            session = get_astradb_session()
            try:
                query = f"SELECT * FROM {table_name};"
                result = session.execute(query)
                df_all = pd.DataFrame(result.all())
                st.write("Complete Table Data:")
                st.dataframe(df_all)
                
                st.write("### Plotting")
                plot_type = st.selectbox("Select plot type", ["line_chart", "bar_chart", "area_chart"])
                column_to_plot = st.selectbox("Select column to plot", df_all.columns)
                
                if plot_type == "line_chart":
                    st.line_chart(df_all[[column_to_plot]])
                elif plot_type == "bar_chart":
                    st.bar_chart(df_all[[column_to_plot]])
                elif plot_type == "area_chart":
                    st.area_chart(df_all[[column_to_plot]])
                
            except Exception as e:
                st.error(f"Error retrieving table data: {e}")

    with tab3:
        st.write("### Generate Reports")
        
        if table_name:
            session = get_astradb_session()
            try:
                query = f"SELECT * FROM {table_name};"
                result = session.execute(query)
                df_all = pd.DataFrame(result.all())
                
                st.write("Report Generation")
                
                if st.button("Generate Summary Report"):
                    st.write("### Summary Report")
                    summary = df_all.describe()
                    st.write(summary)
                    st.download_button(
                        label="Download Summary as CSV",
                        data=summary.to_csv().encode('utf-8'),
                        file_name="summary_report.csv",
                        mime="text/csv"
                    )
                
                st.write("### Export Full Data")
                st.download_button(
                    label="Download Full Data as CSV",
                    data=df_all.to_csv().encode('utf-8'),
                    file_name="full_data.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error retrieving table data: {e}")

    with tab4:
        st.write("### Generate Insights")
        
        if table_name:
            session = get_astradb_session()
            try:
                query = f"SELECT * FROM {table_name};"
                result = session.execute(query)
                df_all = pd.DataFrame(result.all())
                
                st.write("Enter your insight query:")
                insight_query = st.text_input("Enter your query in natural language to generate insights")
                
                if st.button("Generate Insights"):
                    insights = generate_insights_from_text(insight_query, df_all)
                    st.write("Insights:")
                    st.write(insights)
                    
                    # Visualize insights as charts (example)
                    st.write("### Plotting Insights")
                    plot_type_insight = st.selectbox("Select plot type for insights", ["line_chart", "bar_chart", "area_chart"], key='insight_plot_type')
                    column_to_plot_insight = st.selectbox("Select column to plot for insights", df_all.columns, key='insight_column_to_plot')
                    
                    if plot_type_insight == "line_chart":
                        st.line_chart(df_all[[column_to_plot_insight]])
                    elif plot_type_insight == "bar_chart":
                        st.bar_chart(df_all[[column_to_plot_insight]])
                    elif plot_type_insight == "area_chart":
                        st.area_chart(df_all[[column_to_plot_insight]])
                
            except Exception as e:
                st.error(f"Error retrieving table data: {e}")