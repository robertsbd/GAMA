import streamlit as st
import os
import delta
import pyspark

from openai import OpenAI

from pyspark.sql import SparkSession

client = OpenAI()

BRONZE_LAYER = "/Users/benjaminroberts/Documents/Data and AI/spark_storage/bronze/"
SILVER_LAYER = "/Users/benjaminroberts/Documents/Data and AI/spark_storage/silver/"
GOLD_LAYER = "/Users/benjaminroberts/Documents/Data and AI/spark_storage/gold/"

DATA_ARCHITECTURE_CONTEXT = "My data architecture is a bronze layer that contains ingested data, a silver layer containing cleansed data, and a ngold layer with modelled data presented as views."

DATA_MODEL_PROMPT = "Using these tables and their fields describe a data model. Join all tables where possible. Call this model the base_data_model. Provide an output ensuring to include the names of all tables, the entities in these tables, and list the relationships between the tables.."

DATA_MODEL_DOT_PROMPT = "Create a dot file describing the data model as an ER diagram. Include all tables and entities joining the tables. The text in the nodes should be arranged as a vertical list. Make the node shapes rectangular with a grey background color. Edges are lines without arrows. Use a sans-serif font. Only respond with code as plain text without code block syntax around it."

STAR_SCHEMA_PROMPT = "Create a star schema as SQL views from tables in the base_data_model. The star schema shoud have hashed surrogate keys to join the tables. Output a description and the SQL code to create the views. Call this output the gold_star_schema The star schema will allow me to analyse "

STAR_SCHEMA_DOT_PROMPT = "Provide a dot file that visualises the gold_star_schema. Nodes are rectangular and coloured light gold, use a sans serif font, the node should contain the name of the table all fields from the table each on a separate line. Only respond with code as plain text without code block syntax around it."

DOCUMENTATION_PROMPT = "Provide detailed technical documentation useful for data modellers and data engineers. "

LINEAGE_DOT_PROMPT = "Produce a dot file to visualise the data lineage from the base_data_model to the gold_star_schema, set rankdir to LR for the diagram. Colour the nodes in the silver layer grey and in the gold layer light gold. Only respond with code as plain text without code block syntax around it."

SILVER_GOLD_PIPELINES_PROMPT = "Create the pyspark code to update the tables in the gold_star_schema using delta loads and updating the changes based on changes to the tables in the base_data_model. Provide code for all the tables required."

CHAT_GPT_MODEL = "gpt-3.5-turbo"

def get_spark() -> SparkSession:
    builder = (
        pyspark.sql.SparkSession.builder.appName("TestApp")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.warehouse.dir", "spark-warehouse")
        .enableHiveSupport()
    )
    spark = delta.configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_response(prompt, temp, rep) :
    response = client.chat.completions.create(
        model= CHAT_GPT_MODEL,
        temperature=temp,
        n=rep,
        messages=[
            {"role": "system", "content": "You are a data modeller and data engineer"},
            {"role": "user", "content": prompt}
        ]
    )
    out = []
    for o in response.choices :
        out.append(o.message.content)    
    return out


@st.cache_resource
def database_tables(database, _spark) :
    return _spark.catalog.listTables(dbName=database)

@st.cache_resource
def infer_data_model(database, _spark) :
    tbls = _spark.catalog.listTables(dbName=database)
    cols = ""
    tables = ""
    for t in tbls :
        tables += "Table =  " + t.name + "\n"
        cols += "\nTable =  " + t.name + "\n"
        cols += _spark.sql(f"SHOW columns in default.{t.name}")._jdf.showString(20,20,False)

    r = get_response(DATA_ARCHITECTURE_CONTEXT + " " + cols + " " + DATA_MODEL_PROMPT, 0.5, 1)

    return r

@st.cache_data
def create_data_model_diagram(previous) :
    return get_response("Context is: " + previous + DATA_MODEL_DOT_PROMPT, 0.5, 1)

@st.cache_data
def infer_star_schema(user_spec, previous) :
    return get_response(STAR_SCHEMA_PROMPT + " " + user_spec + " Previous context is :" + previous, 0.5, 1)

@st.cache_data
def create_star_schema_diagram(previous) :
    return get_response(STAR_SCHEMA_DOT_PROMPT + " Context is: " + previous, 0.5, 1)

@st.cache_data
def create_documentation(previous) :
    return get_response(DOCUMENTATION_PROMPT + " Context is: " + previous, 0.5, 1)

@st.cache_data
def create_lineage_diagram(previous) :
    return get_response(LINEAGE_DOT_PROMPT + " Context is: " + previous, 0.5, 1)

@st.cache_data
def create_silver_to_gold_pipelines(previous) :
    return get_response(SILVER_GOLD_PIPELINES_PROMPT + " Context is: " + previous, 0.5, 1)

@st.cache_data
def natural_language_query(nl_qry, previous) :
    return get_response("Create a spark sql query to calculate  " + nl_qry + " from the following data model description" + previous + ". Return no chat only SQL code as plain text without code block syntax around it." , 0.5, 1)


def main():
    spark = get_spark()
        
    if 'analysis_tables' not in st.session_state :
        st.session_state['analysis_tables'] = []
        
    if 'inferred_data_model' not in st.session_state :
        st.session_state['inferred_data_model'] = ""

    if 'data_model_diagram' not in st.session_state :
        st.session_state['data_model_diagram'] = ""

    if 'inferred_star_schema' not in st.session_state :
        st.session_state['inferred_star_schema'] = ""

    if 'star_schema_diagram' not in st.session_state :
        st.session_state['star_schema_diagram'] = ""

    if 'documentation' not in st.session_state :
        st.session_state['documentation'] = ""

    if 'lineage' not in st.session_state :
        st.session_state['lineage'] = ""
        
    if 'nl_qry_code' not in st.session_state :
        st.session_state['nl_qry_code'] = ""

    if 'nl_qry_data' not in st.session_state :
        st.session_state['nl_qry_data'] = []
        
    if 'silver_gold_pipelines' not in st.session_state :
        st.session_state['silver_gold_pipelines'] = ""

                
    with st.sidebar:

        # st.subheader("Status of required information")

        # with st.container(height=160, border = True):
        #     st.write(len(st.session_state['analysis_tables']))
        #     if len(st.session_state['analysis_tables']) == 0:
        #         st.write(len(st.session_state['analysis_tables']))
        #         st.warning('No tables loaded', icon="⚠️")
        #     else:
        #         st.write(len(st.session_state['analysis_tables']))
        #         st.success("Tables loaded", icon="✅")

        #     if st.session_state['inferred_data_model'] != "":
        #         st.success("Data model created", icon="✅")
        #     else:
        #         st.warning('No data model created', icon="⚠️")

                
        st.subheader("Create models")

        if st.button("Get analysis tables") :
            st.session_state['analysis_tables'] = database_tables("default", spark)
    
        if st.button("Infer base data model") :
            inferred_data_model = infer_data_model("default", spark)[0]
            st.session_state['inferred_data_model'] = inferred_data_model
            st.session_state['data_model_diagram'] = create_data_model_diagram(inferred_data_model)[0]
    
        with st.form("Infer star schema"):
            st.write("Star schema for analytical reporting")
            star_schema_spec = st.text_input('Describe the star schema', 'A fact table summarising the number of orders with dimensions of customer, employee, order date, vendor, and product and subproduct combined.')
            submitted = st.form_submit_button("Build")
            if submitted:
                inferred_star_schema = infer_star_schema(star_schema_spec, st.session_state['inferred_data_model'])[0]
                st.session_state['inferred_star_schema'] = inferred_star_schema
                st.session_state['star_schema_diagram'] = create_star_schema_diagram(inferred_star_schema)[0]

        if st.button("Create pipelines") :
            st.session_state['silver_gold_pipelines'] = create_silver_to_gold_pipelines(st.session_state['inferred_data_model'] + "\n" + st.session_state['inferred_star_schema'])[0]
        
        if st.button("Create documenation") :
            st.session_state['documentation'] = create_documentation(st.session_state['inferred_data_model'] + "\n" + st.session_state['inferred_star_schema'])[0]
            st.session_state['lineage'] = create_lineage_diagram(st.session_state['inferred_data_model'] + "\n" + st.session_state['inferred_star_schema'] + st.session_state['silver_gold_pipelines'])[0]

        st.subheader("Natural language queries")

        with st.form("Ad-hoc"):
            qry = st.text_input('Enter your questions to query the base data model', 'Total number of orders by product category and product subcategory.')
            submitted = st.form_submit_button("Submit question")
            if submitted:
                st.session_state['nl_qry_code'] = natural_language_query(qry, st.session_state['inferred_data_model'])[0]
                st.session_state['nl_qry_data'] = spark.sql(st.session_state['nl_qry_code'])
        
        st.subheader("Reset model outputs")

        if st.button("Reset all", type="primary") :
            st.session_state['analysis_tables'] = []
            st.session_state['nl_qry_code'] = ""
            st.session_state['nl_qry_data'] = []
            st.session_state['inferred_data_model'] = ""
            st.session_state['data_model_diagram'] = ""
            st.session_state['inferred_star_schema'] = ""
            st.session_state['star_schema_diagram'] = ""
            st.session_state['documentation'] = ""
            st.session_state['lineage'] = ""
            st.session_state['silver_gold_pipelines'] = ""        
            database_tables.clear()
            infer_data_model.clear()
            create_data_model_diagram.clear()
            infer_star_schema.clear()
            create_star_schema_diagram.clear()
            create_documentation.clear()
            create_lineage_diagram.clear()
            natural_language_query.clear()
            create_silver_to_gold_pipelines.clear()

        if st.button("Reset analysis tables") :
            st.session_state['analysis_tables'] = []
            database_tables.clear()

        if st.button("Reset base datamodel") :
            st.session_state['inferred_data_model'] = ""
            st.session_state['data_model_diagram'] = ""
            infer_data_model.clear()
            create_data_model_diagram.clear()

        if st.button("Reset documenation") :
            create_documentation.clear()
            create_lineage_diagram.clear()

        if st.button("Reset pipelines") :
            st.session_state['silver_gold_pipelines'] = ""
            create_silver_to_gold_pipelines.clear()

    st.title('GAMA: Generative AI data modelling, engineering and analysis')
    st.write("Use the buttons to the left generate and reset model outputs. Outputs are stored in the tabs below.")

    with st.expander("About"):
        st.markdown('''
        This tool assits data modellers and data engineers by taking analytic requirements and generating data models and associated pipelines from structured data.

        The tool uses Generative AI and an Apache Spark data platform to:
        - Analyse collections of structured data and infer and automate the production of data models for reporting, analytics and other requirements.
        - Run natural language queries against the generated data models

        ## Feature backlog
        - Optimise AI inputs and outputs for cost reduction
        - Build data engineering pipelines in pyspark
        - Model selection from range of OpenAI models
        - Use Foundation models other than OpenAI
        - Allow users to submit existing data models
        - Enable model training
        - Use local LLMs
    
        ## Built using
        - apache-spark 3.5.1
        - python 3.10.14
        - delta-spark 3.1.0 (python library)
        - streamlit 1.33.0 (python libary)
        - OpenAI 1.23.6 (python library)
        '''
        )

        # Set it so that whatever task a user conducts the output of that is displayed
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Base tables","Base data model", "Natural language query", "Star schema model", "Data pipelines", "Documentation"])
        
    with tab1:
        st.dataframe(st.session_state['analysis_tables'])

    with tab2:
        st.write(st.session_state['inferred_data_model'])
        st.graphviz_chart(st.session_state['data_model_diagram'])            

    with tab3:
        st.code(st.session_state['nl_qry_code'], language = "sql", line_numbers = True)
        st.dataframe(st.session_state['nl_qry_data'])

    with tab4:
        st.write(st.session_state['inferred_star_schema'])
        st.graphviz_chart(st.session_state['star_schema_diagram'])            

    with tab5:
        st.write(st.session_state['silver_gold_pipelines'])
            
    with tab6:
        st.write(st.session_state['documentation'])
        st.subheader('Data lineage diagram')
        st.graphviz_chart(st.session_state['lineage'])
    
if __name__ == "__main__":
    main()





