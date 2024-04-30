import os
import delta
import pyspark

from openai import OpenAI

from pyspark.sql import SparkSession
from pyspark.sql.functions import asc

BRONZE_LAYER = "/Users/benjaminroberts/Documents/Data and AI/spark_storage/bronze/"
SILVER_LAYER = "/Users/benjaminroberts/Documents/Data and AI/spark_storage/silver/"
GOLD_LAYER = "/Users/benjaminroberts/Documents/Data and AI/spark_storage/gold/"

DATA_ARCHITECTURE_CONTEXT = "My data architecture is a bronze layer that contains ingested data, a silver layer containing cleansed data, and a gold layer with modelled data presented as views."

DATA_MODEL_PROMPT = "Using these tables and their fields describe a data model. Join all tables where possible. Call this model the Silver Layer Data Model. Include the names of all tables and and all table fields in the output."

DATA_MODEL_DOT_PROMPT = "Create a dot file describing the data model as an ER diagram. Include all tables and all table fields. The text in the nodes should be arranged as a vertical list. Make the node shapes rectangular with a grey background color. Edges should be yellow and only lines without arrows. Use a sans-serif font. Only respond with code as plain text without code block syntax around it."

STAR_SCHEMA_PROMPT = "Create a star schema as views in the gold layer from the Silver Layer Data Model. The star schema shoud have hashed surrogate keys to join the tables. The star schema will allow me to analyse "

STAR_SCHEMA_DOT_PROMPT = "Provide a dot file that visualises the star schema. Give it a title Gold layer Star Schema. Nodes are rectangular and coloured light gold, use a sans serif font, the node should contain the name of the table all fields from the table each on a separate line. Provide only the code without any description"

DOCUMENTATION_PROMPT = "Provide detailed technical documentation of the data architecture, the silver layer data model, the star schema model, and the data lineage. Format the output as HTML."

LINEAGE_DOT_PROMPT = "Produce a dot file to visualise the data lineage from the silver layer to the gold layer, presented left to right. Colour the nodes in the silver layer silver and in the gold layer light gold. Provide only the code without any description"


client = OpenAI()


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
        model="gpt-4",
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

def infer_data_model(database, spark, reponses) :
    tbls = spark.catalog.listTables(dbName=database)
    cols = ""
    tables = ""
    for t in tbls :
        tables += "Table =  " + t.name + "\n"
        cols += "\nTable =  " + t.name + "\n"
        cols += spark.sql(f"SHOW columns in default.{t.name}")._jdf.showString(20,20,False)

    r = get_response(DATA_ARCHITECTURE_CONTEXT + " " + cols + " " + DATA_MODEL_PROMPT, 0.5, reponses)

    return r
    
def create_data_model_diagram(previous, name) :
    r = get_response("Context is: " + previous + DATA_MODEL_DOT_PROMPT, 0.5, 1)[0]

    with open("ai_outputs/" + name + ".dot", "w") as dot_file:
        print(r, file=dot_file)
        
    os.system("dot -Tsvg -o ai_outputs/" + name + ".svg ai_outputs/" + name + ".dot")
    os.system("open ai_outputs/schema1.svg")
    os.system("rm ai_outputs/schema1.dot")


def natural_language_query(nl_qry, previous) :
    return get_response(nl_qry + "previous context is " + previous, 1)


def infer_star_schema(previous, responses) :
    return get_response(STAR_SCHEMA_PROMPT + "order numbers and sales by product, sub-product, customer, employee and date. Previous context is " + previous, responses)


def create_star_schema_diagram(previous) :
    r = get_response(STAR_SCHEMA_DOT_PROMPT + "previous context is " + previous, 1)
    
    out = ""
    
    for l in r.split('\n')[1:-1]:
        out += l + "\n"

    with open("star_schema.dot", "w") as dot_file:
        print(out, file=dot_file)
        
    os.system("dot -Tsvg -o star_schema.svg star_schema.dot")
    os.system("open star_schema.svg")
    os.system("rm star_schema.dot")

def create_documentation(previous) :
    r = get_response(DOCUMENTATION_PROMPT + "\n" + previous, 1)

    with open("documentation.html", "w") as doc_file:
        print(r, file=doc_file)
        
    os.system("open documentation.html")

    
def create_lineage_diagram(previous) :
    r = get_response(LINEAGE_DOT_PROMPT + "previous context is " + previous, 1)
    
    out = ""
    
    for l in r.split('\n')[1:-1]:
        out += l + "\n"

    with open("lineage.dot", "w") as dot_file:
        print(out, file=dot_file)
        
    os.system("dot -Tsvg -o lineage.svg lineage.dot")
    os.system("open lineage.svg")
    os.system("rm lineage.dot")

    
# def infer_schema_tables(tables) : ### Create this function later, will only look at specific tables

def main():
    spark = get_spark()
    os.system("clear")
    
    print("\n1. INFER DATA MODEL\n")
    
    data_model_responses = infer_data_model("default", spark, 2)

    with open("ai_outputs/data_model1.txt", "w") as doc_file:
        print(data_model_responses[0], file=doc_file)
    os.system("open ai_outputs/data_model1.txt")

    with open("ai_outputs/data_model1.txt", "w") as doc_file:
        print(data_model_responses[1], file=doc_file)
    os.system("open ai_outputs/data_model2.txt")
    
    # create_schema_diagram of response 1
    
    print("\n2, CREATING DATA MODEL DIAGRAMS\n")

    with open("ai_outputs/data_model1.txt", "r") as file:
        base_schema1 = file.read()
    
    schema_diagram1 = create_data_model_diagram(base_schema1, "data_model1")
    
    # create_schema_diagram of response 2

    with open("ai_outputs/data_model2.txt", "r") as file:
        base_schema2 = file.read()
    
    schema_diagram2 = create_data_model_diagram(base_schema2, "data_model2")
    

    # print("**************************************************************")
    # print("********** NATURAL LANGUAGE AD-HOC ANALYTIC QRY **************")
    # print("**************************************************************")
    # print("\n\n")

    # nl_qry = "Return a spark sql query that answers the total number of orders by product category and product subcategory."
    # print(nl_qry)
    # spark_sql_analytical_qry = natural_language_query(nl_qry, base_schema_response)
    # print(spark_sql_analytical_qry)

    # print("**************************************************************")
    # print("****************** CREATE STAR SCHEMA ************************")
    # print("**************************************************************")
    # print("\n\n")
    
    # star_schema_response = infer_star_schema(base_schema_response)
    
    # print(star_schema_response)

    # print("**************************************************************")
    # print("****************** VISUALISE STAR SCHEMA **********************")
    # print("**************************************************************")
    # print("\n\n")
    
    # create_star_schema_diagram(star_schema_response)
    
    # print("**************************************************************")
    # print("****************** DOCUMENT DATA PLATFORM ********************")
    # print("**************************************************************")
    # print("\n\n")

    # documentation_response = create_documentation(base_schema_response + "\n" + star_schema_response)
    
    # print(documentation_response)
    
    # print("**************************************************************")
    # print("******************** DATAWAREHOUSE LINEAGE *******************")
    # print("**************************************************************")
    # print("\n\n")

    # create_lineage_diagram(base_schema_response + "\n" + star_schema_response)
    
if __name__ == "__main__":
    main()




