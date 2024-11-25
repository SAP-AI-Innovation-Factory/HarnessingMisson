
"""In this scenario, we show how we could use genaihub SDK to retrieve embedded response from SAP HANA Cloud Vector.
   We use two of the foundation models. Based on the user prompt, we use the "embedding model" to convert the user prompt
   to embedding. We use COSINE SIMILARITY to retrieve the closest embedding from the HANA DB. Then the corresponding
   text is retrieved along with the source file, actual text, and scoring.    
   Then we provide the retrieved text as input to the "falcon" model to determine the sentiment of the text.
    Please install the necessary packages 
    pip install generative-ai-hub-sdk || hdbcli || hana_ml || python-dotenv || shapely    
"""
# Load all necessary packages
from gen_ai_hub.proxy.native.openai import embeddings
from gen_ai_hub.proxy.native.openai import completions
from hdbcli import dbapi
import requests
import pandas as pd
import hana_ml.dataframe as dataframe
import os
from dotenv import load_dotenv
load_dotenv()
# Provide the credentials to connect to HANA Cloud DB
HANA_HOST = os.getenv('HANA_HOST_VECTOR')
HANA_USER = os.getenv('HANA_VECTOR_USER') 
HANA_PASSWD = os.getenv('HANA_VECTOR_PASS') 
# APIs
DEPLOYMENT_URL = os.getenv('AICORE_BASE_URL')  # API base endpoint
RESOURCE_GROUP = os.getenv('AICORE_RESOURCE_GROUP')  # Resource Group ID
TOKEN_URL = os.getenv('AICORE_AUTH_URL')  # URL to get the token
CLIENT_ID = os.getenv('AICORE_CLIENT_ID')  # Client ID for authentication
CLIENT_SECRET = os.getenv('AICORE_CLIENT_SECRET')  # Client Secret for authentication
# Function to get the access token
def get_access_token():
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    response = requests.post(TOKEN_URL, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Error getting token: {response.status_code} - {response.text}")
# Get the access token
    TOKEN = get_access_token()
# Establish connections
conn = dataframe.ConnectionContext(
    address= HANA_HOST,             
    port=443,
    user=HANA_USER,
    password=HANA_PASSWD,
    encrypt='true'
)
conn1 = dbapi.connect( 
    address=HANA_HOST,
    port=443, 
    user=HANA_USER,
    password=HANA_PASSWD
)
# Here is the User Prompt. Change it to query against the text that was ingested
prompt = "What are the reviews about Tacos?"
# Here we convert your input to a vector using the Azure ada embedding model
res = embeddings.create(input=prompt, model="text-embedding-ada-002")
query_vector = res.data[0].embedding
# We query the DB with the embedded response from the user prompt
sql = '''SELECT TOP {k}  "FILENAME","TEXT" , TO_NVARCHAR("VECTOR") AS VECTOR_STR ,"{metric}"("VECTOR", TO_REAL_VECTOR('{qv}')) as SCORING
                  FROM "VECTOR_DEMO"."REVIEWS_TARGET"  
                  ORDER BY "{metric}"("VECTOR", TO_REAL_VECTOR('{qv}')) {sort}'''.format(k=10, metric="COSINE_SIMILARITY", qv=query_vector, sort="DESC")
hdf = conn.sql(sql)
res = hdf.head(10).collect() 
# Collect the response from the previous SQL
# Configure headers for API
headers = {
    "AI-Resource-Group": RESOURCE_GROUP,
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}
# Now loop around every row to get filename, text, and data
if not res.empty:
    db_results = [(row['FILENAME'], row['TEXT'], row['SCORING']) for _, row in res.iterrows()]
    new_results = []
    for i in range(len(db_results)):
        if i < len(db_results):
            filename, text, scoring = db_results[i]
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": f"Provide sentiment in exactly one word for the following text: '{text}'"}
                ],
                "max_tokens": 60,
                "temperature": 0.0,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            # We are getting the sentiment for every text using the below prompt
            # In order to get the prompt we call the falcon model from aicore config
            sentiment_output = requests.post(f'{DEPLOYMENT_URL}/inference/deployments/de8a35e3dfb4e9f6/chat/completions?api-version=2023-05-15', headers=headers, json=payload)
            # Converting the response to JSON (a Python dictionary)
            response_json = sentiment_output.json()
            # Accessing the content of "choices"
            sentiment = response_json["choices"][0]["message"]["content"]
            # Printing the result
            new_tuple = (filename, text, scoring, sentiment)
            new_results.append(new_tuple)
# Transform the results to a DF and check the output                       
df_new_results = pd.DataFrame(new_results)
print(df_new_results)
