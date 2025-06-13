# Send and process a document with Amazon Nova on Amazon Bedrock.
import os

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

load_dotenv()
REPORT_FILE_PATH = os.getenv("REPORT_FILE")

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g. Amazon Nova Lite.
model_id = "amazon.nova-lite-v1:0"

# Load the document
with open(REPORT_FILE_PATH, "rb") as file:
    document_bytes = file.read()

prompt = """Eres un asistente que extrae datos del reportes vulcanológicos, en formato json 
(es muy importante que el resultado lo des en json) con las siguientes llaves:
- volcano_name, descripción: Nombre de volcán
- analysis_period, descripción: Periodo de análisis, fechas de la forma YYYY-MM-DD separado por una ','
- issued_at, descripción: Fecha de emisión en la forma YYYY-MM-DD
- alert_level, descripción: Nivel de alerta en colores: verde, amarillo, naranja o rojo
- summary, descripción: Resumen
- analysis, el texto en la sección análisis
- perspectives, el texto en la sección perspectivas
- recommendations, recomendaciones
"""

# Start a conversation with a user message and the document
conversation = [
    {
        "role": "user",
        "content": [
            {"text": prompt},
            {
                "document": {
                    # Available formats: html, md, pdf, doc/docx, xls/xlsx, csv, and txt
                    "format": "pdf",
                    "name": "boletin ubinas 2025-08",
                    "source": {"bytes": document_bytes},
                }
            },
        ],
    }
]

try:
    # Send the message to the model using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 500, "temperature": 0.3},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
