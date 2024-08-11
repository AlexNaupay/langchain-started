import pandas
from langchain_community.document_loaders import DataFrameLoader
from pprint import pprint

data_frame = pandas.read_csv('repos_cairo.csv')
data_frame.head()

loader = DataFrameLoader(data_frame, page_content_column='repo_name')
data = loader.load()

print(f"El archivo es de tipo {type(data)} y tiene una longitud de {len(data)} debido a la cantidad de observaciones en el CSV.")

pprint(data[:5])
