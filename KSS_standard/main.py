""" Created on 2025. 05.09 
@author:Sun.hong
"""


from fastapi import FastAPI, UploadFile, File
from retriever import HybridRetriever
from loader import load_and_insert_excel
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from modules.preprocess import normalize_query  # 전처리 함수 import

sw_version = "Release 0.0.0.0"


app = FastAPI()

retriever = HybridRetriever(
    uri="http://172.17.0.1:19530",
    collection_name="milvus_equipment",
    dense_embedding_function=BGEM3EmbeddingFunction(use_fp16=False, device="cuda"),
)

@app.post("/reload")
async def reload_excel(file: UploadFile = File(...)):
    temp_path = f"./data/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    load_and_insert_excel(retriever, temp_path)
    return {"status": "reload complete"}

@app.post("/search")
def search(model: str, maker: str, category: str, k: int = 5, mode: str = "hybrid"):
    # 1. 전처리
    model = normalize_query(model)
    maker = normalize_query(maker)
    category = normalize_query(category)

    # 2. 쿼리 문자열 조합
    query = f"model: {model} maker: {maker} category: {category}".strip()

    # 3. 검색
    return {"results": retriever.search(query, k=k, mode=mode)}

@app.post("/version")
def get_version():
    return {"version": sw_version}