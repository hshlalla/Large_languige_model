from retriever import HybridRetriever
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from modules.preprocess import normalize_query
import gradio as gr
import pandas as pd
import os
import tempfile

# Milvus와 임베딩 모델 초기화
retriever = HybridRetriever(
    uri="http://172.17.0.1:19530",
    collection_name="milvus_equipment",
    dense_embedding_function=BGEM3EmbeddingFunction(use_fp16=False, device="cuda"),
)

# 검색 함수
def gradio_search(model, maker, category, mode="hybrid", top_k=5):
    model = normalize_query(model)
    maker = normalize_query(maker)
    category = normalize_query(category)
    query = f"model: {model} maker: {maker} category: {category}".strip()
    results = retriever.search(query, k=top_k, mode=mode)
    return "\n".join([f"모델명 : {r['original_index']} \n 점수 : {r['score']:.3f} \n 내용 : {r['content'].upper()}" for r in results])


def predict_and_export_excel(file):
    try:
        test_df = pd.read_excel(file)
    except Exception as e:
        error_df = pd.DataFrame([{"error": f"파일을 불러올 수 없습니다: {e}"}])
        temp_path = os.path.join(tempfile.gettempdir(), "error_output.xlsx")
        error_df.to_excel(temp_path, index=False)
        return temp_path

    required_columns = {"MODEL_NAME", "MAKER_NAME", "Category_Name"}
    if not required_columns.issubset(test_df.columns):
        error_df = pd.DataFrame([{"error": f"엑셀 파일에 다음 컬럼이 필요합니다: {required_columns}"}])
        temp_path = os.path.join(tempfile.gettempdir(), "error_output.xlsx")
        error_df.to_excel(temp_path, index=False)
        return temp_path

    test_df.fillna("", inplace=True)
    test_df["MODEL_NAME"] = test_df["MODEL_NAME"].apply(normalize_query)
    test_df["MAKER_NAME"] = test_df["MAKER_NAME"].apply(normalize_query)
    test_df["Category"] = test_df["Category_Name"].apply(normalize_query)

    output_rows = []

    for _, row in test_df.iterrows():
        query = f"model: {row['MODEL_NAME']} maker: {row['MAKER_NAME']} category: {row['Category']}".strip()

        try:
            results = retriever.search(query, k=5)
        except Exception as e:
            results = [{"original_index": f"검색 실패: {e}"}]

        output_row = {
            "MODEL_NAME": row["MODEL_NAME"],
            "MAKER_NAME": row["MAKER_NAME"],
            "Category": row["Category"],
        }

        for i in range(5):
            output_row[f"top{i+1}"] = results[i]["original_index"] if i < len(results) else ""

        output_rows.append(output_row)

    result_df = pd.DataFrame(output_rows)

    # 엑셀로 저장
    temp_path = os.path.join(tempfile.gettempdir(), "search_result.xlsx")
    result_df.to_excel(temp_path, index=False)
    return temp_path

# 텍스트 검색용 UI
tab1 = gr.Interface(
    fn=gradio_search,
    inputs=[
        gr.Textbox(label="모델명"),
        gr.Textbox(label="제조사"),
        gr.Textbox(label="카테고리"),
        gr.Radio(["dense", "sparse", "hybrid"], value="hybrid", label="검색 모드"),
        gr.Slider(1, 20, value=5, step=1, label="Top K")
    ],
    outputs=gr.Textbox(label="결과"),
    title="🔍 장비 검색",
    description="텍스트로 모델명을 입력하여 장비 검색"
)

# 엑셀 검색용 UI
tab2 = gr.Interface(
    fn=predict_and_export_excel,
    inputs=gr.File(type="filepath", label="엑셀 업로드 (MODEL_NAME, MAKER_NAME, Category 포함)"),
    outputs=gr.File(label="엑셀 다운로드"),
    title="📄 엑셀 기반 장비 매칭",
    description="엑셀에 포함된 모델명과 제조사, 카테고리를 기반으로 유사한 모델을 예측합니다."
)

# 탭으로 통합
gr.TabbedInterface([tab1, tab2], ["텍스트 검색", "엑셀 업로드"]).launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)


