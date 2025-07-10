import pandas as pd
from modules.preprocess import *
from modules.config_loader import load_config
from tqdm import tqdm

config=load_config()

def load_and_insert_excel(retriever, file_path, batch_size=1024):
    df = pd.read_excel(file_path)
    df.fillna("", inplace=True)

    equipment_list = config["EQUIPMENTS_CATEGORY"]

    df = df[df['Category_Name'].isin(equipment_list)]

    for col in ["MODEL_NAME", "MAKER_NAME", "Category_Name", "Process_NM"]:
        df[col] = df[col].apply(clean_unknown_to_empty).apply(normalize_query)

    df["Process_NM"] = df["Process_NM"].apply(
        lambda x: " ".join(dict.fromkeys(word.strip() 
            for part in x.split(",") for word in part.strip().split())) if isinstance(x, str) else x
    )

    retriever.client.drop_collection(retriever.collection_name)
    retriever.build_collection()

    chunks, metadata_list = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        content = f"model: {row['MODEL_NAME']} maker: {row['MAKER_NAME']} category: {row['Category_Name']}".strip()
        chunks.append(content)
        metadata_list.append({
            "original_index": str(row["MODEL_ID"]),
            "content": content,
        })

        if len(chunks) >= batch_size:
            retriever.insert_data_batch(chunks, metadata_list)
            chunks, metadata_list = [], []

    if chunks:
        retriever.insert_data_batch(chunks, metadata_list)

if __name__ == "__main__":
    pass