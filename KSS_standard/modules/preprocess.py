import yaml
import os
import numpy as np 
import pandas as pd
import re

def load_config(config_path="../config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)



def alias_info_dataframe(model_df, save_path=PREPROCESSED_DIR, save=False):
    """ 모델의 alias 정보를 분리하여 DataFrame을 변환하는 함수 """
    model_df["new_alias"]=model_df["modelAlias"].str.split(",")
    model_df = model_df.explode('new_alias', ignore_index=True)
    model_df.drop(columns=["modelAlias"], inplace=True)
    model_df.rename(columns={"new_alias":"modelAlias"}, inplace=True)
    model_df['modelAlias'] = model_df['modelAlias'].fillna(model_df['modelSTDName'])
    model_df["modelAlias"] = model_df["modelAlias"].str.strip()
    model_df.dropna(inplace=True)
    model_df = model_df.astype(str)
    if save:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "model_alias.csv")
        model_df.to_csv(save_file, index=False)
    
    return model_df



def clean_unknown(value):
    """
    모델명에서 "unknown" 처리 규칙을 적용한 정제 함수.
    
    규칙:
    - NaN, None, "na", "n/a" → np.nan
    - "unknown"만 존재하거나 변형 형태만 있으면 → np.nan
    - 문자열 중간에 "unknown" 계열이 포함된 경우 해당 부분만 제거
    """
    if pd.isna(value) or value is None:
        return np.nan

    value = str(value).strip().lower()

    # "na", "n/a" 처리
    if value in ["na", "n/a"]:
        return np.nan

    # "unknown"만 있는 경우 (변형 포함)
    if re.fullmatch(r"u?n?k?known(_\w+)?", value):
        return np.nan

    # 문자열 중간의 "unknown" 제거
    cleaned_value = re.sub(r"u?n?k?known(_\w+)?", "", value).strip()

    return cleaned_value if cleaned_value else np.nan




if __name__ == "__main__":
    config = load_config()

    RAW_DIR = config["raw_data"]["data_path"]
    PREPROCESSED_DIR = config["preprocessed_data"]["data_path"]

    tracker_info=pd.read_excel(os.path.join(RAW_DIR,"tacker_Info.xlsx"), header=1)
    model_df=pd.read_excel(os.path.join(RAW_DIR,"250224 model_STD.xlsx"))
    maker_df=pd.read_excel(os.path.join(RAW_DIR,"250224 maker_STD.xlsx"))
    alias_info_dataframe(model_df, save=True)