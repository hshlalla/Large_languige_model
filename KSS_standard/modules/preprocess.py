import os
import numpy as np 
import pandas as pd
import re

def to_half_width(text):
    return ''.join([chr(ord(char) - 0xFEE0) if 0xFF01 <= ord(char) <= 0xFF5E else char for char in text])

def normalize_query(text):
    # 숫자뿐만 아니라 모든 값을 문자열로 변환
    text = str(text)

    # 전각 → 반각 변환
    text = to_half_width(text)

    # 소문자 통일
    text = text.lower()

    # 하이픈 제거 → 공백으로 대체
    text = text.replace("-", " ")

    # # 문자-숫자 / 숫자-문자 분리 (예: abc123 -> abc 123)
    # text = re.sub(r"([a-z]+)(\d+)", r"\1 \2", text)
    # text = re.sub(r"(\d+)([a-z]+)", r"\1 \2", text)

    # +, / 주변 공백 제거
    text = re.sub(r"\s*\+\s*", "+", text)
    text = re.sub(r"\s*/\s*", "/", text)

    # 중복 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text



def alias_info_dataframe(model_df, explode_column="modelAlias", save_path=None, save=False):
    """모델의 alias 정보를 분리하여 DataFrame을 변환하는 함수"""

    model_df = model_df.copy()


    # 쉼표로 분리
    model_df["new_alias"] = model_df[explode_column].str.split(",")

    # explode
    model_df = model_df.explode("new_alias", ignore_index=True)
    model_df.drop(columns=[explode_column], inplace=True)
    model_df.rename(columns={"new_alias": explode_column}, inplace=True)

    # 정리
    model_df[explode_column] = model_df[explode_column].str.strip()
    model_df.dropna(inplace=True)
    model_df = model_df.astype(str)
    model_df.reset_index(drop=True, inplace=True)

    # 저장
    if save and save_path is not None:
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
        return ""

    value = str(value).strip().lower()

    # "na", "n/a" 처리
    if value in ["na", "n/a"]:
        return ""

    # "unknown"만 있는 경우 (변형 포함)
    if re.fullmatch(r"u?n?k?known(_\w+)?", value):
        return ""

    # 문자열 중간의 "unknown" 제거
    cleaned_value = re.sub(r"u?n?k?known(_\w+)?", "", value).strip()

    return cleaned_value if cleaned_value else np.nan


def clean_unknown_to_empty(value):
    "nan파일을 unknown 문자열로 반환하는 함수"
    result = clean_unknown(value)
    return "" if pd.isna(result) else result

if __name__ == "__main__":
    config = load_config()

    RAW_DIR = config["raw_data"]["data_path"]
    PREPROCESSED_DIR = config["preprocessed_data"]["data_path"]

    tracker_info=pd.read_excel(os.path.join(RAW_DIR,"tacker_Info.xlsx"), header=1)
    model_df=pd.read_excel(os.path.join(RAW_DIR,"250224 model_STD.xlsx"))
    maker_df=pd.read_excel(os.path.join(RAW_DIR,"250224 maker_STD.xlsx"))
    alias_info_dataframe(model_df, save=True)