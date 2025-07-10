
# Equipment Hybrid Search API


- Milvus 벡터 DB, FastAPI, Gradio를 활용하여 장비 정보를 **dense + sparse 하이브리드 검색**할 수 있는 시스템입니다.  
- 데이터는 주기적으로 업데이트할수 있도록 insert 부분을 endpoint로 추가하였습니다. 파일은 엑셀 파일 기반이며, 전처리 및 임베딩 후 Milvus에 저장됩니다.


[원본 데이터] 
     ↓ normalize_text()
[정제된 텍스트]
     ↓ embedding_model.encode()
[임베딩 벡터]
     ↓ Milvus insert()

--- 검색 ---
[사용자 쿼리]
     ↓ normalize_text()
[정제 쿼리]
     ↓ embedding_model.encode()
[검색 벡터]
     ↓ Milvus.search()
[Top-K 결과 반환]





##  기능 개요

### 1. 엘셀 업로드 및 임벤딩 `/reload`

* 엑셀 파일 업로드 → 카테고리 필터링 + 정규화(normalize) → BGEM3 임벤드 → Milvus 저장

#### 요청 예시

```http
POST /reload
Form-data: file=[Excel File]
```

---

### 2. 검색 API `/search`

* 세 개의 텍스트 필드를 받아 정규화 후 커리 조합
* 검색 모드는 `dense`, `sparse`, `hybrid` 중 선택 가능

#### 요청 예시

```http
POST /search
Content-Type: application/json

{
  "modelAlias": "centura epi",
  "makerAlias": "amat",
  "category": "cvd",
  "k": 5,
  "mode": "hybrid"
}
```

---

### 3. Gradio 데모 UI

* 검색어 입력 → 검색 결과 확인
* 모드 선택 및 Top-K 조절 가능
* search_demo.py

---

## ⚙️ 설정파일 (`config.yaml`)

```yaml
raw_data:
  data_path: "../../data/"

preprocessed_data:
  data_path: "../data/processed/"

EQUIPMENTS_CATEGORY:
  - CVD
  - Etch
  - PVD
  - ...
```

> 장비 카테고리 필터링은 이 설정을 기준으로 작동합니다.

---

## 🧰 실행 방법

### 1. 의존 패키지 설치

```bash
pip install -r requirements.txt
```

또는 수동 설치:

```bash
pip install fastapi uvicorn[standard] gradio pymilvus pyyaml tqdm pandas
```

---

### 2. FastAPI 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 9090 --reload
```

---

### 3. 접속 주소

| 서비스                | 주소                                                       |
| ------------------ | -------------------------------------------------------- |
| FastAPI Swagger UI | [http://localhost:9090/docs](http://localhost:9090/docs) |
| Gradio 검색 데모       | search_demo.py 실행후 url 접속          |

> 내부맹일 경우 `http://192.168.x.x:9090/docs` 처럼 접속해야 합니다.

---

## 디렉토리 구조
- 프로젝트 안에는 raw data는 넣지말것. raw data는 DB 폴더에 보관하고 project 안에 data 폴더에는 precessed data로 보관할것
```
project/
├── main.py                  # FastAPI + Gradio 실행 진입점
├── loader.py                # 엘셀 전처리 및 Milvus 삽입
├── retriever.py             # 검색기 클래스
├── config.yaml              # 카테고리 및 경로 설정
├── notebook/                # 실험 데이터
├── modules/
│   ├── config_loader.py     # YAML 설정 로더
│   └── preprocess.py        # 전처리 함수 등
└── data/
     └── processed
         └── processed_tracker_info.xlsx  # 업데이트 데이터
```
---

## 사용 기술

* **FastAPI**: RESTful API 서버
* **Milvus 2.5**: 벡터 DB
* **pymilvus**: Milvus Python SDK
* **BGEM3EmbeddingFunction**: 벡터 임벤드 함수
* **Gradio**: 웹 UI 기반 검색 데모
* **Docker**: 환경 격리 및 배포


## 문의

> 답변자: [sun.hong@surplusglobal.com](mailto:sun.hong@surplusglobal.com)
```
