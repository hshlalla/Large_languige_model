import json

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)


class HybridRetriever:
    def __init__(self, uri, collection_name="hybrid", dense_embedding_function=None):
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_function = dense_embedding_function
        self.use_reranker = True
        self.use_sparse = True
        self.client = MilvusClient(uri=uri)

    def build_collection(self):
        if isinstance(self.embedding_function.dim, dict):
            dense_dim = self.embedding_function.dim["dense"]
        else:
            dense_dim = self.embedding_function.dim

        tokenizer_params = {
            "tokenizer": "standard",
            "filter": [
                "lowercase",
                {
                    "type": "length",
                    "max": 200,
                },
                {"type": "stemmer", "language": "english"},
                {
                    "type": "stop",
                    "stop_words": [
                        "?",
                        "an",
                        "and",
                        "are",
                        "as",
                        "at",
                        "be",
                        "but",
                        "by",
                        "for",
                        "if",
                        "in",
                        "into",
                        "is",
                        "it",
                        "no",
                        "not",
                        "of",
                        "on",
                        "or",
                        "such",
                        "that",
                        "the",
                        "their",
                        "then",
                        "there",
                        "these",
                        "they",
                        "this",
                        "to",
                        "was",
                        "will",
                        "with",
                    ],
                },
            ],
        }

        schema = MilvusClient.create_schema()
        schema.add_field(
            field_name="pk",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=100,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            analyzer_params=tokenizer_params,
            enable_match=True,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim
        )
        # schema.add_field(
        #     field_name="original_uuid", datatype=DataType.VARCHAR, max_length=128
        # )
        # schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=64)
        # schema.add_field(
        #     field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64
        # ),
        schema.add_field(field_name="original_index", datatype=DataType.VARCHAR, max_length=100,)

        functions = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names="sparse_vector",
        )

        schema.add_function(functions)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_data(self, chunk, metadata):
        embedding = self.embedding_function([chunk])
        if isinstance(embedding, dict) and "dense" in embedding:
            dense_vec = embedding["dense"][0]
        else:
            dense_vec = embedding[0]
        self.client.insert(
            self.collection_name, {"dense_vector": dense_vec, **metadata}
        )

    def insert_data_batch(self, chunks, metadata_list):
        """
        chunks: List[str] - 임베딩할 텍스트 리스트
        metadata_list: List[Dict] - 각 텍스트에 대응하는 메타데이터 리스트
        """
        assert len(chunks) == len(metadata_list), "chunks와 metadata_list 길이가 다릅니다."

        # 배치 임베딩
        embeddings = self.embedding_function(chunks)

        # dense vector 추출
        if isinstance(embeddings, dict) and "dense" in embeddings:
            dense_vecs = embeddings["dense"]
        else:
            dense_vecs = embeddings  # 이미 dense만 있을 경우

        # Milvus insert용 리스트 생성
        entities = []
        for dense_vec, metadata in zip(dense_vecs, metadata_list):
            entity = {"dense_vector": dense_vec}
            entity.update(metadata)
            entities.append(entity)

        # insert
        self.client.insert(self.collection_name, entities)


    def search(self, query: str, k: int = 20, mode="hybrid"):

        output_fields = [
            "content",
            "original_index",
        ]
        if mode in ["dense", "hybrid"]:
            embedding = self.embedding_function([query])
            if isinstance(embedding, dict) and "dense" in embedding:
                dense_vec = embedding["dense"][0]
            else:
                dense_vec = embedding[0]

        if mode == "sparse":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse_vector",
                limit=k,
                output_fields=output_fields,
            )
        elif mode == "dense":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vec],
                anns_field="dense_vector",
                limit=k,
                output_fields=output_fields,
            )
        elif mode == "hybrid":
            full_text_search_params = {"metric_type": "BM25"}
            full_text_search_req = AnnSearchRequest(
                [query], "sparse_vector", full_text_search_params, limit=k
            )

            dense_search_params = {"metric_type": "IP"}
            dense_req = AnnSearchRequest(
                [dense_vec], "dense_vector", dense_search_params, limit=k
            )

            results = self.client.hybrid_search(
                self.collection_name,
                [full_text_search_req, dense_req],
                ranker=RRFRanker(),
                limit=k,
                output_fields=output_fields,
            )
        else:
            raise ValueError("Invalid mode")
        return [
            {
                "original_index": doc["entity"]["original_index"],
                "content": doc["entity"]["content"],
                "score": doc["distance"],
            }
            for doc in results[0]
        ]
