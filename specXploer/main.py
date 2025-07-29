import os
from llm.openai_llm import OpenAILLM
from llm.perplexity_llm import PerplexityLLM

def load_prompt(path):
    with open(path, "r") as f:
        return f.read()

if __name__ == "__main__":
    # 환경변수에서 키 불러오기
    openai_key = os.getenv("OPENAI_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    # LLM 선택
    llm = OpenAILLM(openai_key)
    # llm = PerplexityLLM(perplexity_key)

    prompt = load_prompt("prompts/part_detail_prompt.txt").format(part_name="2N2222 transistor")
    result = llm.invoke(prompt)
    print(result)