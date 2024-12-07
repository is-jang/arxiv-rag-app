from preprocessing import load_document, split_text, get_embeddings, store_vectorstore
from langchain.prompts import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.base import RunnableMap
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
import torch
from transformers import pipeline


def get_llm():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    transformers_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        truncation=True,
        temperature=0.5,
    )
    return HuggingFacePipeline(pipeline=transformers_pipeline)


keyword = input("주제를 정해주세요 ex) Federated Learning: ")
papers = load_document(keyword)
documents = split_text(papers)
embeddings = get_embeddings()
vectorstore = store_vectorstore(documents, embeddings)

retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
"""당신은 대학원생을 위해 최신 논문들을 찾아주는 친절한 친구입니다. 당신의 임무는 주어진 논문들(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 논문들(context)을 사용하여 질문(question) 에 답하세요. 만약, 주어진 논문들(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

llm = get_llm()
# 모델을 LangChain LLM에 연결
rag_chain = RunnableMap({
    "context": retriever,
    "question": RunnablePassthrough()
    }) | prompt | llm | StrOutputParser()


while True:
    question = input("질문하세요 ex)찾은 논문들에서 privacy와 관련된 주제가 있어?, 종료[X]: ")
    if question == "X":
        break
    response = rag_chain.invoke(question)
    print("Answer:", response)
    print("=====================================================")

