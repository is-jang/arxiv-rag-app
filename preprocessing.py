import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def load_document(keyword: str) -> list[dict]:
    def parse_papers(results):
        papers = []
        for result in results:
            title_tag = result.find("p", class_="title")
            abstract_tag = result.find("span", class_="abstract-full")
            authors_tag = result.find("p", class_="authors")

            title = title_tag.get_text(separator=" ", strip=True) if title_tag else "No Title"
            abstract = abstract_tag.get_text(separator=" ", strip=True) if abstract_tag else "No Abstract"
            if abstract.endswith("△ Less"):
                abstract = abstract[:-7]

            authors = (
                ", ".join(a.get_text(separator=" ", strip=True) for a in authors_tag.find_all("a"))
                if authors_tag
                else "No Authors"
            )
            papers.append({"title": title, "abstract": abstract, "authors": authors})
        return papers

    query = keyword.replace(" ", "+")
    url = f"https://arxiv.org/search/?searchtype=all&query={query}&abstracts=show&size=50&order=-announced_date_first"
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, "lxml")
    results = soup.find_all("li", class_="arxiv-result")
    papers = parse_papers(results)
    return papers


def split_text(papers: list[dict]) -> list[Document]:
    documents = []

    for paper in papers:
        content = (
            f"Title: {paper['title']} Abstract: {paper['abstract']} Authors: {paper['authors']}\n"
        )
        document = Document(page_content=content, metadata={})
        documents.append(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(documents)
    return split_documents


def get_embeddings() -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings()
    return embeddings


def store_vectorstore(documents: list[str], embeddings: HuggingFaceEmbeddings) -> FAISS:
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore