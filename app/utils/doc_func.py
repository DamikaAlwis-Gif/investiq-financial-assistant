from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# load documents from json files
def load_docs_from_json_files(file_path):

  loader = JSONLoader(
      file_path=file_path,
      jq_schema='.[]',
      text_content=False,
      content_key='content',
      metadata_func=metadata_func
  )
  docs = loader.load()
  return docs

# add metadata to the documents
def metadata_func(record, metadata):

    metadata["date"] = record.get("date")
    metadata["title"] = record.get("title")
    metadata["url"] = record.get("url")
    metadata["category"] = record.get("category")

    return metadata


# split the documents into chunks
def split_docs(docs):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)

  return splits


# create docs from web search results
def create_docs_from_search_results(search_results: dict):
  docs = []
  for item in search_results["organic"]:
    content = item["snippet"]
    metadata = {
        "title": item["title"],
        "link": item["link"],

    }
    doc = Document(
        page_content=content,
        metadata=metadata
    )
    docs.append(doc)
  return docs
