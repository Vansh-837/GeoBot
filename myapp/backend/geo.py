#Develop a geobot using langchain and openai that takes in the places.txt file
#And use recursive text splitter to make chunks
#And create embeddings from chunks in vector db in chromadb
#Then take in a querry and return the most similar chunk
#Finally again use open ai QnA model to rephrase and answer the question.




#=================importing libraries=======================


import openai
import chromadb
import langchain
import os


#=================OpenAI API Key=======================


from demo import OPENAI_API_KEY 
openai.api_key=OPENAI_API_KEY


#===================get a document=======================


from langchain_community.document_loaders import TextLoader  
loader = TextLoader("dataa/places.txt")
doc = loader.load() 


#===============split the document into chunks=======================


from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)
chunks = text_splitter.split_documents(doc)


#=================store embeddings in chromadb=======================


from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding_dir = "chromadb_dir"

texts = [chunk.page_content for chunk in chunks]
embedding_func = OpenAIEmbeddings(api_key=openai.api_key)
embeddings = embedding_func.embed_documents(texts)
dbb = Chroma.from_documents(chunks, embedding_func, persist_directory=embedding_dir)
#dbb.persist()
print(len(embeddings), "embeddings created and stored in ChromaDB")
    

#=================query the database=======================


#query = "What is the language of Kuala Lumpur?"
query="Where is Belem Tower located?"
#query="What is the population of Edinburgh?"
#query="Where is Arabic spoken?"


ans= dbb.similarity_search(query)
context_txt=ans[0].page_content


#============define the prompt template=======================


PROMPT_TEMPLATE = """ 
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {query}
"""


#========use this prompt to get answer using gpt 3.5 turbo  model=======================

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
promptt = prompt_template.format(context=context_txt, query=query)
print(promptt)
response=openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": promptt}],
    max_tokens=60)

print(response.choices[0].message.content)


#=================end of code=======================