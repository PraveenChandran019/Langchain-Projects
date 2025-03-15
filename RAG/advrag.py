from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("attention.pdf")
docs = loader.load()
#print(docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20 )
split_docs = text_splitter.split_documents(docs)
#print(split_docs[0:2])

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
embed = OllamaEmbeddings(model = 'llama2')

db = FAISS.from_documents(split_docs, embed)
query = 'who is the author of this paper'
result1 = db.similarity_search(query)
#print(result1.page_content)

from langchain_ollama.llms import OllamaLLM
llm = OllamaLLM(model = 'llama2')

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """ Answer the following question based only on the provided context.
    think step by step before providing answer.
    I will tip you 1000 dollars if the user finds the answer helpful.
    <context>
    {context}
    </context>
    
    question : {input} """
)

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm,prompt)

retriever = db.as_retriever()

from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

result = retrieval_chain.invoke({"input" : "An attention function can be described as mapping a query" })
print(result['answer'])