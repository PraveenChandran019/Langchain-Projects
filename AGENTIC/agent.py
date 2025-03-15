from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper)

#print(tool.name)

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs1 = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
split_docs1 = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200).split_documents(docs1) 

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(model = 'llama2:latest')
vectordb= FAISS.from_documents(split_docs1,embed)
retriever = vectordb.as_retriever()
#print(retriever)

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriever, "langsmith_search",
                      "Search for information about LangSmith. For any question about LangSmith, you must use this tool!")
#print(retriever_tool.name)

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

tools = [wiki,arxiv,retriever_tool]

from langchain_ollama import ChatOllama
llm = ChatOllama(
    model = 'llama2:latest',
    temperature = 0.2
)

from langchain import hub
prompt  = hub.pull('hwchase17/openai-tools-agent')
#print(prompt.messages)

from langchain.agents import AgentExecutor, create_tool_calling_agent
agent = create_tool_calling_agent(llm,tools,prompt)
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)
agent_executor.invoke(
    {
        "input": "tell me about langsmith"
    }
)