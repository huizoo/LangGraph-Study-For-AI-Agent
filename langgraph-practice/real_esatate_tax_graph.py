# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str # 사용자 질문
    answer: str # 세율
    tax_base_equation: str # 과세표준 계산 수식
    tax_deduction: str # 공제액
    market_ratio: str # 공정시장가액비율
    tax_base: str # 과세표준 계산

graph_builder = StateGraph(AgentState)

# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = Chroma(
    embedding_function = embedding_function,
    collection_name = 'real_estate_tax',
    persist_directory='./real_estate_tax_collection',
)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# %%
query = '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?'

# %%
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
 
client = Client()

rag_prompt = client.pull_prompt('rlm/rag-prompt', dangerously_pull_public_prompt=True)
llm = ChatOpenAI(model='gpt-4o')
small_llm = ChatOpenAI(model='gpt-4o-mini')

# %%
tax_base_retrieval_chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

tax_base_equation_prompt = ChatPromptTemplate.from_messages([
    ('system', '아래 설명에서 과세표준 계산 방법을 수식으로만 추출하세요. 부연설명 없이 수식만 반환하세요.'),
    ('human', '{tax_base_equation_information}')
])

tax_base_equation_chain = (
    {'tax_base_equation_information': RunnablePassthrough()}
    | tax_base_equation_prompt
    | llm
    | StrOutputParser()
)

tax_base_chain = tax_base_retrieval_chain | tax_base_equation_chain

def get_tax_base_equation(state: AgentState):
    tax_base_equation_question = '주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요'
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)
    return {'tax_base_equation': tax_base_equation}

# %%

# {'tax_base_equation': '과세표준 = (주택 공시가격 합계 - 공제금액) × 공정시장가액비율'}

# %%
tax_deduction_chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

def get_tax_deduction(state: AgentState):
    tax_deduction_question = '주택에 대한 종합부동산세 계산시 공제금액을 알려주세요'
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return {'tax_deduction': tax_deduction}

# %%
# {'tax_deduction': '주택에 대한 종합부동산세 계산 시 공제금액은 상황에 따라 다릅니다. 1세대 1주택자의 경우 12억 원이 공제되며, 그 외의 경우 9억 원이 공제됩니다. 법인 또는 법인으로 보는 단체의 경우에는 6억 원이 공제됩니다.'}

# %%
from langchain_tavily import TavilySearch
from datetime import date

tavily_search_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

tax_market_ratio_prompt = ChatPromptTemplate.from_messages([
    ('system', f'아래 정보를 기반으로 공정시장 가액비율을 계산해주세요\n\nContext:\n{{context}}'),
    ('human', '{query}')
])

def get_market_ratio(state: AgentState):
    query = f'오늘 날짜:({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇%인가요?'
    context = tavily_search_tool.invoke(query)
    print(f'context == {context}')
    tax_market_ratio_chain = (
        tax_market_ratio_prompt
        | llm
        | StrOutputParser()
    )
    market_ratio = tax_market_ratio_chain.invoke({
        'context': context,
        'query': query,
    })
    return {'market_ratio': market_ratio}

# %%
# {'market_ratio': '2026년 현재 주택 공시가격에 대한 공정시장가액비율은 60%입니다.'}

# %%
from langchain_core.prompts import ChatPromptTemplate

tax_base_calculation_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',"""
주어진 내용을 기반으로 과세표준을 계산해주세요

과세표준 계산 공식: {tax_base_equation}
공제금액: {tax_deduction}
공정시장가액비율: {market_ratio}"""),
        ('human', '사용자 주택 공시가격 정보: {query}')
    ]
)

def calculate_tax_base(state: AgentState):
    tax_base_equation = state['tax_base_equation']
    tax_deduction = state['tax_deduction']
    market_ratio = state['market_ratio']
    query = state['query']
    tax_base_calculation_chain =(
        tax_base_calculation_prompt
        | llm
        | StrOutputParser()
    )
    tax_base = tax_base_calculation_chain.invoke({
        'tax_base_equation': tax_base_equation,
        'tax_deduction': tax_deduction,
        'market_ratio': market_ratio,
        'query': query,
    })
    print(f'tax_base == {tax_base}')
    return {'tax_base': tax_base}

# %%
initial_state = {
    'tax_base_equation': '과세표준 = (주택 공시가격 합계 - 공제금액) × 공정시장가액비율',
    'tax_deduction': '주택에 대한 종합부동산세 계산 시 공제금액은 상황에 따라 다릅니다. 1세대 1주택자의 경우 12억 원이 공제되며, 그 외의 경우 9억 원이 공제됩니다. 법인 또는 법인으로 보는 단체의 경우에는 6억 원이 공제됩니다.',
    'market_ratio': '2026년 현재 주택 공시가격에 대한 공정시장가액비율은 60%입니다.',
    'query': query
}


# %%
tax_rate_calculation_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요
     
종합부동산세 세율:{context}'''),
    ('human', '''과세표준과 사요자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요
     
과세표준: {tax_base}
주택 수:{query}''')
])


def calculate_tax_rate(state: AgentState):
    query = state['query']
    tax_base = state['tax_base']
    context = retriever.invoke(query)
    tax_rate_chain = (
        tax_rate_calculation_prompt
        | llm
        | StrOutputParser()
    )
    tax_rate = tax_rate_chain.invoke({
        'context': context,
        'tax_base': tax_base,
        'query': query,
    })
    print(f'tax_rate == {tax_rate}')
    return {'answer': tax_rate}

# %%
calculate_tax_base(initial_state)

# %%
tax_base_state = {'tax_base': '사용자가 소유한 주택의 공시가격 합계는 5억 원 + 10억 원 + 20억 원 = 35억 원입니다.\n\n이 경우 "1세대 1주택자"가 아닌 경우이므로 공제금액은 9억 원이 됩니다.\n\n따라서 과세표준을 계산하면 다음과 같습니다.\n\n과세표준 = (35억 원 - 9억 원) × 60%\n              = 26억 원 × 60%\n              = 15.6억 원\n\n이 과세표준에 따라 해당하는 종합부동산세를 계산하면 됩니다. 다만, 세율에 대한 정보는 주어지지 않았으므로 추가로 관련 세율표를 참조하여 실제 세금을 계산해야 합니다.', 'query': query}

# %%
calculate_tax_rate(tax_base_state)

# %%
graph_builder.add_node('get_tax_base_equation', get_tax_base_equation)
graph_builder.add_node('get_tax_deduction', get_tax_deduction)
graph_builder.add_node('get_market_ratio', get_market_ratio)
graph_builder.add_node('calculate_tax_base', calculate_tax_base)
graph_builder.add_node('calculate_tax_rate', calculate_tax_rate)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'get_tax_base_equation')
graph_builder.add_edge(START, 'get_tax_deduction')
graph_builder.add_edge(START, 'get_market_ratio')
graph_builder.add_edge('get_tax_base_equation', 'calculate_tax_base')
graph_builder.add_edge('get_tax_deduction', 'calculate_tax_base')
graph_builder.add_edge('get_market_ratio', 'calculate_tax_base')
graph_builder.add_edge('calculate_tax_base', 'calculate_tax_rate')
graph_builder.add_edge('calculate_tax_rate', END)

# %%
graph = graph_builder.compile()

