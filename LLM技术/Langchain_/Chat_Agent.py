from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# 初始化对话模型
llm = ChatOpenAI(temperature=0.2)

# 创建对话记忆（网页3的记忆模块）
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 定义领域工具（网页8的工具装饰器方式）
tools = [
    Tool(
        name="OrderCheck",
        func=lambda order_id: f"订单{order_id}状态：已发货",
        description="订单状态查询工具"
    )
]

# 构建ChatAgent（网页5的CHAT类型）
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 多轮对话示例
agent.run("请帮我查询订单123456的状态")
response = agent.run("运输公司是哪家？预计何时到达？")
print(response)