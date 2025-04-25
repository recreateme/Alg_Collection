from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool

# 初始化模型
llm = OpenAI(temperature=0, model_name="gpt-4-1106")

# 定义工具集（网页7工具定义方式）
tools = [
    Tool(
        name="Search",
        func=lambda q: "北京今日气温28℃，晴转多云",
        description="实时信息检索工具"
    ),
    Tool(
        name="Calculator",
        func=lambda expr: str(eval(expr)),
        description="数学计算工具"
    )
]

# 创建ZeroShotAgent（网页5的初始化方式）
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 执行多步骤任务
response = agent.run("查询北京当前气温，计算华氏温度是多少？")
print(response)