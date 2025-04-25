LangChain 的 Agents（智能代理） 模块是实现复杂任务自动化处理的核心组件，通过动态调用工具（Tools）扩展大语言模型（LLM）的能力。以下是其核心知识点及代码示例：

---

一、Agents 核心功能
1. **动态工具调用**
• 工具（Tools）：预定义或自定义的功能模块（如计算器、搜索引擎、代码执行器），用于处理 LLM 不擅长的任务（如数学计算、实时数据查询）。

• 代理类型：

  • `ZERO_SHOT_REACT_DESCRIPTION`：基础代理，根据工具描述动态选择工具。

  • `STRUCTURED_CHAT_ZERO_SHOT`：结构化对话代理，支持多轮上下文记忆。

• 决策流程：分析输入→选择工具→执行工具→组合结果→生成最终响应。


2. **核心组件**
• AgentExecutor：代理执行器，管理工具调用流程及迭代控制。

• Memory：记忆模块（如 `ConversationBufferMemory`），支持多轮对话上下文维护。


---

二、代码示例与场景应用
1. **使用内置工具**
场景：调用计算器与搜索引擎完成复杂查询：
```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# 初始化模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 加载内置工具（需安装：pip install wikipedia numexpr）
tools = load_tools(["llm-math", "wikipedia"], llm=llm)

# 创建代理
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 执行任务
response = agent.run("爱因斯坦的相对论公式是什么？该公式在量子力学中的意义是什么？")
print(response)
```

2. **自定义工具**
场景：创建数据库查询工具：
```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# 定义输入模型
class DBQueryInput(BaseModel):
    query: str = Field(description="SQL查询语句")

# 自定义工具
class DatabaseTool(BaseTool):
    name = "database_query"
    description = "执行SQL查询并返回结果"
    args_schema = DBQueryInput

    def _run(self, query: str) -> str:
        # 模拟数据库查询
        return f"查询结果：{query} 返回了3条记录"

# 使用自定义工具
tools = [DatabaseTool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
response = agent.run("查询用户表中2024年的注册用户数")
```

3. **多步骤任务处理**
场景：结合代码生成与执行：
```python
from langchain.agents import Tool
from langchain_experimental.tools import PythonREPLTool

# 加载Python代码执行工具
python_tool = PythonREPLTool()

# 定义代理
tools = [
    Tool(
        name="Python REPL",
        func=python_tool.run,
        description="执行Python代码并返回结果"
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# 执行排序任务
response = agent.run("用Python对列表 [5, 2, 9, 1] 进行升序排序并打印结果")
print(response)  # 输出：sorted_list = [1, 2, 5, 9]
```

---

三、高级功能与参数
1. ReAct 推理模式：通过 `create_react_agent` 实现思考-行动-观察循环：
   ```python
   from langchain.agents import create_react_agent
   react_agent = create_react_agent(llm, tools, prompt=react_prompt)
   ```

2. 错误处理：
   • `handle_parsing_errors=True` 自动重试解析失败的任务。

   • `max_iterations=5` 限制最大迭代次数防止死循环。


3. 记忆集成：维护对话历史：
   ```python
   from langchain.memory import ConversationBufferMemory
   memory = ConversationBufferMemory(memory_key="chat_history")
   agent = initialize_agent(..., memory=memory)
   ```

---

四、典型应用场景
1. 智能客服：自动调用知识库、订单系统工具处理用户咨询。
2. 数据分析：结合 SQL 工具查询数据库并生成可视化报告。
3. 自动化办公：集成邮件发送、日程管理工具实现任务自动化。

---

五、最佳实践建议
• 工具选择：优先使用内置工具（如 `llm-math`、`serpapi`），复杂场景再自定义。

• 性能优化：启用 `verbose=True` 调试执行流程，结合缓存减少 API 调用。

• 安全控制：限制代码执行工具的权限，避免恶意代码注入。


> 提示：更多示例可参考 [LangChain 官方文档](https://python.langchain.com/docs/modules/agents/) 或搜索来源（如网页1、网页3）。