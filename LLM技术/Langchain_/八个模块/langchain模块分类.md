LangChain作为大语言模型（LLM）应用开发框架，其核心架构可分为7个主要模块，各模块职责如下：

---

一、模型I/O（Model I/O）
• 核心职责：标准化LLM输入输出，连接各类语言模型

  • 提示模板：动态生成上下文相关的提示词（如`ChatPromptTemplate`）

  • 模型调用：集成OpenAI、Anthropic等30+模型服务商接口

  • 输出解析：将模型输出转为JSON等结构化数据（如`JsonOutputParser`）


二、数据连接（Data Connection）
• 核心职责：整合外部数据源构建知识库

  • 文档加载：支持PDF/SQL/网页等格式（如`PyPDFLoader`）

  • 文本分割：按语义切分长文本（如`RecursiveCharacterTextSplitter`）

  • 向量检索：与Faiss/Pinecone等向量数据库对接实现语义搜索


三、链（Chains）
• 核心职责：组合多步骤任务流程

  • LLMChain：基础链，直接调用LLM

  • 顺序链：串联数据预处理→模型调用→后处理（如`SimpleSequentialChain`）

  • 路由链：根据输入动态选择执行路径（如`LLMRouterChain`）


四、代理（Agents）
• 核心职责：动态决策调用工具

  • 工具定义：封装API调用、数据库查询等操作（如`GoogleSearchTool`）

  • 决策引擎：通过LLM生成行动策略（如`initialize_agent`函数）

  • 执行控制：协调工具调用与状态管理（如`AgentExecutor`）


五、回调（Callbacks）
• 核心职责：全流程监控与调试

  • 日志追踪：记录模型调用参数及响应时间

  • 异常处理：捕获并标记低置信度输出

  • LangSmith集成：提供可视化调试面板和版本管理


六、记忆（Memory）
• 核心职责：维护对话/任务状态

  • 短期记忆：缓存最近N轮对话（如`ConversationBufferMemory`）

  • 长期记忆：持久化存储关键实体信息（如`EntityMemory`）


七、扩展框架
• LangChain-Core：基础抽象和表达式语言（如`Runnable`协议）

• LangServe：将链部署为REST API

• LangGraph：构建带状态的多角色工作流


---

技术演进：早期版本划分6大模块（模型/提示/索引/链/代理/记忆），2024年后迭代为当前7模块体系，新增回调系统并强化扩展框架。开发者可通过模块化组合快速构建智能问答、文档分析等应用，如结合LlamaIndex实现企业知识库系统。