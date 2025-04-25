LangChain 的 Memory 模块通过为语言模型提供上下文记忆支持，解决了传统对话系统中“金鱼记忆”的缺陷。以下是其核心功能、主要类型及多场景代码示例：

---

## 一、Memory 核心功能

1. 上下文连贯性：保持多轮对话的连贯性，避免信息割裂。
2. 动态记忆管理：支持按需存储、检索和更新关键信息。
3. 灵活存储形式：支持文本、摘要、向量等多种记忆存储方式。

---

## 二、常用 Memory 类型及代码示例

1. ### **对话缓冲记忆（ConversationBufferMemory）**

  功能：完整记录所有对话历史，适合简单短对话场景。  
  场景：客服问答、个性化助手（如记住用户偏好）。  
  代码示例：
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# 初始化记忆模块与模型
memory = ConversationBufferMemory()
llm = ChatOpenAI(temperature=0.7)
chain = ConversationChain(llm=llm, memory=memory, verbose=True)

# 模拟对话
chain.run("你好，我的生日是8月15日，喜欢科幻电影")
chain.run("你还记得我的生日吗？")  # 输出生日信息
```

2. ### **对话窗口记忆（ConversationBufferWindowMemory）**

  功能：仅保留最近 K 轮对话，防止内存溢出。  
  场景：高频对话场景（如实时客服）。  
  代码示例：
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3)  # 保留最近3轮对话
memory.save_context({"input": "如何提高代码质量？"}, {"output": "使用单元测试和代码审查"})
memory.save_context({"input": "单元测试工具有哪些？"}, {"output": "PyTest、JUnit"})
print(memory.load_memory_variables({}))  # 仅显示最后3轮
```

3. ### **Token限制记忆（ConversationTokenBufferMemory）**

  功能：根据 Token 数量动态裁剪历史记录。  
  场景：长文本处理（如文档分析）。  
  代码示例：
```python
from langchain.memory import ConversationTokenBufferMemory

# 限制最大Token数为50
memory = ConversationTokenBufferMemory(llm=ChatOpenAI(), max_token_limit=50)
memory.save_context({"input": "自我介绍：我是AI助手"}, {"output": "收到"})
memory.save_context({"input": "今天的天气适合出游"}, {"output": "是的，晴空万里"})
print(memory.buffer)  # 自动删除超限部分
```

4. ### **摘要记忆（ConversationSummaryMemory）**

  功能：用大模型总结历史对话，节省存储空间。  
  场景：长对话摘要（如会议记录）。  
  代码示例：
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=ChatOpenAI())
memory.save_context({"input": "项目需求：开发智能客服系统"}, {"output": "需集成NLP模块"})
memory.save_context({"input": "技术选型建议"}, {"output": "推荐使用LangChain框架"})
print(memory.buffer)  # 显示摘要后的历史
```

5. ### **向量数据库记忆（VectorStoreRetrieverMemory）**

  功能：通过向量检索获取相关记忆片段。  
  场景：复杂上下文问答（如企业知识库）。  
  代码示例：
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma

# 初始化向量库
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 存储并检索记忆
memory.save_context({"input": "LangChain的核心模块"}, {"output": "Model I/O、Chains、Memory"})
result = memory.load_memory_variables({"input": "LangChain有哪些组件？"})
print(result)  # 返回相似度最高的记忆
```

---

## 三、高级应用场景

1. ### **跨会话长期记忆**

  场景：用户偏好持久化（如电商推荐系统）  
  代码示例：
```python
# 结合SQL数据库存储长期记忆
from langchain.memory import SQLiteMemory

memory = SQLiteMemory(database_path="user_prefs.db")
memory.save_context({"user_id": 1001, "input": "喜欢黑色系服装"}, {"output": "已记录"})
prefs = memory.load_context({"user_id": 1001})  # 下次登录时读取
```

2. ### **多模态记忆融合**

  场景：结合文本与图像记忆（如医疗影像报告）  
  代码示例：
```python
# 存储CT影像文本描述
memory.save_context(
    {"image": "CT-20240422-001", "text": "右肺下叶3cm结节"}, 
    {"diagnosis": "良性肿瘤可能性大"}
)
```

---

## 四、参数调优建议

1. 容量控制：根据场景选择窗口大小（`k=5`）或 Token 限制（`max_token_limit=1000`）。
2. 性能优化：对高频访问的记忆启用缓存机制。
3. 安全策略：敏感信息（如密码）需在保存前脱敏处理。

> 更多示例可参考 LangChain 官方文档或搜索来源（如网页1、网页3、网页6）。