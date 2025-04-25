LangChain的模型I/O模块是连接大语言模型（LLM）与应用逻辑的核心组件，其设计目标是标准化模型交互流程、提升开发效率。该模块由三个核心子模块构成，具体功能与实现方式如下：

---

## 一、提示词管理（Prompt Management）

1. 动态模板构建  
   • PromptTemplate：支持创建带变量的字符串模板，通过`.format()`或`.invoke()`动态填充参数。例如，`"请给我取一个{adjective}的{name}名字"`可生成不同风格的提示词。  

   • ChatPromptTemplate：专为对话场景设计，支持角色化提示（如`SystemMessage`、`HumanMessage`），允许定义多轮对话结构和上下文关联。


2. 复杂模板支持  
   • FewShotPromptTemplate：通过示例模板（如输入-输出对）增强模型理解能力，适用于需要参考案例的任务（如实体识别）。  

   • 文件加载：支持从YAML/JSON文件加载模板，实现提示词与代码逻辑分离，便于维护和版本控制。
   
   LangChain 是一个用于开发由语言模型驱动的应用程序的框架，提示词管理在其中起着至关重要的作用，它能够帮助开发者更高效地组织、定制和使用提示词，以实现不同的应用场景。以下是关于 LangChain 提示词管理的详细说明，以及多种场景下的代码示例和使用说明。
   
   ### 提示词管理的关键组件
   
   1. **PromptTemplate**
   
   PromptTemplate 是 LangChain 中用于创建提示模板的核心类。它允许你定义一个包含变量的提示模板，然后在运行时将具体的值填充到这些变量中。
   
   2. **FewShotPromptTemplate**
   
   FewShotPromptTemplate 用于创建带有少量示例的提示模板。在某些场景下，提供一些示例可以帮助语言模型更好地理解任务要求。
   
   #### 场景一：简单文本生成
   
   #### 代码示例
   
   ```
   from langchain.prompts import PromptTemplate
   from langchain.llms import OpenAI
   
   # 定义提示模板
   prompt = PromptTemplate(
       input_variables=["topic"],
       template="请为我生成一篇关于 {topic} 的简短介绍。"
   )
   
   # 初始化语言模型
   llm = OpenAI(openai_api_key="your_openai_api_key")
   
   # 填充变量并生成文本
   input_text = prompt.format(topic="人工智能")
   output = llm(input_text)
   print(output)
   ```
   
   #### 使用说明
   
   1. **定义提示模板**：使用 PromptTemplate 类定义一个包含变量 topic 的提示模板。
   
   1. **初始化语言模型**：使用 OpenAI 类初始化一个语言模型实例，并传入你的 OpenAI API 密钥。
   
   1. **填充变量**：使用 prompt.format() 方法将具体的值填充到提示模板的变量中。
   
   1. **生成文本**：将填充后的提示文本传递给语言模型，调用 llm() 方法生成文本。
   
   #### 场景二：带有少量示例的文本生成
   
   #### 代码示例
   
   ```
   from langchain.prompts import FewShotPromptTemplate, PromptTemplate
   from langchain.llms import OpenAI
   
   # 定义示例
   examples = [
       {
           "animal": "猫",
           "description": "猫是一种温顺可爱的动物，喜欢睡觉和玩耍。"
       },
       {
           "animal": "狗",
           "description": "狗是人类的好朋友，忠诚且活泼。"
       }
   ]
   
   # 定义示例提示模板
   example_prompt = PromptTemplate(
       input_variables=["animal", "description"],
       template="动物: {animal}\n描述: {description}"
   )
   
   # 定义最终提示模板
   prompt = FewShotPromptTemplate(
       examples=examples,
       example_prompt=example_prompt,
       prefix="请根据示例为以下动物生成描述。",
       suffix="动物: {animal}",
       input_variables=["animal"],
       example_separator="\n\n"
   )
   
   # 初始化语言模型
   llm = OpenAI(openai_api_key="your_openai_api_key")
   
   # 填充变量并生成文本
   input_text = prompt.format(animal="兔子")
   output = llm(input_text)
   print(output)
   ```
   
   #### 使用说明
   
   1. **定义示例**：创建一个包含多个示例的列表，每个示例是一个字典，包含输入和输出的键值对。
   
   1. **定义示例提示模板**：使用 PromptTemplate 类定义一个用于格式化示例的提示模板。
   
   1. **定义最终提示模板**：使用 FewShotPromptTemplate 类定义最终的提示模板，将示例、示例提示模板、前缀、后缀等信息传入。
   
   1. **初始化语言模型**：使用 OpenAI 类初始化一个语言模型实例，并传入你的 OpenAI API 密钥。
   
   1. **填充变量**：使用 prompt.format() 方法将具体的值填充到提示模板的变量中。
   
   1. **生成文本**：将填充后的提示文本传递给语言模型，调用 llm() 方法生成文本。
   
   #### 场景三：多步骤任务的提示词管理
   
   #### 代码示例
   
   ```
   from langchain.prompts import PromptTemplate
   from langchain.llms import OpenAI
   
   # 第一步：生成主题
   prompt_step1 = PromptTemplate(
       input_variables=["keyword"],
       template="请根据关键词 {keyword} 生成一个有趣的主题。"
   )
   
   # 第二步：根据主题生成故事
   prompt_step2 = PromptTemplate(
       input_variables=["topic"],
       template="请根据主题 {topic} 生成一个简短的故事。"
   )
   
   # 初始化语言模型
   llm = OpenAI(openai_api_key="your_openai_api_key")
   
   # 第一步：生成主题
   input_step1 = prompt_step1.format(keyword="太空旅行")
   topic = llm(input_step1)
   
   # 第二步：根据主题生成故事
   input_step2 = prompt_step2.format(topic=topic)
   story = llm(input_step2)
   
   print("生成的主题:", topic)
   print("生成的故事:", story)
   ```
   
   #### 使用说明
   
   1. **定义多个提示模板**：根据多步骤任务的需求，定义多个提示模板，每个模板对应一个步骤。
   
   1. **初始化语言模型**：使用 OpenAI 类初始化一个语言模型实例，并传入你的 OpenAI API 密钥。
   
   1. **执行多步骤任务**：按照步骤顺序，依次填充变量并调用语言模型生成结果，将上一步的结果作为下一步的输入。
   
   ### 总结
   
   通过使用 PromptTemplate 和 FewShotPromptTemplate 等组件，LangChain 提供了强大的提示词管理功能，能够帮助开发者轻松应对各种场景下的提示词定制和使用需求。在实际应用中，你可以根据具体任务的要求，灵活组合和调整提示模板，以获得更好的生成效果。


---

## 二、语言模型接口（Language Models）

1. 模型类型抽象  
   • LLM包装器：处理基础文本补全任务，如`OpenAI`的`text-davinci-003`，输入输出均为字符串。  

   • ChatModel包装器：优化多轮对话，输入为结构化消息列表（如`[SystemMessage, HumanMessage]`），输出为`AIMessage`对象，支持角色化交互。


2. 统一调用接口  
   • 通过`invoke()`或`generate()`方法标准化调用流程，兼容50+模型服务商（如OpenAI、DeepSeek、本地Ollama模型），开发者仅需调整初始化参数即可切换模型。
   
   LangChain 的语言模型接口是其核心组件之一，旨在标准化不同大语言模型（LLM）的调用方式，提升开发效率。以下是其核心接口分类及对应的代码示例：
   
   ---
   
   ### 一、基础语言模型接口（LLM）
   
   功能：处理文本输入→文本输出的基础模型（如 GPT-3）  
   特性：  
   • 支持文本补全、问答生成等单次交互任务  
   
   • 提供异步调用和批量生成优化  
   
   
   代码示例（调用 OpenAI 的 text-davinci-003 模型）：  
   ```python
   from langchain.llms import OpenAI
   
   # 初始化模型（国内用户可配置稳定访问端点）
   llm = OpenAI(
       model_name="text-davinci-003",
       openai_api_key="YOUR_KEY",
       base_url="https://yunwu.ai/v1"  # 国内代理
   )
   
   # 单次生成
   response = llm("量子计算机的原理是什么？")
   print(response)
   
   # 批量生成（支持并发）
   batch_responses = llm.generate(["写一首春天的诗", "解释相对论"])
   for resp in batch_responses.generations:
       print(resp[0].text)
   ```
   
   ---
   
   ### 二、聊天模型接口（ChatModel）
   
   功能：处理多轮对话的模型（如 ChatGPT），支持结构化消息输入  
   特性：  
   • 支持系统消息（`SystemMessage`）、用户消息（`HumanMessage`）和 AI 回复（`AIMessage`）  
   
   • 可维护上下文实现连贯对话  
   
   
   代码示例（使用 GPT-4 实现中英翻译）：  
   ```python
   from langchain.chat_models import ChatOpenAI
   from langchain.schema import SystemMessage, HumanMessage
   
   # 初始化聊天模型
   chat = ChatOpenAI(
       model="gpt-4",
       openai_api_key="YOUR_KEY",
       temperature=0.7
   )
   
   # 构建多轮对话
   messages = [
       SystemMessage(content="你是一名专业翻译，专注技术文档中英互译"),
       HumanMessage(content="The transformer architecture uses self-attention mechanisms.")
   ]
   
   # 调用模型
   response = chat(messages)
   print(response.content)  # 输出：Transformer架构使用自注意力机制。
   ```
   
   ---
   
   ### 三、嵌入模型接口（Embedding Models）
   
   功能：将文本转换为向量表示，用于语义检索  
   特性：  
   • 支持 OpenAI、Hugging Face 等嵌入模型  
   
   • 与向量数据库（如 FAISS）无缝集成  
   
   
   代码示例（生成文本向量并计算相似度）：  
   ```python
   from langchain.embeddings import OpenAIEmbeddings
   
   # 初始化嵌入模型
   embeddings = OpenAIEmbeddings(
       model="text-embedding-ada-002",
       openai_api_key="YOUR_KEY"
   )
   
   # 生成向量
   texts = ["机器学习", "深度学习"]
   vectors = embeddings.embed_documents(texts)
   
   # 计算余弦相似度
   from numpy import dot
   from numpy.linalg import norm
   
   cos_sim = dot(vectors[0], vectors[1])/(norm(vectors[0])*norm(vectors[1]))
   print(f"相似度: {cos_sim:.2f}")  # 输出：相似度: 0.89
   ```
   
   ---
   
   ### 四、高级功能
   
   1. 流式输出：通过 `stream=True` 参数实现实时响应  
      ```python
      for chunk in chat.stream(messages):
          print(chunk.content, end="", flush=True)
      ```
   
   2. 缓存优化：减少重复 API 调用成本  
      ```python
      from langchain.cache import SQLiteCache
      import langchain
      langchain.llm_cache = SQLiteCache(database_path=".cache.db")
      ```
   
   3. 异步调用：提升高并发场景性能  
      ```python
      async def async_call():
          return await llm.agenerate(["异步生成示例"])
      ```
   
   ---
   
   应用建议
   • 模型选择：简单任务用 `LLM`，对话场景用 `ChatModel`，检索场景用 `Embedding`  
   
   • 参数调优：通过 `temperature` 控制生成随机性，`max_tokens` 限制输出长度  
   
   • 错误处理：启用 `max_retries=3` 参数实现自动重试  
   
   
   > 提示：开发者可通过 `langchain-community` 包扩展支持更多模型（如 Claude、文心一言）。


---

## 三、输出解析器（Output Parsers）

1. 结构化转换  
   • 基础解析：将原始文本输出转换为JSON、列表等格式，例如使用`JsonOutputParser`解析模型响应的结构化数据。  

   • Pydantic模型集成：定义数据验证规则，自动提取并校验关键字段，适用于医疗报告生成等需严格格式的场景。


2. 复杂处理逻辑  
   • 多级解析链：支持链式解析操作，例如先提取关键段落再转换为表格格式，适用于科研论文摘要生成等任务。  

   • 错误重试机制：当模型输出不符合预期时，可自动触发提示词优化和重新生成，提升结果可靠性。


---

技术演进与行业应用
• 跨模型兼容性：通过抽象层屏蔽不同模型API差异（如OpenAI文本补全接口与智谱清言聊天接口），实现“一次编码，多模型适配”。  

• 医学场景案例：在医疗影像分析中，模型I/O模块可快速集成CT报告生成流程——提示词模板定义诊断要素，输出解析器提取病灶特征并生成结构化报告。


> 提示：开发者可通过`model = ChatOpenAI()`初始化模型，结合`PromptTemplate`和`JsonOutputParser`，3-5行代码即可完成从提示词构造到结构化输出的完整流程。

LangChain 的输出解析器是模型I/O模块的核心组件，负责将大语言模型的非结构化文本输出转换为程序可处理的结构化数据。以下分场景介绍常用解析器及代码示例：

---

#### 一、基础类型解析

1. 字符串解析器（StrOutputParser）
功能：将 `AIMessage` 类型转换为纯文本字符串  
场景：对话回复提取、文本摘要生成  
示例（网页1）：  
```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
response = model.invoke("量子计算机的原理是什么？")
print(output_parser.parse(response))  # 输出: "量子计算机基于量子比特..."
```

2. 列表解析器（CommaSeparatedListOutputParser）
功能：将逗号分隔的文本转为列表  
场景：关键词提取、多选项生成  
示例（网页6）：  
```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
prompt = "列出五种水果："
response = model.invoke(prompt)
print(parser.parse(response))  # 输出: ['苹果', '香蕉', '橙子', '葡萄', '草莓']
```

---

#### 二、结构化数据解析

1. 字段化解析（StructuredOutputParser）
功能：提取指定字段的键值对  
场景：商品描述生成、报告结构化  
示例（网页2）：  
```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

schemas = [
    ResponseSchema(name="description", description="鲜花描述文案"),
    ResponseSchema(name="reason", description="文案设计理由")
]
parser = StructuredOutputParser.from_response_schemas(schemas)

response = model.invoke("为玫瑰撰写促销文案")
print(parser.parse(response))  # 输出: {'description': '...', 'reason': '...'}
```

2. Pydantic 解析器（PydanticOutputParser）
功能：结合数据验证生成对象  
场景：医疗报告生成、金融数据校验  
示例（网页5）：  
```python
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class PatientReport(BaseModel):
    diagnosis: str = Field(description="诊断结论")
    severity: int = Field(description="严重程度(1-5)")

parser = PydanticOutputParser(pydantic_object=PatientReport)
response = model.invoke("CT显示肺部有3cm结节")
report = parser.parse(response)  # 自动验证数据类型
```

---

#### 三、复杂场景解析

1. 日期时间解析（DatetimeOutputParser）
功能：提取时间实体并标准化  
场景：事件时间线构建、日程管理  
示例（网页6）：  
```python
from langchain.output_parsers.datetime import DatetimeOutputParser

parser = DatetimeOutputParser()
response = model.invoke("新中国成立的日期是什么？")
print(parser.parse(response))  # 输出: 1949-10-01 00:00:00
```

2. 自动修复解析（Auto-FixingParser）
功能：自动修正格式错误  
场景：用户评论清洗、日志修复  
示例（网页7）：  
```python
from langchain.output_parsers import OutputFixingParser

malformed_json = "{'name': 'Alice', 'age': thirty}"
parser = OutputFixingParser.from_llm(parser=JsonOutputParser(), llm=model)
print(parser.parse(malformed_json))  # 修正为 {"name": "Alice", "age": 30}
```

3. 重试解析器（RetryWithErrorOutputParser）
功能：自动重试错误输出  
场景：金融数据生成、法律条款生成  
示例（网页7）：  
```python
from langchain.output_parsers import RetryWithErrorOutputParser

parser = RetryWithErrorOutputParser.from_llm(
    parser=PydanticOutputParser(PatientReport),
    llm=model,
    max_retries=3
)
response = parser.parse_with_prompt("患者血压偏高", prompt_template)
```

---

四、行业应用案例
1. 医疗领域  
   ```python
   # 解析CT报告生成结构化数据（网页5）
   class TumorReport(BaseModel):
       location: str = Field(description="肿瘤位置")
       size_mm: float = Field(description="肿瘤尺寸(mm)")
   parser = PydanticOutputParser(pydantic_object=TumorReport)
   ```

2. 电商领域  
   ```python
   # 商品评论情感分析（网页4）
   response = model.invoke("用户评论：物流快但包装差")
   parser.parse(response)  # 输出: {'logistics_score':4, 'packaging_score':2}
   ```

3. 金融领域  
   ```python
   # 财报数据提取（网页8）
   response = model.invoke("2024年Q1营收同比增长15%")
   parser.parse(response)  # 输出: {'year':2024, 'quarter':1, 'growth_rate':0.15}
   ```

---

参数配置建议
1. 错误处理：启用 `max_retries=3` 实现自动重试  
2. 格式控制：通过 `get_format_instructions()` 生成格式说明（网页5）  
3. 混合使用：组合多个解析器处理嵌套结构  

> 提示：开发者可通过 `langchain.output_parsers` 模块查看全部解析器类型，具体参数参考官方文档。



