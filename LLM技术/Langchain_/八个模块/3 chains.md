LangChain 中的 Chain（链） 是构建复杂应用的核心模块，通过将多个组件（如模型、工具、数据处理步骤）按特定顺序组合，实现端到端的任务自动化。以下是 Chain 的主要类型、使用场景及代码示例：

---

## 一、基础链（Basic Chains）

1. **LLMChain**  
功能：直接与语言模型交互，处理单次输入输出任务。  
场景：文本生成、简单问答、翻译等。  
代码示例（网页1、网页5）：  
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# 初始化模型和提示模板
llm = OpenAI(temperature=0.7)
template = "用一句话解释{concept}的原理"
prompt = PromptTemplate(input_variables=["concept"], template=template)

# 创建链并执行
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"concept": "量子计算"})
print(result["text"])  # 输出量子计算的原理解释
```

2. **TextPromptChain**  
功能：结合预定义文本模板生成提示词。  
场景：标准化问答模板（如客服话术）。  
示例（网页2）：  
```python
from langchain.chains import TextPromptChain

template = "用户说：{input}。请回复："
chain = TextPromptChain.from_template(template, llm=llm)
print(chain.run("我想退货"))  # 生成标准化回复
```

---

## 二、组合链（Composite Chains）

1. **SimpleSequentialChain**  
功能：顺序执行多个单输入/单输出链。  
场景：多步骤任务（如生成→润色→翻译）。  
代码示例（网页1、网页5）：  
```python
from langchain.chains import SimpleSequentialChain

# 定义子链：生成菜品 → 生成菜谱
chain1 = LLMChain(...)  # 生成菜品名称
chain2 = LLMChain(...)  # 生成详细菜谱

# 组合顺序链
overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
result = overall_chain.run("川菜")  # 输出川菜名称及做法
```

2. **SequentialChain**  
功能：支持多输入/多输出的顺序链。  
场景：复杂文档处理（如摘要→关键词提取→分类）。  
示例（网页2、网页6）：  
```python
from langchain.chains import SequentialChain

chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["doc"],
    output_variables=["summary", "keywords"],
    verbose=True
)
```

---

## 三、条件链（Conditional Chains）

1. **RouterChain**  
功能：根据输入动态选择执行路径。  
场景：多分支任务（如根据问题类型路由到不同处理模块）。  
代码示例（网页3、网页6）：  
```python
from langchain.chains import RouterChain

# 定义路由逻辑：技术问题 → 技术链，售后问题 → 售后链
router_chain = RouterChain(
    route_mapper={"technical": tech_chain, "service": service_chain},
    default_chain=general_chain
)
result = router_chain.run("如何安装Python库？")  # 调用技术链
```

---

## 四、多步链（Multi-Step Chains）

1. **RetrievalQA**  
功能：结合检索与生成的问答链。  
场景：企业知识库问答。  
代码示例（网页6、网页8）：  
```python
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# 加载向量库并创建检索器
vector_db = FAISS.load_local("faiss_index")
retriever = vector_db.as_retriever()

# 构建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever,
    chain_type="stuff"
)
answer = qa_chain.run("LangChain的核心模块有哪些？")  # 返回检索增强的答案
```

2. **TransformChain**  
功能：自定义数据处理逻辑。  
场景：文本清洗、格式转换。  
示例（网页3、网页7）：  
```python
from langchain.chains import TransformChain

def clean_text(inputs):
    text = inputs["text"].strip().lower()
    return {"cleaned_text": text}

clean_chain = TransformChain(
    input_variables=["text"],
    output_variables=["cleaned_text"],
    transform=clean_text
)
```

---

## 五、自定义链（Custom Chains）

功能：灵活组合业务逻辑。  
场景：个性化任务（如数据清洗→模型调用→结果校验）。  
代码示例（网页5、网页7）：  

```python
from langchain.chains import Chain

class CustomChain(Chain):
    def __init__(self, chain1, chain2):
        self.chain1 = chain1
        self.chain2 = chain2
    
    def _call(self, inputs):
        res1 = self.chain1.run(inputs)
        final_res = self.chain2.run(res1)
        return {"result": final_res}

# 使用自定义链
custom_chain = CustomChain(chain1, chain2)
```

---

## 六、典型应用场景

| 场景           | 推荐链类型            | 技术要点                  |
| -------------- | --------------------- | ------------------------- |
| 客服对话       | `ConversationalChain` | 结合记忆模块维护上下文    |
| 文档摘要       | `MapReduceChain`      | 分块处理长文本后合并结果  |
| 数据分析       | `SQLDatabaseChain`    | 自然语言转SQL查询         |
| 自动化报告生成 | `SequentialChain`     | 多步骤生成+结构化输出解析 |

参数调优建议：  
• 启用 `verbose=True` 调试中间过程（网页1）  

• 使用 `temperature=0.3` 平衡生成结果稳定性与创造性  

• 通过 `max_retries=3` 增强链的容错性  


> 提示：更多示例可参考 [LangChain官方文档](https://python.langchain.com/docs/modules/chains/) 或搜索来源（如网页1、网页6）。