LangChain 的回调模块是监控和控制模型执行流程的核心工具，支持在模型生命周期的不同阶段注入自定义逻辑。以下是多场景下的使用说明及代码示例：

---

一、基础回调功能
1. **同步回调（日志追踪）**  
场景：实时打印模型执行状态（如启动/结束事件）。  
代码示例（网页1、网页6）：  
```python
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain

handler = StdOutCallbackHandler()
chain = LLMChain(..., callbacks=[handler])  # 构造函数注入回调
chain.invoke({"input": "示例输入"})  # 控制台自动输出执行日志
```

2. **异步回调（事件流处理）**  
场景：处理异步生成任务（如流式输出）。  
代码示例（网页4）：  
```python
from langchain_core.callbacks.manager import adispatch_custom_event

async def async_chain():
    await adispatch_custom_event("event_start", {"status": "running"})
    # 执行异步任务...
    await adispatch_custom_event("event_end", {"result": "success"})

async for event in async_chain.astream_events():  # 消费事件流
    print(f"事件类型: {event['name']}, 数据: {event['data']}")
```

---

二、生产级应用回调
1. **文件日志记录**  
场景：将运行日志持久化到文件，便于后期审计。  
代码示例（网页6）：  
```python
from langchain.callbacks import FileCallbackHandler
from loguru import logger

logfile = 'app.log'
logger.add(logfile)  
handler = FileCallbackHandler(logfile)

chain = LLMChain(..., callbacks=[handler], verbose=True)
chain.invoke({"input": "数据"})  # 日志写入app.log文件
```

2. **Token计数与成本监控**  
场景：统计API调用消耗的Token量。  
代码示例（网页6）：  
```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    llm.invoke("解释量子力学")  # 模型调用
    print(f"总消耗Token: {cb.total_tokens}")  # 输出: 总消耗Token: 150
```

---

三、高级回调技术
1. **构造函数注入 vs 运行时注入**  
• 构造函数注入（网页2、网页5）  

  ```python
  class LoggingHandler(BaseCallbackHandler): ...  # 定义回调逻辑
  llm = ChatAnthropic(callbacks=[LoggingHandler()])  # 对象级回调
  ```
  *特点*：仅作用于当前对象，不传播给子组件。

• 运行时注入（网页7、网页8）  

  ```python
  chain.invoke(inputs, config={"callbacks": [LoggingHandler()]})  # 全局回调
  ```
  *优势*：覆盖当前调用链的所有子组件（如模型、工具）。

2. **自定义事件调度**  
场景：实现业务特定的监控逻辑（如进度条更新）。  
代码示例（网页4）：  
```python
from langchain_core.runnables import RunnableLambda

@RunnableLambda
async def custom_task():
    await adispatch_custom_event("progress", {"percent": 30})
    # 业务逻辑...
    await adispatch_custom_event("progress", {"percent": 100})
```

---

四、典型问题解决方案
1. 回调未触发  
   • 检查回调处理器是否继承 `BaseCallbackHandler`（网页1）  

   • 确认 LangChain 版本 ≥0.2.15（网页4）  


2. 异步兼容性问题  
   ```python
   # Python 3.10以下需手动传播配置（网页4）
   chain.invoke(..., config=RunnableConfig(callbacks=[handler])) 
   ```

3. 敏感数据过滤  
   ```python
   def on_llm_end(self, response, **kwargs):
       if "密码" in response: 
           response = response.replace("密码", "***")  # 脱敏处理
   ```

---

五、最佳实践建议
1. 回调类型选择  
   • 简单调试 → `StdOutCallbackHandler`  

   • 生产环境 → `FileCallbackHandler` + 日志分析工具  


2. 性能优化  
   • 启用 `verbose=False` 关闭冗余日志（网页6）  

   • 使用 `AsyncCallbackHandler` 避免阻塞主线程（网页4）  


3. 错误隔离  
   ```python
   def on_tool_error(self, error, **kwargs):
       send_alert(f"工具执行失败: {str(error)}")  # 自定义告警
   ```

> 提示：完整代码参考 [LangChain官方文档](https://python.langchain.com/docs/modules/callbacks/) 或搜索来源（如网页2、网页6、网页7）。