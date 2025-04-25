### LangChain Memory库常用类及函数解析

#### **一、核心记忆类**
1. **ConversationBufferMemory**  
   • **功能**：完整存储所有对话历史，适合短对话场景（<10轮）  
   • **特性**：  
     ```python
     memory = ConversationBufferMemory()
     memory.save_context({"input": "你好"}, {"output": "你好"})
     print(memory.load_memory_variables({}))
     # 输出完整对话历史
     ```
   • **适用场景**：需完整上下文的简单问答系统

2. **ConversationBufferWindowMemory**  
   • **功能**：滑动窗口记忆，仅保留最近的K轮对话  
   • **配置参数**：  
     ```python
     memory = ConversationBufferWindowMemory(k=3)  # 保留最近3轮对话
     ```
   • **优势**：避免长对话的内存爆炸问题

3. **ConversationTokenBufferMemory**  
   • **功能**：基于Token数量限制记忆长度  
   • **配置示例**：  
     ```python
     memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=500)
     ```
   • **特点**：动态裁剪早期对话，保留关键信息

4. **ConversationSummaryMemory**  
   • **功能**：生成对话摘要替代原始记录，适合长对话  
   • **实现原理**：  
     ```python
     memory = ConversationSummaryMemory(llm=llm)
     # 自动生成摘要："用户讨论了天气和行程安排"
     ```

#### **二、高级记忆类**
5. **ConversationEntityMemory**  
   • **功能**：提取并跟踪对话中的实体信息（如人名、地点）  
   • **使用示例**：  
     ```python
     memory = ConversationEntityMemory(entities=["用户偏好", "订单号"])
     ```

6. **ConversationKGMemory**  
   • **功能**：以知识图谱形式存储实体关系  
   • **应用场景**：复杂对话中的关系推理（如医疗诊断记录）

7. **VectorStoreRetrieverMemory**  
   • **功能**：将记忆存入向量数据库，支持语义检索  
   • **集成示例**：  
     ```python
     memory = VectorStoreRetrieverMemory(retriever=FAISS.as_retriever())
     ```

#### **三、组合记忆方案**
8. **CombinedMemory**  
   • **功能**：融合多种记忆类型  
   • **典型配置**：  
     ```python
     combined_memory = CombinedMemory(memories=[
         ConversationBufferWindowMemory(k=3),
         ConversationEntityMemory(entities=["产品型号"])
     ])
     ```

#### **四、关键函数及方法**
| 函数/方法                       | 功能说明                       | 来源文档 |
| ------------------------------- | ------------------------------ | -------- |
| `save_context(inputs, outputs)` | 保存当前对话到内存             |          |
| `load_memory_variables({})`     | 加载记忆内容到提示模板         |          |
| `clear()`                       | 清空所有记忆                   |          |
| `add_message(BaseMessage)`      | 添加结构化消息（Human/AI类型） |          |

#### **五、工程实践建议**
1. **性能优化**  
   • 长对话使用`ConversationSummaryBufferMemory`压缩历史（网页7）  
   • 高频对话启用异步存储避免阻塞：  
     ```python
     asyncio.create_task(async_save_memory(...))
     ```

2. **领域适配**  
   • 金融场景：组合`EntityMemory`与`KGMemory`跟踪交易实体  
   • 医疗场景：通过`VectorStoreRetrieverMemory`关联病历数据

3. **错误处理**  
   • 设置记忆存储异常重试机制：  
     ```python
     ToolExecutor(max_retries=3, retry_delay=2)
     ```

---

**引用说明**：  
: 短期记忆与长期记忆实现方案  
: 记忆类型选择策略与混合存储  
: 核心类运行流程及源码解析  
: Token控制与窗口记忆实现  
: 基础类结构与对话链集成  
: 多领域应用场景分析  
: 知识图谱与实体记忆实践