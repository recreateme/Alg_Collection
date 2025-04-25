# -*- coding: utf-8 -*-
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# 初始化大模型（网页5的模型配置参考）
llm = ChatOpenAI(model="gpt-4", temperature=0)


# ================== CoT增强模块 ==================
# （网页9的链式推理设计）
def build_cot_chain():
    """构建包含思维链推理的流程"""

    # 问题拆解链（网页3的RAG预处理思路）
    problem_analysis_prompt = PromptTemplate(
        input_variables=["input"],
        template="""
        将用户问题拆分为可执行的子任务列表：
        问题：{input}

        # 输出要求（网页7的格式约束）
        按Markdown列表格式返回，每个子任务包含<任务类型>和<关键词>
        """
    )
    analysis_chain = LLMChain(llm=llm, prompt=problem_analysis_prompt, output_key="subtasks")

    # CoT推理链（网页9的LCEL语法）
    cot_reasoning_prompt = PromptTemplate(
        input_variables=["subtasks"],
        template="""
        执行分步推理：
        子任务列表：{subtasks}

        # 推理步骤（网页5的ReAct范式）
        1. 优先级排序（紧急度/重要度）
        2. 数据依赖关系分析
        3. 潜在冲突检测
        4. 执行顺序规划
        """
    )
    reasoning_chain = LLMChain(llm=llm, prompt=cot_reasoning_prompt, output_key="reasoning_steps")

    # 数据检索链（网页3的RAG实现）
    data_retrieval_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        根据推理结果检索数据：
        {query}

        # 检索规则（网页8的相似度阈值）
        - 使用FAISS向量库
        - 相似度阈值>0.7
        """
    )
    retrieval_chain = LLMChain(llm=llm, prompt=data_retrieval_prompt, output_key="data")

    # 结果生成链（网页7的代码执行逻辑）
    result_generator_prompt = PromptTemplate(
        input_variables=["data"],
        template="""
        整合数据生成最终回答：
        {data}

        # 输出要求（网页13的格式规范）
        - 包含数据来源引用
        - 关键结论用**加粗**标注
        """
    )
    generator_chain = LLMChain(llm=llm, prompt=result_generator_prompt, output_key="answer")

    # 自洽性校验链（网页9的验证层）
    validation_prompt = PromptTemplate(
        input_variables=["answer"],
        template="""
        验证答案逻辑一致性：
        {answer}

        # 检查项（网页12的错误传播控制）
        - 子任务覆盖率 ≥80%
        - 数据引用无矛盾
        - 结论有证据支撑
        """
    )
    validation_chain = LLMChain(llm=llm, prompt=validation_prompt, output_key="validation")

    return SequentialChain(
        chains=[analysis_chain, reasoning_chain, retrieval_chain,
                generator_chain, validation_chain],
        input_variables=["input"],
        output_variables=["answer", "validation"],
        verbose=True
    )


# ================== Agent集成 ==================
# （网页5的Agent架构）
master_chain = build_cot_chain()

tools = [
    Tool(
        name="CoT_Processor",
        func=master_chain.run,
        description="带思维链推理的复杂问题处理器"
    )
]

# 初始化Agent（网页13的链式调用规范）
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ================== 执行示例 ==================
if __name__ == "__main__":
    question = "分析2024年全球人工智能产业趋势"
    response = agent.run(question)
    print(f"\n最终答案：{response['answer']}")
    print(f"验证结果：{response['validation']}")