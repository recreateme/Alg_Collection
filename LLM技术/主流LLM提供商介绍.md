### 关键要点
- 当前主流的大语言模型提供商包括 OpenAI、Google、Meta、Microsoft、Anthropic、Cohere、Baidu、Alibaba、DeepSeek、Mistral AI 和 xAI。
- 研究表明，大多数提供商的模型是专有的，但一些（如 Meta 的 LLaMA、Alibaba 的 Qwen、DeepSeek 的模型）是开源的，具体取决于模型版本。
- 证据显示，xAI 的早期 Grok 模型（如 Grok-1）是开源的，但最新模型（如 Grok-3）似乎是专有的，存在一定争议。

---

### 提供商简介
以下是各主流提供商的简要介绍及其模型的开源状态：

- **OpenAI**：以 GPT 系列（如 GPT-4、GPT-4o）闻名，专注于对话和多模态 AI，模型为专有，通过 API 和订阅访问。
- **Google**：提供 Gemini、PaLM 和 BERT 等模型，其中 Gemma（2B、7B 参数）是开源的，其他为专有，可通过 Google Cloud 和 Vertex AI 访问。
- **Meta**：提供 LLaMA 模型（如 LLaMA 3.1，405B 参数），广泛用于研究和行业，是开源的，许可多种。
- **Microsoft**：通过 Azure OpenAI Service 和模型如 Phi（开源，如 Phi-3）和 Orca 参与，部分开源，部分通过合作伙伴关系专有。
- **Anthropic**：开发 Claude 模型（如 Claude 3.5 Sonnet），注重安全和宪法 AI，为专有，通过 API 访问。
- **Cohere**：提供企业 LLM 如 Command 和 Rerank，可定制，为专有，通过 API 访问。
- **Baidu**：提供 Ernie（如 Ernie 4.0），在中国市场流行，超过 4500 万用户，为专有。
- **Alibaba**：提供 Qwen 模型（如 Qwen2.5，最高 72B 参数），在 Apache 2.0 下开源，可通过 Alibaba Cloud 和 Hugging Face 访问。
- **DeepSeek**：提供推理型模型如 DeepSeek-R1（总计 671B 参数，37B 活跃），为开源，竞争数学和编码任务。
- **Mistral AI**：提供如 Mistral Large 2（123B 参数）模型，主要为专有，部分变体开源，通过 API 访问。
- **xAI**：开发 Grok 模型（如 Grok-3），早期版本如 Grok-1 为开源（Apache 2.0），但最新模型似乎为专有，可通过应用和 API 访问。

---

### 一个意想不到的细节
有趣的是，xAI 的 Grok-1 在 2024 年 3 月开源，但 2025 年的 Grok-3 似乎转为专有，这可能反映了商业策略的变化，特别是在与 OpenAI 等竞争对手的背景下。

---

---

### 调查笔记：当前主流大语言模型提供商及开源状态详解

大型语言模型（LLM）在 2025 年继续推动 AI 领域的快速发展，其提供商涵盖技术巨头、初创公司和研究机构。本报告详细介绍主流提供商，并分析其模型的开源状态，基于近期可靠来源如 TechTarget、Signity Solutions 和 Exploding Topics 的报道。

#### 提供商列表及开源状态
以下是主流提供商及其代表性模型的概述，整理为表格形式，便于比较：

| **提供商** | **代表性模型**                    | **开源状态**          | **访问方式**                     | **备注**                                                     |
| ---------- | --------------------------------- | --------------------- | -------------------------------- | ------------------------------------------------------------ |
| OpenAI     | GPT-4, GPT-4o, GPT-3.5            | 专有                  | API, 订阅                        | 对话和多模态 AI 领导者，免费通过第三方门户可用。             |
| Google     | Gemini, PaLM, BERT, Gemma         | 部分开源（Gemma）     | Google Cloud, Vertex AI, API     | Gemma（2B, 7B 参数）开源，其他专有。                         |
| Meta       | LLaMA 3.1 (405B), LLaMA 3.2       | 开源                  | 公开可用，Hugging Face           | 广泛用于研究，参数范围从 11B 到 405B，许可多样。             |
| Microsoft  | Phi-3, Orca, Azure OpenAI Service | 部分开源（Phi, Orca） | API, Azure 云                    | Phi 和 Orca 开源，其他通过合作伙伴专有。                     |
| Anthropic  | Claude 3.5 Sonnet                 | 专有                  | API                              | 注重安全和宪法 AI，最新模型性能优异。                        |
| Cohere     | Command, Rerank, Embed            | 专有                  | API                              | 企业级定制 LLM，不限于单一云提供商。                         |
| Baidu      | Ernie 4.0                         | 专有                  | 内部平台                         | 中国市场流行，超过 4500 万用户，传闻参数达 10 万亿。         |
| Alibaba    | Qwen2.5, QwQ-32B                  | 开源                  | Alibaba Cloud, Hugging Face, API | Apache 2.0 许可，支持 29 种语言，参数最高 72B。              |
| DeepSeek   | DeepSeek-R1, DeepSeek-V3          | 开源                  | API, 公开可用                    | 推理型模型，671B 总参数，37B 活跃，竞争数学和编码。          |
| Mistral AI | Mistral Large 2, Pixtral Large    | 部分开源              | API, 部分模型公开                | Mistral Large 2（123B 参数）专有，部分变体开源。             |
| xAI        | Grok-1, Grok-3                    | 部分开源              | API, 应用                        | Grok-1（314B 参数）开源（Apache 2.0），Grok-3 似乎专有，需进一步确认。 |

#### 详细分析
- **OpenAI**：以 GPT 系列闻名，如 GPT-4 和 GPT-4o，专注于对话和多模态任务，全部为专有，通过 API 和订阅模式访问。Signity Solutions 的报告（[Top 15 Large Language Models in 2025](https://www.signitysolutions.com/blog/top-large-language-models)）确认其专有性质，无开源计划。
- **Google**：提供多种模型，包括 Gemini 和 PaLM，专有，但 Gemma 系列（2B, 7B 参数）在 2024 年开源，Exploding Topics 列出其开源状态（[Best 39 Large Language Models in 2025](https://explodingtopics.com/blog/list-of-llms)）。其他模型如 BERT 通过 Vertex AI 访问，为专有。
- **Meta**：LLaMA 系列（如 LLaMA 3.1，405B 参数）是开源的，TechTarget 提到其在 [llama.com](https://llama.com) 可用，广泛用于研究和行业，参数范围从 11B 到 405B。
- **Microsoft**：通过 Azure OpenAI Service 提供专有模型，同时 Phi 和 Orca 为开源，TechTarget 确认 Phi-3 系列（3.82B, 41.9B, 4.15B 参数）开源，适合开发者。
- **Anthropic**：Claude 系列（如 Claude 3.5 Sonnet）为专有，专注于安全 AI，Signity Solutions 报告其通过 API 访问，无开源计划。
- **Cohere**：企业级 LLM 如 Command 和 Rerank，为专有，可定制，TechTarget 提到其不限于单一云提供商。
- **Baidu**：Ernie 4.0 在中国市场流行，超过 4500 万用户，为专有，TechTarget 提到传闻参数达 10 万亿，无开源迹象。
- **Alibaba**：Qwen 系列（如 Qwen2.5，最高 72B 参数）在 Apache 2.0 下开源，Shakudo 的报告（[Top 9 Large Language Models as of March 2025](https://www.shakudo.io/blog/top-9-large-language-models)）确认其通过 Alibaba Cloud 和 Hugging Face 可用。
- **DeepSeek**：推理型模型如 DeepSeek-R1（671B 总参数）为开源，Exploding Topics 列出其 API 和公开可用性，竞争数学和编码任务。
- **Mistral AI**：Mistral Large 2（123B 参数）为专有，部分变体开源，TechTarget 提到其通过 API 访问。
- **xAI**：Grok 系列早期版本如 Grok-1（314B 参数）在 2024 年 3 月开源（InfoQ 报道，[xAI Releases Grok as Open-Source](https://www.infoq.com/news/2024/03/xai-grok-ai/)），在 GitHub 和 Hugging Face 可用，但 2025 年的 Grok-3 似乎专有，Bloomberg 的报道（[Musk’s xAI Unveils Grok-3 AI Bot](https://www.bloomberg.com/news/articles/2025-02-18/musk-s-xai-debuts-grok-3-ai-bot-touting-benchmark-superiority)）未提及开源，ZDNET 文章（[If Musk wants AI for the world, why not open-source all the Grok models?](https://www.zdnet.com/article/if-musk-wants-ai-for-the-world-why-not-open-source-all-the-grok-models/)）暗示可能转为专有，存在争议。

#### 一个意想不到的细节
有趣的是，xAI 的策略似乎从开源（如 Grok-1）转向专有（如可能的 Grok-3），这可能反映了商业竞争的压力，尤其是在与 OpenAI、Google 和 Anthropic 的竞争中。ZDNET 的分析（[If Musk wants AI for the world, why not open-source all the Grok models?](https://www.zdnet.com/article/if-musk-wants-ai-for-the-world-why-not-open-source-all-the-grok-models/)）指出，这可能与行业趋势有关，即开源和专有模型的平衡。

#### 结论
主流 LLM 提供商中，开源模型如 Meta 的 LLaMA、Alibaba 的 Qwen 和 DeepSeek 的模型在研究和开发中扮演重要角色，而专有模型如 OpenAI 的 GPT 和 Anthropic 的 Claude 则在商业应用中占主导。xAI 的开源状态存在争议，Grok-1 开源但 Grok-3 可能专有，需关注未来动态。

---

### 关键引文
- [Top 15 Large Language Models in 2025](https://www.signitysolutions.com/blog/top-large-language-models)
- [25 of the best large language models in 2025](https://www.techtarget.com/whatis/feature/12-of-the-best-large-language-models)
- [Best 39 Large Language Models in 2025](https://explodingtopics.com/blog/list-of-llms)
- [xAI Releases Grok as Open-Source](https://www.infoq.com/news/2024/03/xai-grok-ai/)
- [Musk’s xAI Unveils Grok-3 AI Bot](https://www.bloomberg.com/news/articles/2025-02-18/musk-s-xai-debuts-grok-3-ai-bot-touting-benchmark-superiority)
- [If Musk wants AI for the world, why not open-source all the Grok models?](https://www.zdnet.com/article/if-musk-wants-ai-for-the-world-why-not-open-source-all-the-grok-models/)