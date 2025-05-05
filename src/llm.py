import json
import requests
from openai import OpenAI  # 导入OpenAI库用于访问GPT模型
from logger import LOG  # 导入日志模块

class LLM:
    def __init__(self, config):
        """
        初始化 LLM 类，根据配置选择使用的模型（OpenAI 或 Ollama）。

        :param config: 配置对象，包含所有的模型配置参数。
        """
        self.config = config
        self.model = config.llm_model_type.lower()  # 获取模型类型并转换为小写
        print(self.model)
        if self.model == "openai":
            self.client = OpenAI()  # 创建OpenAI客户端实例
        elif self.model == "ollama":
            self.api_url = config.ollama_api_url  # 设置Ollama API的URL
        else:
            LOG.error(f"不支持的模型类型: {self.model}")
            raise ValueError(f"不支持的模型类型: {self.model}")  # 如果模型类型不支持，抛出错误

    def generate_report(self, system_prompt, user_content):
        """
        生成技术报告，包含输入验证和提示词优化逻辑
        
        :param system_prompt: 基础系统提示词
        :param user_content: 用户提交的原始内容
        :return: 结构化技术报告
        """
        # 稳定性增强措施
        required_sections = ["新增功能", "主要改进", "修复问题"]
        validation_note = "\n警告：检测到输入结构不完整，请补充[新增功能]、[主要改进]、[修复问题]中的缺失部分"
        
        # 补充稳定性提示词模板
        optimized_prompt = f"""{system_prompt}
        
    [报告规范]
    1. 必须包含的章节：新增功能、主要改进、修复问题
    2. 每个技术点需标注：
    - 所属模块（前端/后端/测试）
    - 关联版本号（如v2.3.1→v2.4.0）
    - 技术变更类型（功能/优化/修复）
    3. 不确定信息处理：
    !时间区间模糊时使用「版本待确认」标注
    !存在矛盾描述时保留原始提交记录
    4. 格式要求：
    → 日期格式ISO 8601（YYYY-MM-DD）
    → 技术名词首字母大写（如Azure Cosmos）
    → 禁止使用非标准符号（❌/✅等）

    [输入验证]
    {validation_note if not all(section in user_content for section in required_sections) else ""}
    """

        messages = [
            {"role": "system", "content": optimized_prompt},
            {"role": "user", "content": user_content},
        ]

        # 模型分发逻辑
        if self.model == "openai":
            return self._generate_report_openai(messages)
        elif self.model == "ollama":
            return self._generate_report_ollama(messages)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _generate_report_openai(self, messages):
        """
        使用 OpenAI GPT 模型生成报告。

        :param messages: 包含系统提示和用户内容的消息列表。
        :return: 生成的报告内容。
        """
        LOG.info(f"使用 OpenAI {self.config.openai_model_name} 模型生成报告。")
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model_name,  # 使用配置中的OpenAI模型名称
                messages=messages
            )
            LOG.debug("GPT 响应: {}", response)
            return response.choices[0].message.content  # 返回生成的报告内容
        except Exception as e:
            LOG.error(f"生成报告时发生错误：{e}")
            raise

    def _generate_report_ollama(self, messages):
        """
        使用 Ollama LLaMA 模型生成报告。

        :param messages: 包含系统提示和用户内容的消息列表。
        :return: 生成的报告内容。
        """
        LOG.info(f"使用 Ollama {self.config.ollama_model_name} 模型生成报告。")
        try:
            payload = {
                "model": self.config.ollama_model_name,  # 使用配置中的Ollama模型名称
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 0.7,
                "stream": False
            }

            response = requests.post(self.api_url, json=payload)  # 发送POST请求到Ollama API
            response_data = response.json()

            # 调试输出查看完整的响应结构
            LOG.debug("Ollama 响应: {}", response_data)

            # 直接从响应数据中获取 content
            message_content = response_data.get("message", {}).get("content", None)
            if message_content:
                return message_content  # 返回生成的报告内容
            else:
                LOG.error("无法从响应中提取报告内容。")
                raise ValueError("Ollama API 返回的响应结构无效")
        except Exception as e:
            LOG.error(f"生成报告时发生错误：{e}")
            raise

if __name__ == '__main__':
    from config import Config  # 导入配置管理类
    config = Config()
    llm = LLM(config)

    markdown_content="""
# Progress for langchain-ai/langchain (2024-08-20 to 2024-08-21)

## Issues Closed in the Last 1 Days
- partners/chroma: release 0.1.3 #25599
- docs: few-shot conceptual guide #25596
- docs: update examples in api ref #25589
"""

    # 示例：生成 GitHub 报告
    system_prompt = "Your specific system prompt for GitHub report generation"
    github_report = llm.generate_report(system_prompt, markdown_content)
    LOG.debug(github_report)
