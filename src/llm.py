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
        # GitHub项目迭代特征强化模板
        iteration_template = """
    [GitHub迭代规范]
    1. 必须包含的迭代要素：
    ► 功能开发（Feature commits）
    ► 问题修复（Issue fixes）
    ► 文档更新（Docs update）
    ► 测试验证（Test cases）
    ► 发布说明（Release notes）

    2. 技术特征标注：
    ⚙️ 关联Issue编号（如#192）
    ⚙️ 版本跨度标记（v1.2.0→v1.3.0）
    ⚙️ 变更分类标识：[Core]/[API]/[UI]
    ⚙️ CI/CD管道状态（成功/失败）

    3. 社区交互要素：
    👥 涉及贡献者数量
    👥 关联讨论帖（Discussion link）
    👥 PR审核周期（小时数）
    👥 Issue响应时效

    [异常处理规则]
    ! 缺失commit信息时标注「迭代记录不完整」
    ! 版本冲突时保留原始git记录
    ! 未关闭的Issue添加「待跟进」标记
    """

        optimized_prompt = f"""{system_prompt}
    {iteration_template}
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
