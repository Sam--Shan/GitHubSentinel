import gradio as gr  # 导入gradio库用于创建GUI

from config import Config  # 导入配置管理模块
from github_client import GitHubClient  # 导入用于GitHub API操作的客户端
from hacker_news_client import HackerNewsClient
from report_generator import ReportGenerator  # 导入报告生成器模块
from llm import LLM  # 导入可能用于处理语言模型的LLM类
from subscription_manager import SubscriptionManager  # 导入订阅管理器
from logger import LOG  # 导入日志记录器

# 创建各个组件的实例
config = Config()
github_client = GitHubClient(config.github_token)
hacker_news_client = HackerNewsClient() # 创建 Hacker News 客户端实例
subscription_manager = SubscriptionManager(config.subscriptions_file)

def generate_github_report(model_type, model_name, repo, days):
    config.llm_model_type = model_type

    if model_type == "openai":
        config.openai_model_name = model_name
    else:
        config.ollama_model_name = model_name

    llm = LLM(config)  # 创建语言模型实例
    report_generator = ReportGenerator(llm, config.report_types)  # 创建报告生成器实例

    # 定义一个函数，用于导出和生成指定时间范围内项目的进展报告
    raw_file_path = github_client.export_progress_by_date_range(repo, days)  # 导出原始数据文件路径
    report, report_file_path = report_generator.generate_github_report(raw_file_path)  # 生成并获取报告内容及文件路径

    return report, report_file_path  # 返回报告内容和报告文件路径

def generate_hn_hour_topic(model_type, model_name):
    config.llm_model_type = model_type

    if model_type == "openai":
        config.openai_model_name = model_name
    else:
        config.ollama_model_name = model_name

    llm = LLM(config)  # 创建语言模型实例
    report_generator = ReportGenerator(llm, config.report_types)  # 创建报告生成器实例

    markdown_file_path = hacker_news_client.export_top_stories()
    report, report_file_path = report_generator.generate_hn_topic_report(markdown_file_path)

    return report, report_file_path  # 返回报告内容和报告文件路径


# 定义一个回调函数，用于根据 Radio 组件的选择返回不同的 Dropdown 选项
def update_model_list(model_type):
    if model_type == "openai":
        return gr.Dropdown(choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], label="选择模型")
    elif model_type == "ollama":
        return gr.Dropdown(choices=["llama3.1", "gemma2:2b", "qwen2:7b"], label="选择模型")


# 创建 Gradio 界面（应用现代风格主题）
with gr.Blocks(
    title="GitHubSentinel",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        font=[gr.themes.GoogleFont("Open Sans")]
    )
) as demo:
    # 创建 GitHub 项目进展 Tab
    with gr.Tab("GitHub 项目进展"):
        gr.Markdown("## 📊 GitHub 项目进展分析")  # 更新为更生动的标题

        # 模型选择部分
        with gr.Row():
            with gr.Column(min_width=300):
                model_type = gr.Radio(
                    ["openai", "ollama"],
                    label="模型平台",
                    info="选择使用的AI模型平台",
                    interactive=True
                )
            with gr.Column(min_width=300):
                model_name = gr.Dropdown(
                    choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                    label="模型版本",
                    interactive=True
                )

        # 参数配置部分
        with gr.Row():
            with gr.Column(min_width=300):
                subscription_list = gr.Dropdown(
                    subscription_manager.list_subscriptions(),
                    label="订阅项目",
                    info="选择已订阅的GitHub仓库",
                    interactive=True
                )
            with gr.Column(min_width=300):
                days = gr.Slider(
                    value=2,
                    minimum=1,
                    maximum=7,
                    step=1,
                    label="分析周期（天）",
                    info="设置要分析的时间范围",
                    interactive=True
                )

        # 动态更新模型列表
        model_type.change(fn=update_model_list, inputs=model_type, outputs=model_name)

        # 操作按钮
        with gr.Row():
            btn_generate = gr.Button("🚀 生成分析报告", variant="primary", scale=2)
            btn_clear = gr.Button("🔄 清空输入", variant="secondary")

        # 输出展示
        with gr.Accordion("实时预览", open=True):
            markdown_output = gr.Markdown()
        with gr.Row():
            file_output = gr.File(label="下载完整报告", interactive=False)

        # 绑定事件
        btn_generate.click(
            generate_github_report,
            inputs=[model_type, model_name, subscription_list, days],
            outputs=[markdown_output, file_output]
        )
        btn_clear.click(lambda: [None, None], outputs=[markdown_output, file_output])

    # 创建 Hacker News 热点话题 Tab
    with gr.Tab("Hacker News 热点话题"):
        gr.Markdown("## 🔥 实时热点话题分析")

        # 模型选择部分
        with gr.Row():
            with gr.Column(min_width=300):
                hn_model_type = gr.Radio(
                    ["openai", "ollama"],
                    label="模型平台",
                    value="openai",
                    interactive=True
                )
            with gr.Column(min_width=300):
                hn_model_name = gr.Dropdown(
                    choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                    label="模型版本",
                    value="gpt-4o",
                    interactive=True
                )

        # 动态更新模型列表
        hn_model_type.change(fn=update_model_list, inputs=hn_model_type, outputs=hn_model_name)

        # 操作按钮
        with gr.Row():
            hn_btn_generate = gr.Button("🚀 生成热点分析", variant="primary")
            hn_btn_clear = gr.Button("🔄 清空输入", variant="secondary")

        # 输出展示
        with gr.Accordion("分析结果预览", open=True):
            hn_markdown_output = gr.Markdown()
        with gr.Row():
            hn_file_output = gr.File(label="下载完整报告", interactive=False)

        # 绑定事件
        hn_btn_generate.click(
            generate_hn_hour_topic,
            inputs=[hn_model_type, hn_model_name],
            outputs=[hn_markdown_output, hn_file_output]
        )
        hn_btn_clear.click(lambda: [None, None], outputs=[hn_markdown_output, hn_file_output])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        favicon_path="https://www.svgrepo.com/show/512459/github-142.svg"
    )