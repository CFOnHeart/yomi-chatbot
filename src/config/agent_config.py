# LangGraph Agent 配置文件

# 数据库配置
DATABASE_PATH = "chat_history.db"

# 记忆管理配置
MAX_TOKENS = 3200  # 超过此阈值会触发摘要
SUMMARY_KEEP_RECENT = 2  # 摘要时保留最近的消息数量

# 工具配置
TOOL_CONFIDENCE_THRESHOLD = 0.7  # 工具执行的最低置信度阈值

# LLM配置
DEFAULT_SYSTEM_MESSAGE = """你是一个智能的AI助手，具有以下能力：
1. 记忆历史对话内容
2. 根据需要调用工具
3. 提供准确和有用的回答

请保持友好、专业和有帮助的态度。"""

# 工具提示模板
TOOL_DETECTION_PROMPT = """你是一个工具检测助手。分析用户的输入，判断是否需要调用工具。

可用工具：
{available_tools}

请分析用户输入，如果需要调用工具，请返回JSON格式：
{{
    "needs_tool": true,
    "tool_name": "工具名称",
    "confidence": 0.95,
    "reason": "需要调用工具的原因",
    "suggested_args": {{
        "arg1": "value1",
        "arg2": "value2"
    }}
}}

如果不需要调用工具，请返回：
{{
    "needs_tool": false,
    "confidence": 0.9,
    "reason": "不需要工具的原因"
}}

用户输入：{user_input}"""

# 摘要提示模板
SUMMARY_PROMPT = """你是一个对话摘要助手。请将以下对话历史进行摘要，保留重要信息和上下文。

摘要应该：
1. 保留关键信息和用户偏好
2. 简洁明了，避免冗余
3. 保持对话的连贯性
4. 长度控制在原文的30%以内

对话历史：
{history}

请提供摘要："""
