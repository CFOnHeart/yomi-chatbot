"""
Prompt模板管理工具
使用YAML配置文件和Jinja2模板引擎来管理prompt
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from jinja2 import StrictUndefined, Template


class PromptManager:
    """Prompt模板管理器"""
    
    def __init__(self):
        self.RAG_PROMPT_PATH = "src/config/prompts/prompts.yaml"
        self.SUPERVISOR__AGENT_PROMPT_PATH = "src/config/prompts/supervisor_agent_prompts.yaml"
        self._cache_templates = {}
    
    def _load_templates(self, prompt_path: str) -> Any:
        """加载YAML配置文件中的模板"""
        if self._cache_templates.keys().__contains__(prompt_path):
            return self._cache_templates[prompt_path]

        prompt_path_obj = Path(prompt_path)
        try:
            if not prompt_path_obj.exists():
                raise FileNotFoundError(f"Prompt配置文件不存在: {prompt_path_obj}")
            
            with open(prompt_path_obj, 'r', encoding='utf-8') as file:
                templates = yaml.safe_load(file)
            
            print(f"✅ 加载了 {len(templates)} 个prompt模板")
            self._cache_templates[prompt_path] = templates
            return templates
            
        except Exception as e:
            print(f"❌ 加载prompt模板失败: {e}")
            # 设置默认模板
            templates = {
                "structured_rag_prompt": "Error loading template. Question: {{ user_question }}",
                "error_response_prompt": "Error: {{ error_message }}",
                "fallback_response_prompt": "No relevant documents found for: {{ user_question }}"
            }
            return templates
    
    def populate_template(self, prompt_path: str, template_name: str, variables: Dict[str, Any]) -> str:
        """
        使用Jinja2渲染模板
        
        Args:
            template_name: 模板名称
            variables: 模板变量字典
            
        Returns:
            渲染后的字符串
        """
        templates = self._load_templates(prompt_path)
        if template_name not in templates:
            raise ValueError(f"模板 '{template_name}' 不存在")
        
        template_content = templates[template_name]
        compiled_template = Template(template_content, undefined=StrictUndefined)
        
        try:
            return compiled_template.render(**variables)
        except Exception as e:
            raise Exception(f"Jinja模板渲染错误: {type(e).__name__}: {e}")
    
    def get_structured_rag_prompt(self, user_question: str, documents: list) -> str:
        """获取结构化RAG prompt"""
        variables = {
            "user_question": user_question,
            "documents": documents
        }
        return self.populate_template(self.RAG_PROMPT_PATH, "structured_rag_prompt", variables)
    
    def get_error_response_prompt(self, error_message: str) -> str:
        """获取错误响应prompt"""
        variables = {"error_message": error_message}
        return self.populate_template(self.RAG_PROMPT_PATH, "error_response_prompt", variables)
    
    def get_fallback_response_prompt(self, user_question: str) -> str:
        """获取备用响应prompt"""
        variables = {"user_question": user_question}
        return self.populate_template(self.RAG_PROMPT_PATH, "fallback_response_prompt", variables)


# 创建全局实例
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """获取全局Prompt管理器实例"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
