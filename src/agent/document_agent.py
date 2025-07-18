"""
文档管理Agent，专门处理文档相关的操作。
"""
from typing import Dict, Any, Optional
from src.agent.base_agent import AbstractManagedAgent
from src.config.prompt_manager import get_prompt_manager
from src.config.settings import get_llm_model
from src.rag.rag_system import RAGSystem

class DocumentAgent(AbstractManagedAgent):
    """
    专门处理文档管理和检索的Agent。
    """
    
    def __init__(self):
        super().__init__(
            description="专门处理文档管理、上传、检索和分析的Agent。"
                       "适合处理文档上传、文档搜索、文档摘要、文档分析等任务。"
                       "擅长从大量文档中快速找到相关信息并进行深度分析。"
        )
        self.prompt_manager = get_prompt_manager()
        self.llm = get_llm_model()
        self.rag_system = RAGSystem()
    
    def invoke(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        处理文档相关的查询。
        
        Args:
            query (str): 用户的文档相关问题
            context (Optional[Dict[str, Any]]): 上下文信息
            
        Returns:
            str: 处理结果
        """
        print(f"📚 DocumentAgent received query: {query}")
        
        # 判断查询类型
        if any(keyword in query.lower() for keyword in ["上传", "添加", "导入", "upload", "add"]):
            return self._handle_document_upload(query, context)
        elif any(keyword in query.lower() for keyword in ["搜索", "查找", "检索", "search", "find"]):
            return self._handle_document_search(query, context)
        elif any(keyword in query.lower() for keyword in ["摘要", "总结", "summary", "summarize"]):
            return self._handle_document_summary(query, context)
        elif any(keyword in query.lower() for keyword in ["分析", "analyze", "analysis"]):
            return self._handle_document_analysis(query, context)
        else:
            # 默认进行文档搜索
            return self._handle_document_search(query, context)
    
    def _handle_document_upload(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """处理文档上传相关的请求"""
        print("📤 Handling document upload request...")
        
        # 这里可以返回上传指引或处理上传逻辑
        return ("我可以帮您上传和管理文档。请提供文档的文件路径或内容，"
                "我将为您添加到知识库中。支持的格式包括PDF、TXT、DOCX等。")
    
    def _handle_document_search(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """处理文档搜索请求"""
        print("🔍 Searching documents...")
        
        try:
            # 使用RAG系统搜索相关文档
            search_results = self.rag_system.search_relevant_documents(query, top_k=5)
            
            if not search_results:
                return "抱歉，没有找到与您的查询相关的文档。请尝试使用不同的关键词。"
            
            # 格式化搜索结果
            response = "📖 找到以下相关文档：\n\n"
            for i, doc in enumerate(search_results, 1):
                response += f"{i}. **{doc.get('title', '未知标题')}**\n"
                response += f"   相关度: {doc.get('similarity_score', 0):.3f}\n"
                response += f"   内容摘要: {doc.get('content', '')[:200]}...\n\n"
            
            return response
            
        except Exception as e:
            return f"搜索文档时发生错误：{str(e)}"
    
    def _handle_document_summary(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """处理文档摘要请求"""
        print("📝 Generating document summary...")
        
        # 首先搜索相关文档
        search_results = self.rag_system.search_relevant_documents(query, top_k=3)
        
        if not search_results:
            return "没有找到相关文档进行摘要。"
        
        # 构建摘要prompt
        documents_text = "\n".join([doc.get('content', '') for doc in search_results])
        
        summary_prompt = f"""
        请为以下文档内容生成一个简洁的摘要：

        {documents_text[:3000]}  # 限制长度避免token过多

        摘要要求：
        1. 突出主要观点和关键信息
        2. 保持逻辑清晰
        3. 长度控制在200-300字
        """
        
        try:
            summary = self.llm.invoke(summary_prompt)
            return f"📄 文档摘要：\n\n{summary}"
        except Exception as e:
            return f"生成摘要时发生错误：{str(e)}"
    
    def _handle_document_analysis(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """处理文档分析请求"""
        print("🔬 Analyzing documents...")
        
        # 搜索相关文档
        search_results = self.rag_system.search_relevant_documents(query, top_k=5)
        
        if not search_results:
            return "没有找到相关文档进行分析。"
        
        # 构建分析prompt
        documents_text = "\n".join([doc.get('content', '') for doc in search_results])
        
        analysis_prompt = f"""
        请对以下文档内容进行深度分析：

        用户查询：{query}
        
        文档内容：
        {documents_text[:4000]}

        分析要求：
        1. 识别关键主题和概念
        2. 分析文档之间的关联性
        3. 提取重要见解和结论
        4. 回答用户的具体问题
        """
        
        try:
            analysis = self.llm.invoke(analysis_prompt)
            return f"🔍 文档分析结果：\n\n{analysis}"
        except Exception as e:
            return f"分析文档时发生错误：{str(e)}"
