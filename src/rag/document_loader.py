from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.model.embedding.base_embedding import BaseManagedEmbedding


@dataclass
class DocumentChunk:
    """文档块，包含内容和embedding"""
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentLoader(ABC):
    """文档加载器抽象基类"""

    def __init__(self,
                 embeddings: BaseManagedEmbedding,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 50,
                 encoding: str = 'utf-8',
                 **kwargs):
        """
        初始化文档加载器

        Args:
            embeddings: 嵌入模型，如果为None则使用默认的Azure OpenAI嵌入
            chunk_size: 每个chunk的最大字符数
            chunk_overlap: chunk之间的重叠字符数
            min_chunk_size: 最小chunk大小，小于此值的chunk会被跳过
            encoding: 文件编码
            **kwargs: 其他参数
        """
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.encoding = encoding
        self.kwargs = kwargs

    @abstractmethod
    def load_documents(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """
        加载文档并返回文档块列表

        Args:
            file_path: 文件路径

        Returns:
            List[DocumentChunk]: 文档块列表
        """
        pass

    @abstractmethod
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """
        检查文件是否支持

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否支持该文件类型
        """
        pass

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """生成文本的embedding"""
        if not self.embeddings:
            return None
        
        try:
            embedding_vector = self.embeddings.embed_documents([text])[0]
            return np.array(embedding_vector)
        except Exception as e:
            print(f"⚠️ 生成embedding失败: {e}")
            return None

    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """验证文件路径"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if not path.is_file():
            raise ValueError(f"路径不是文件: {file_path}")
        return path

    def _create_base_metadata(self, file_path: Path) -> Dict[str, Any]:
        """创建基础元数据"""
        return {
            'source_file': str(file_path),
            'file_name': file_path.name,
            'file_type': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
            'created_at': file_path.stat().st_ctime,
            'modified_at': file_path.stat().st_mtime,
        }

    def _should_skip_chunk(self, content: str) -> bool:
        """判断是否应该跳过该chunk"""
        return len(content.strip()) < self.min_chunk_size

    def _clean_content(self, content: str) -> str:
        """清理文档内容"""
        # 移除多余的空白字符
        content = content.strip()
        # 规范化换行符
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        # 移除多余的空行
        lines = content.split('\n')
        cleaned_lines = []
        prev_empty = False

        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')
                prev_empty = True

        return '\n'.join(cleaned_lines)

class PdfDocumentLoader(DocumentLoader):
    """PDF文档加载器"""

    def __init__(self,
                 embeddings: BaseManagedEmbedding,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 50,
                 extract_images: bool = False,
                 password: Optional[str] = None,
                 **kwargs):
        """
        初始化PDF文档加载器

        Args:
            chunk_size: 每个chunk的最大字符数
            chunk_overlap: chunk之间的重叠字符数
            min_chunk_size: 最小chunk大小
            extract_images: 是否提取图片信息
            password: PDF密码
            embeddings: 嵌入模型，如果为None则使用默认的Azure OpenAI嵌入
            **kwargs: 其他参数
        """
        super().__init__(embeddings, chunk_size, chunk_overlap, min_chunk_size, **kwargs)
        self.extract_images = extract_images
        self.password = password

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """检查是否为PDF文件"""
        path = Path(file_path)
        return path.suffix.lower() == '.pdf'

    def load_documents(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """
        加载PDF文档并返回文档块列表

        Args:
            file_path: PDF文件路径

        Returns:
            List[DocumentChunk]: 文档块列表
        """
        path = self._validate_file_path(file_path)

        if not self.is_supported_file(path):
            raise ValueError(f"不支持的文件类型: {path.suffix}")

        # 使用PyPDFLoader加载PDF
        loader = PyPDFLoader(str(path))
        documents = loader.load()

        # 创建基础元数据
        base_metadata = self._create_base_metadata(path)
        base_metadata.update({
            'total_pages': len(documents),
            'loader_type': 'pdf',
            'extract_images': self.extract_images,
        })

        document_chunks = []

        # 处理每一页
        for page_num, doc in enumerate(documents, 1):
            page_content = self._clean_content(doc.page_content)

            if self._should_skip_chunk(page_content):
                continue

            # 分割页面内容
            chunks = self.text_splitter.split_text(page_content)
            
            # 为每个chunk创建DocumentChunk对象
            for chunk_idx, chunk_content in enumerate(chunks):
                if self._should_skip_chunk(chunk_content):
                    continue
                
                # 创建chunk特定的元数据
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'page_number': page_num,
                    'chunk_index': chunk_idx,
                    'chunk_id': f"page_{page_num}_chunk_{chunk_idx}",
                    'chunk_size': len(chunk_content),
                })
                
                # 生成embedding
                embedding = self._generate_embedding(chunk_content)
                
                # 创建DocumentChunk对象
                chunk = DocumentChunk(
                    content=chunk_content,
                    embedding=embedding,
                    metadata=chunk_metadata
                )
                document_chunks.append(chunk)

        return document_chunks

    def get_document_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """获取PDF文档信息"""
        path = self._validate_file_path(file_path)

        try:
            # 使用PyPDFLoader来获取基本信息
            loader = PyPDFLoader(str(path))
            documents = loader.load()
            
            # 计算总文本长度
            total_text_length = sum(len(doc.page_content) for doc in documents)
            
            info = {
                'total_pages': len(documents),
                'estimated_text_length': total_text_length,
                'estimated_chunks': max(1, total_text_length // self.chunk_size),
                'file_path': str(path),
                'file_size': path.stat().st_size,
            }
            
            return info

        except Exception as e:
            print(f"⚠️ 获取PDF信息失败: {e}")
            return {'error': str(e)}