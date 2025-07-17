#!/usr/bin/env python3
"""
RAG文档管理工具

用于管理RAG系统中的文档，包括添加、删除、搜索等操作。
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.agent.conversation_agent import create_agent
from src.rag.rag_system import RAGSystem
from src.database.faiss_document_db import FAISSDocumentDatabase

class RAGManager:
    """RAG文档管理器"""
    
    def __init__(self):
        self.agent = create_agent()
        self.rag_system = RAGSystem()
        self.doc_db = FAISSDocumentDatabase()
    
    def add_file(self, file_path: str, title: str = None, category: str = None, 
                tags: str = None, author: str = None) -> str:
        """添加文件到RAG系统"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            metadata = {}
            if title:
                metadata['title'] = title
            if category:
                metadata['category'] = category
            if tags:
                metadata['tags'] = tags
            if author:
                metadata['author'] = author
            
            doc_id = self.rag_system.add_document_from_file(str(path), **metadata)
            print(f"✅ 文件已添加: {file_path}")
            print(f"📄 文档ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            print(f"❌ 添加文件失败: {e}")
            return None
    
    def add_directory(self, dir_path: str, pattern: str = "*", 
                     category: str = None, tags: str = None) -> List[str]:
        """批量添加目录中的文件"""
        try:
            path = Path(dir_path)
            if not path.exists():
                raise FileNotFoundError(f"目录不存在: {dir_path}")
            
            files = list(path.rglob(pattern))
            doc_ids = []
            
            for file_path in files:
                if file_path.is_file():
                    try:
                        metadata = {
                            'category': category or 'imported',
                            'tags': tags or 'batch_import',
                            'source': 'directory_import'
                        }
                        
                        doc_id = self.rag_system.add_document_from_file(str(file_path), **metadata)
                        doc_ids.append(doc_id)
                        print(f"✅ 已添加: {file_path.name}")
                        
                    except Exception as e:
                        print(f"❌ 添加文件失败 {file_path.name}: {e}")
            
            print(f"\n📊 总共添加了 {len(doc_ids)} 个文档")
            return doc_ids
            
        except Exception as e:
            print(f"❌ 批量添加失败: {e}")
            return []
    
    def search_documents(self, query: str, top_k: int = 10):
        """搜索文档"""
        try:
            results = self.rag_system.search_relevant_documents(query, top_k)
            
            if not results:
                print("📝 没有找到相关文档")
                return
            
            print(f"🔍 搜索结果 (共 {len(results)} 个):")
            print("=" * 60)
            
            for i, doc in enumerate(results, 1):
                print(f"\n【{i}】 {doc.title}")
                print(f"📄 ID: {doc.document_id}")
                if doc.file_path:
                    print(f"📁 文件: {doc.file_path}")
                if doc.start_line > 0:
                    print(f"📍 位置: 第 {doc.start_line}-{doc.end_line} 行")
                print(f"🎯 相似度: {doc.similarity_score:.2f}")
                print(f"🔍 搜索类型: {doc.search_type}")
                
                if doc.snippet:
                    print(f"📝 摘要: {doc.snippet[:100]}...")
                else:
                    print(f"📝 内容: {doc.content[:100]}...")
                
                if doc.metadata:
                    meta_info = []
                    if doc.metadata.get('category'):
                        meta_info.append(f"分类: {doc.metadata['category']}")
                    if doc.metadata.get('tags'):
                        meta_info.append(f"标签: {doc.metadata['tags']}")
                    if doc.metadata.get('author'):
                        meta_info.append(f"作者: {doc.metadata['author']}")
                    if meta_info:
                        print(f"ℹ️  元数据: {' | '.join(meta_info)}")
                
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
    
    def list_documents(self, limit: int = 20):
        """列出所有文档"""
        try:
            # 使用空查询搜索所有文档
            results = self.doc_db.search_documents("", limit=limit, search_type='keyword')
            
            if not results:
                print("📝 没有找到任何文档")
                return
            
            print(f"📚 文档列表 (共 {len(results)} 个):")
            print("=" * 60)
            
            for i, doc in enumerate(results, 1):
                print(f"\n【{i}】 {doc.get('title', 'Untitled')}")
                print(f"📄 ID: {doc['id']}")
                if doc.get('file_path'):
                    print(f"📁 文件: {doc['file_path']}")
                print(f"📅 创建时间: {doc.get('created_at', 'Unknown')}")
                print(f"📊 字数: {doc.get('word_count', 0)} 字")
                
                if doc.get('category'):
                    print(f"🏷️  分类: {doc['category']}")
                if doc.get('tags'):
                    print(f"🏷️  标签: {doc['tags']}")
                    
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ 列出文档失败: {e}")
    
    def delete_document(self, doc_id: str):
        """删除文档"""
        try:
            success = self.rag_system.delete_document(doc_id)
            if success:
                print(f"✅ 文档已删除: {doc_id}")
            else:
                print(f"❌ 删除文档失败: {doc_id}")
                
        except Exception as e:
            print(f"❌ 删除文档失败: {e}")
    
    def get_stats(self):
        """获取统计信息"""
        try:
            stats = self.rag_system.get_document_stats()
            
            print("📊 RAG系统统计信息:")
            print("=" * 40)
            print(f"📚 总文档数: {stats.get('total_documents', 0)}")
            print(f"📝 总字数: {stats.get('total_words', 0):,}")
            
            if stats.get('by_type'):
                print(f"\n📁 按文件类型统计:")
                for file_type, count in stats['by_type'].items():
                    print(f"   {file_type or 'unknown'}: {count}")
            
            if stats.get('by_category'):
                print(f"\n🏷️  按分类统计:")
                for category, count in stats['by_category'].items():
                    print(f"   {category}: {count}")
                    
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
    
    def test_chat(self, query: str, session_id: str = "test_session"):
        """测试聊天功能"""
        try:
            print(f"🤖 测试聊天: {query}")
            print("=" * 60)
            
            response = self.agent.chat(session_id, query)
            print(f"\n💬 AI回复:")
            print(response)
            
        except Exception as e:
            print(f"❌ 测试聊天失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RAG文档管理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 添加文件命令
    add_file_parser = subparsers.add_parser('add-file', help='添加文件')
    add_file_parser.add_argument('file_path', help='文件路径')
    add_file_parser.add_argument('--title', help='文档标题')
    add_file_parser.add_argument('--category', help='文档分类')
    add_file_parser.add_argument('--tags', help='文档标签')
    add_file_parser.add_argument('--author', help='文档作者')
    
    # 添加目录命令
    add_dir_parser = subparsers.add_parser('add-dir', help='批量添加目录')
    add_dir_parser.add_argument('dir_path', help='目录路径')
    add_dir_parser.add_argument('--pattern', default='*', help='文件匹配模式')
    add_dir_parser.add_argument('--category', help='文档分类')
    add_dir_parser.add_argument('--tags', help='文档标签')
    
    # 搜索命令
    search_parser = subparsers.add_parser('search', help='搜索文档')
    search_parser.add_argument('query', help='搜索查询')
    search_parser.add_argument('--top-k', type=int, default=10, help='返回结果数量')
    
    # 列出文档命令
    list_parser = subparsers.add_parser('list', help='列出所有文档')
    list_parser.add_argument('--limit', type=int, default=20, help='限制结果数量')
    
    # 删除文档命令
    delete_parser = subparsers.add_parser('delete', help='删除文档')
    delete_parser.add_argument('doc_id', help='文档ID')
    
    # 统计命令
    subparsers.add_parser('stats', help='显示统计信息')
    
    # 测试聊天命令
    test_parser = subparsers.add_parser('test', help='测试聊天')
    test_parser.add_argument('query', help='测试查询')
    test_parser.add_argument('--session-id', default='test_session', help='会话ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = RAGManager()
    
    if args.command == 'add-file':
        manager.add_file(args.file_path, args.title, args.category, args.tags, args.author)
    elif args.command == 'add-dir':
        manager.add_directory(args.dir_path, args.pattern, args.category, args.tags)
    elif args.command == 'search':
        manager.search_documents(args.query, args.top_k)
    elif args.command == 'list':
        manager.list_documents(args.limit)
    elif args.command == 'delete':
        manager.delete_document(args.doc_id)
    elif args.command == 'stats':
        manager.get_stats()
    elif args.command == 'test':
        manager.test_chat(args.query, args.session_id)

if __name__ == '__main__':
    main()
