#!/usr/bin/env python3
"""
FAISS RAG系统测试脚本
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
try:
    import faiss
    import numpy as np

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

def test_faiss_installation():
    """测试FAISS安装"""
    print("🧪 测试FAISS安装...")
    
    if not FAISS_AVAILABLE:
        print("❌ FAISS未安装，请运行: pip install faiss-cpu")
        return False
    
    try:
        # 创建简单的测试索引
        d = 64  # 向量维度
        nb = 100  # 数据库大小
        
        # 创建随机数据
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        
        # 创建索引
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        
        # 测试搜索
        xq = np.random.random((5, d)).astype('float32')
        D, I = index.search(xq, 4)
        
        print(f"✅ FAISS测试通过，索引大小: {index.ntotal}")
        return True
        
    except Exception as e:
        print(f"❌ FAISS测试失败: {e}")
        return False

def test_faiss_document_db():
    """测试FAISS文档数据库"""
    print("\n🧪 测试FAISS文档数据库...")
    
    if not FAISS_AVAILABLE:
        print("❌ FAISS未安装，跳过测试")
        return False
    
    try:
        from src.database.faiss_document_db import FAISSDocumentDatabase
        
        # 创建数据库
        db = FAISSDocumentDatabase("test_faiss.db", "test_vectors.index")
        
        # 添加测试文档（不带embedding）
        doc_id1 = db.add_document(
            title="Python编程指南",
            content="Python是一种简单易学的编程语言，适合初学者学习。",
            category="programming",
            tags="python,tutorial"
        )
        
        doc_id2 = db.add_document(
            title="机器学习基础",
            content="机器学习是人工智能的一个重要分支，包括监督学习和无监督学习。",
            category="ai",
            tags="machine learning,ai"
        )
        
        print(f"✅ 添加文档1: {doc_id1[:8]}...")
        print(f"✅ 添加文档2: {doc_id2[:8]}...")
        
        # 测试搜索（不带embedding）
        results = db.search_documents("Python编程", limit=5, search_type='keyword')
        print(f"✅ 关键词搜索结果: {len(results)} 个")
        
        # 测试统计
        stats = db.get_document_stats()
        print(f"✅ 文档统计: {stats}")
        
        # 清理测试文件
        cleanup_files = ["test_faiss.db", "test_vectors.index", "test_vectors_metadata.pkl"]
        for file in cleanup_files:
            if os.path.exists(file):
                os.remove(file)
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS文档数据库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_with_embeddings():
    """测试FAISS与embedding集成"""
    print("\n🧪 测试FAISS与embedding集成...")
    
    if not FAISS_AVAILABLE:
        print("❌ FAISS未安装，跳过测试")
        return False
    
    try:
        from src.database.faiss_document_db import FAISSDocumentDatabase
        
        # 创建数据库
        db = FAISSDocumentDatabase("test_faiss_embed.db", "test_embed_vectors.index")
        
        # 创建模拟embedding
        embedding_dim = 1536  # Azure OpenAI维度
        
        # 添加带embedding的文档
        doc1_embedding = np.random.random((embedding_dim,)).astype('float32')
        doc2_embedding = np.random.random((embedding_dim,)).astype('float32')
        
        doc_id1 = db.add_document(
            title="向量搜索原理",
            content="向量搜索是基于向量空间模型的信息检索技术。",
            embedding=doc1_embedding,
            category="technology"
        )
        
        doc_id2 = db.add_document(
            title="FAISS使用指南",
            content="FAISS是Facebook开发的高效向量搜索库。",
            embedding=doc2_embedding,
            category="technology"
        )
        
        print(f"✅ 添加带embedding的文档1: {doc_id1[:8]}...")
        print(f"✅ 添加带embedding的文档2: {doc_id2[:8]}...")
        
        # 测试语义搜索
        query_embedding = np.random.random((embedding_dim,)).astype('float32')
        semantic_results = db.semantic_search(query_embedding, top_k=5)
        print(f"✅ 语义搜索结果: {len(semantic_results)} 个")
        
        for doc_id, similarity in semantic_results:
            print(f"   - {doc_id[:8]}... (相似度: {similarity:.4f})")
        
        # 测试混合搜索
        results = db.search_documents(
            "向量搜索",
            query_embedding=query_embedding,
            limit=5,
            search_type='hybrid'
        )
        print(f"✅ 混合搜索结果: {len(results)} 个")
        
        # 测试统计
        stats = db.get_document_stats()
        print(f"✅ FAISS统计: {stats.get('faiss_vectors', 0)} 个向量")
        
        # 清理测试文件
        cleanup_files = ["test_faiss_embed.db", "test_embed_vectors.index", "test_embed_vectors_metadata.pkl"]
        for file in cleanup_files:
            if os.path.exists(file):
                os.remove(file)
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS embedding测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_system():
    """测试RAG系统"""
    print("\n🧪 测试RAG系统...")
    
    try:
        from src.rag.rag_system import RAGSystem
        
        # 创建RAG系统
        rag = RAGSystem()
        
        # 添加测试文档
        doc_id = rag.add_document(
            title="测试文档",
            content="这是一个用于测试RAG系统的示例文档。",
            category="test"
        )
        
        print(f"✅ RAG系统添加文档: {doc_id[:8]}...")
        
        # 测试搜索
        results = rag.search_relevant_documents("测试", top_k=3)
        print(f"✅ RAG搜索结果: {len(results)} 个")
        
        # 测试上下文格式化
        if results:
            context = rag.format_context_for_llm(results)
            print(f"✅ 生成上下文长度: {len(context)} 字符")
            
            references = rag.format_source_references(results)
            print(f"✅ 生成引用长度: {len(references)} 字符")
        
        # 获取统计
        stats = rag.get_document_stats()
        print(f"✅ RAG统计: {stats}")
        
        # 清理测试文件
        cleanup_files = ["documents.db", "vectors.index", "vectors_metadata.pkl"]
        for file in cleanup_files:
            if os.path.exists(file):
                os.remove(file)
        
        return True
        
    except Exception as e:
        print(f"❌ RAG系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 FAISS RAG系统测试")
    print("=" * 50)
    
    if not FAISS_AVAILABLE:
        print("❌ FAISS未安装！")
        print("请运行以下命令安装：")
        print("  pip install faiss-cpu")
        print("或者：")
        print("  pip install faiss-gpu  # 如果有GPU支持")
        return
    
    tests = [
        ("FAISS基本功能", test_faiss_installation),
        ("FAISS文档数据库", test_faiss_document_db),
        ("FAISS embedding集成", test_faiss_with_embeddings),
        ("RAG系统", test_rag_system)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                failed += 1
                print(f"❌ {test_name} 失败")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过！FAISS RAG系统准备就绪！")
    else:
        print("⚠️ 部分测试失败，请检查配置")

if __name__ == '__main__':
    main()
