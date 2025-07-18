#!/usr/bin/env python3
"""
PdfDocumentLoader测试脚本
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.document_loader import PdfDocumentLoader, DocumentChunk
from langchain_core.documents import Document


def create_test_pdf(content: str, file_path: str):
    """创建简单的测试PDF文件"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.fonts import addMapping
        import platform

        # 注册中文字体
        try:
            # 根据操作系统选择合适的中文字体
            if platform.system() == 'Windows':
                # Windows系统使用微软雅黑或宋体
                font_paths = [
                    'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
                    'C:/Windows/Fonts/simsun.ttc',  # 宋体
                    'C:/Windows/Fonts/simhei.ttf',  # 黑体
                ]
            elif platform.system() == 'Darwin':  # macOS
                font_paths = [
                    '/System/Library/Fonts/PingFang.ttc',
                    '/System/Library/Fonts/Helvetica.ttc',
                ]
            else:  # Linux
                font_paths = [
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                ]

            # 尝试注册字体
            font_registered = False
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        pdfmetrics.registerFont(TTFont('Chinese', font_path))
                        font_registered = True
                        print(f"✅ 成功注册中文字体: {font_path}")
                        break
                except Exception as e:
                    print(f"⚠️ 字体注册失败 {font_path}: {e}")
                    continue

            if not font_registered:
                print("⚠️ 未找到合适的中文字体，使用默认字体")
                font_name = 'Helvetica'
            else:
                font_name = 'Chinese'

        except Exception as e:
            print(f"⚠️ 字体设置失败，使用默认字体: {e}")
            font_name = 'Helvetica'

        # 创建PDF文档
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # 创建支持中文的段落样式
        chinese_style = ParagraphStyle(
            'Chinese',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=12,
            leading=16,
            encoding='utf-8'
        )

        story = []

        # 分段添加内容
        paragraphs = content.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                # 清理段落内容
                clean_text = paragraph.strip()
                # 替换可能导致问题的字符
                clean_text = clean_text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')

                try:
                    story.append(Paragraph(clean_text, chinese_style))
                    story.append(Spacer(1, 12))
                except Exception as e:
                    print(f"⚠️ 段落处理失败: {e}")
                    # 降级处理：只使用ASCII字符
                    ascii_text = clean_text.encode('ascii', 'ignore').decode('ascii')
                    if ascii_text.strip():
                        story.append(Paragraph(ascii_text, styles['Normal']))
                        story.append(Spacer(1, 12))

        # 如果没有内容，添加一个默认段落
        if not story:
            story.append(Paragraph("Test PDF Document", styles['Normal']))

        doc.build(story)
        return True

    except ImportError:
        print("⚠️ 需要安装reportlab库: pip install reportlab")
        return False
    except Exception as e:
        print(f"⚠️ 创建PDF失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_loader_basic():
    """基本PDF加载测试"""
    print("🧪 测试PdfDocumentLoader基本功能")
    print("=" * 50)

    # 创建临时PDF文件
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # 创建测试内容
        test_content = """
        这是一个测试PDF文档。

        第一章：介绍
        这是一个关于Python编程的测试文档。Python是一种高级编程语言，具有简洁的语法和强大的功能。

        第二章：基础语法
        变量定义：
        name = "测试"
        age = 25

        函数定义：
        def hello_world():
            print("Hello, World!")

        第三章：数据结构
        列表：[1, 2, 3, 4, 5]
        字典：{"key": "value", "name": "测试"}

        这是一个较长的段落，用于测试文档分块功能。我们需要确保PDF加载器能够正确地将长文档分割成合适的块，同时保持内容的完整性和可读性。每个块都应该有适当的元数据，包括页码、块索引等信息。
        """

        # 创建测试PDF文件
        if not create_test_pdf(test_content, tmp_path):
            print("❌ 无法创建测试PDF文件，跳过测试")
            return

        # 测试PDF加载器
        loader = PdfDocumentLoader(
            chunk_size=200,  # 较小的块大小用于测试
            chunk_overlap=50,
            min_chunk_size=20
        )

        # 1. 测试文件类型检查
        print("\n1. 测试文件类型检查...")
        assert loader.is_supported_file(tmp_path), "PDF文件应该被支持"
        assert not loader.is_supported_file("test.txt"), "非PDF文件不应该被支持"
        print("✅ 文件类型检查通过")

        # 2. 测试文档加载
        print("\n2. 测试文档加载...")
        documents = loader.load_documents(tmp_path)
        print(f"   加载了 {len(documents)} 个文档块")

        # 验证返回的是DocumentChunk对象
        assert all(isinstance(doc, DocumentChunk) for doc in documents), "应该返回DocumentChunk对象"
        print("✅ 文档加载成功")

        # 3. 测试文档内容
        print("\n3. 测试文档内容...")
        assert len(documents) > 0, "应该至少有一个文档块"

        # 检查第一个文档
        first_doc = documents[0]
        assert first_doc.content, "文档内容不应为空"
        assert first_doc.embedding is not None, "应该生成embedding"
        assert isinstance(first_doc.embedding, np.ndarray), "embedding应该是numpy数组"
        assert 'page_number' in first_doc.metadata, "应该包含页码信息"
        assert 'chunk_index' in first_doc.metadata, "应该包含块索引信息"
        assert 'file_name' in first_doc.metadata, "应该包含文件名信息"
        assert 'loader_type' in first_doc.metadata, "应该包含加载器类型信息"

        print(f"   第一个文档块内容预览: {first_doc.content[:100]}...")
        print(f"   第一个文档块embedding维度: {first_doc.embedding.shape}")
        print(f"   第一个文档块元数据: {first_doc.metadata}")
        print("✅ 文档内容验证通过")

        # 4. 测试文档信息获取
        print("\n4. 测试文档信息获取...")
        doc_info = loader.get_document_info(tmp_path)
        assert 'total_pages' in doc_info, "应该包含总页数信息"
        assert 'estimated_text_length' in doc_info, "应该包含估计文本长度"
        assert 'estimated_chunks' in doc_info, "应该包含估计块数"

        print(f"   文档信息: {doc_info}")
        print("✅ 文档信息获取成功")

        # 5. 测试内容完整性
        print("\n5. 测试内容完整性...")
        all_content = " ".join(doc.content for doc in documents)
        assert "测试PDF文档" in all_content, "应该包含原始内容"
        assert "Python编程" in all_content, "应该包含原始内容"
        print("✅ 内容完整性验证通过")

        # 6. 测试embedding生成
        print("\n6. 测试embedding生成...")
        embeddings_generated = sum(1 for doc in documents if doc.embedding is not None)
        print(f"   生成embedding的文档块数: {embeddings_generated}/{len(documents)}")

        # 验证embedding的维度一致性
        if embeddings_generated > 0:
            embedding_dims = [doc.embedding.shape[0] for doc in documents if doc.embedding is not None]
            assert len(set(embedding_dims)) == 1, "所有embedding的维度应该一致"
            print(f"   embedding维度: {embedding_dims[0]}")

        print("✅ embedding生成验证通过")

        print("\n🎉 PdfDocumentLoader基本功能测试全部通过!")

    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except:
            pass


def test_pdf_loader_edge_cases():
    """边界情况测试"""
    print("\n🧪 测试PdfDocumentLoader边界情况")
    print("=" * 50)
    
    loader = PdfDocumentLoader()
    
    # 1. 测试不存在的文件
    print("\n1. 测试不存在的文件...")
    try:
        loader.load_documents("nonexistent.pdf")
        assert False, "应该抛出FileNotFoundError"
    except FileNotFoundError:
        print("✅ 正确处理不存在的文件")
    
    # 2. 测试非PDF文件
    print("\n2. 测试非PDF文件...")
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(b"This is not a PDF file")
        tmp_path = tmp_file.name
    
    try:
        try:
            loader.load_documents(tmp_path)
            assert False, "应该抛出ValueError"
        except ValueError:
            print("✅ 正确处理非PDF文件")
    finally:
        os.unlink(tmp_path)
    
    # 3. 测试不同的参数配置
    print("\n3. 测试不同的参数配置...")
    
    # 测试大块大小
    large_chunk_loader = PdfDocumentLoader(chunk_size=2000, chunk_overlap=100)
    assert large_chunk_loader.chunk_size == 2000, "应该正确设置块大小"
    
    # 测试小块大小
    small_chunk_loader = PdfDocumentLoader(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    assert small_chunk_loader.chunk_size == 100, "应该正确设置小块大小"
    assert small_chunk_loader.min_chunk_size == 10, "应该正确设置最小块大小"
    
    print("✅ 参数配置测试通过")
    
    print("\n🎉 PdfDocumentLoader边界情况测试全部通过!")


def test_pdf_loader_performance():
    """性能测试"""
    print("\n🧪 测试PdfDocumentLoader性能")
    print("=" * 50)

    # 创建较大的测试PDF文件
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # 创建长文本内容
        long_content = "\n\n".join([
            f"这是第{i}段内容。" + "测试内容 " * 50 + f"段落{i}结束。"
            for i in range(1, 21)  # 20个段落
        ])

        if not create_test_pdf(long_content, tmp_path):
            print("❌ 无法创建测试PDF文件，跳过性能测试")
            return

        import time

        loader = PdfDocumentLoader(chunk_size=500, chunk_overlap=100)

        # 测试加载时间
        start_time = time.time()
        documents = loader.load_documents(tmp_path)
        load_time = time.time() - start_time

        print(f"   加载时间: {load_time:.2f}秒")
        print(f"   生成文档块数: {len(documents)}")
        print(f"   平均每块处理时间: {load_time/len(documents):.4f}秒")

        # 验证性能指标
        assert load_time < 10, "加载时间不应超过10秒"
        assert len(documents) > 0, "应该生成至少一个文档块"

        print("✅ 性能测试通过")

    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def main():
    """运行所有测试"""
    print("📋 开始PdfDocumentLoader测试套件")
    print("=" * 60)
    
    try:
        test_pdf_loader_basic()
        test_pdf_loader_edge_cases()
        test_pdf_loader_performance()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
