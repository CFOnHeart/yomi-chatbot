#!/usr/bin/env python3
"""
启动Yomi Chatbot API服务器
"""
import os
import sys
import uvicorn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """启动API服务器"""
    print("🚀 启动 Yomi Chatbot API 服务器...")
    print("📍 访问地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    print("🔄 流式聊天: http://localhost:8000/chat/stream")
    print("🌐 网页客户端: http://localhost:8000/static/chat.html")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "src.api.streaming_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
