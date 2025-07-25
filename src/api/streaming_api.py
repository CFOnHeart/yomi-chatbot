"""
FastAPI流式聊天API
支持与ConversationAgent的实时对话，并提供选择性的流式输出
"""
import asyncio
import json
import uuid
from typing import AsyncGenerator, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.agent.conversation_agent import conversation_agent
from src.api.streaming_handler import StreamingHandler


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    session_id: Optional[str] = None


class ToolConfirmationRequest(BaseModel):
    """工具确认请求模型"""
    session_id: str
    confirmed: bool
    tool_args: Optional[Dict[str, Any]] = None


class ChatStreamEvent(BaseModel):
    """流式聊天事件模型"""
    event_type: str  # tool_detected, rag_search, llm_response, final_response, tool_confirmation_needed
    data: Dict[str, Any]
    timestamp: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("🚀 启动 Yomi Chatbot API 服务...")
    yield
    print("🛑 关闭 Yomi Chatbot API 服务...")


# 创建FastAPI应用
app = FastAPI(
    title="Yomi Chatbot API",
    description="支持流式输出的AI聊天机器人API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局流式处理器
streaming_handler = StreamingHandler()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/chat/tool-confirm")
async def confirm_tool_execution(request: ToolConfirmationRequest):
    """
    工具执行确认接口
    
    Args:
        request: 包含会话ID和确认结果
        
    Returns:
        确认结果
    """
    try:
        # 通过streaming_handler传递确认结果
        streaming_handler.set_tool_confirmation(
            request.session_id, 
            request.confirmed, 
            request.tool_args
        )
        
        return {
            "success": True,
            "message": f"工具执行{'确认' if request.confirmed else '取消'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口
    
    Args:
        request: 包含用户消息和可选的会话ID
        
    Returns:
        StreamingResponse: 服务器发送事件流
    """
    # 生成或使用提供的会话ID
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    
    async def generate_response() -> AsyncGenerator[str, None]:
        """生成流式响应"""
        try:
            # 设置当前会话的流式处理器
            streaming_handler.set_current_session(session_id)
            
            # 发送开始事件
            yield f"data: {json.dumps({
                'event_type': 'session_start',
                'data': {'session_id': session_id},
                'timestamp': streaming_handler.get_current_timestamp()
            })}\n\n"
            
            # 在后台运行agent处理
            task = asyncio.create_task(
                run_agent_async(session_id, request.message)
            )
            
            # 实时获取流式输出
            while not task.done():
                events = streaming_handler.get_pending_events(session_id)
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n"
                
                await asyncio.sleep(0.1)  # 短暂等待，避免过度占用CPU
            
            # 获取剩余事件（任务完成后可能还有事件）
            final_events = streaming_handler.get_pending_events(session_id)
            for event in final_events:
                yield f"data: {json.dumps(event)}\n\n"
            
            # 获取最终结果
            final_result = await task
            
            # 发送最终事件
            yield f"data: {json.dumps({
                'event_type': 'session_complete',
                'data': {'final_result': final_result},
                'timestamp': streaming_handler.get_current_timestamp()
            })}\n\n"
            
            # 发送结束标记
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            # 发送错误事件
            yield f"data: {json.dumps({
                'event_type': 'error',
                'data': {'error': str(e)},
                'timestamp': streaming_handler.get_current_timestamp()
            })}\n\n"
            
        finally:
            # 清理会话
            streaming_handler.cleanup_session(session_id)
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )


async def run_agent_async(session_id: str, user_input: str) -> str:
    """异步运行agent处理"""
    # 在线程池中运行同步的agent.chat方法
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        conversation_agent.chat, 
        session_id, 
        user_input
    )


@app.get("/chat/sessions")
async def list_sessions():
    """获取所有会话列表"""
    try:
        sessions = conversation_agent.list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions/{session_id}")
async def get_session_info(session_id: str):
    """获取特定会话信息"""
    try:
        info = conversation_agent.get_session_info(session_id)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除特定会话"""
    try:
        success = conversation_agent.delete_session(session_id)
        if success:
            return {"message": f"会话 {session_id} 已删除"}
        else:
            raise HTTPException(status_code=404, detail="会话不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/stats")
async def get_document_stats():
    """获取文档统计信息"""
    try:
        stats = conversation_agent.get_document_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/search")
async def search_documents(query: str, top_k: int = 5):
    """搜索文档"""
    try:
        results = conversation_agent.search_documents(query, top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "Yomi Chatbot API"}


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动 Yomi Chatbot API 服务器...")
    print("📍 访问地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    print("🔄 流式聊天: http://localhost:8000/chat/stream")
    
    uvicorn.run(
        "src.api.streaming_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
