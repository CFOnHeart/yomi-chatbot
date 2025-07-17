from langchain_core.tools import tool

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human for complex problems that require human judgment or expertise.
    
    Args:
        query: The question or problem that needs human assistance
    
    Returns:
        Human response to the query
    """
    print(f"\n🙋‍♂️ 人工协助请求:")
    print(f"📝 问题: {query}")
    print("-" * 40)
    
    # 获取人工输入
    human_response = input("💭 请提供您的回答: ").strip()
    
    if not human_response:
        return "用户未提供回答"
    
    print(f"✅ 收到人工回答: {human_response}")
    return human_response