from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Union, Any

# 思考配置模型
class ThinkingConfig(BaseModel):
    thinking_budget: Optional[int] = None  # 0-32768, 0=禁用思考, None=自动
    include_thoughts: Optional[bool] = True  # 是否在响应中包含思考过程

    class Config:
        extra = "allow"

    @validator('thinking_budget')
    def validate_thinking_budget(cls, v):
        if v is not None:
            if not isinstance(v, int) or v < 0 or v > 32768:
                raise ValueError("thinking_budget must be an integer between 0 and 32768")
        return v


# 文件数据模型
class InlineData(BaseModel):
    """内联数据模型 - 用于小文件(<20MB)"""
    mime_type: Optional[str] = None  # 兼容旧字段名
    mimeType: Optional[str] = None  # Gemini 2.5标准字段名
    data: str  # base64编码的文件数据

    def __init__(self, **data):
        # 确保两种字段名都支持
        if 'mime_type' in data and 'mimeType' not in data:
            data['mimeType'] = data['mime_type']
        elif 'mimeType' in data and 'mime_type' not in data:
            data['mime_type'] = data['mimeType']
        super().__init__(**data)


class FileData(BaseModel):
    """文件引用模型 - 用于已上传的文件"""
    mime_type: Optional[str] = None  # 兼容旧字段名
    mimeType: Optional[str] = None  # Gemini 2.5标准字段名
    file_uri: Optional[str] = None  # 兼容旧字段名
    fileUri: Optional[str] = None  # Gemini 2.5标准字段名

    def __init__(self, **data):
        # 确保两种字段名都支持
        if 'mime_type' in data and 'mimeType' not in data:
            data['mimeType'] = data['mime_type']
        elif 'mimeType' in data and 'mime_type' not in data:
            data['mime_type'] = data['mimeType']

        if 'file_uri' in data and 'fileUri' not in data:
            data['fileUri'] = data['file_uri']
        elif 'fileUri' in data and 'file_uri' not in data:
            data['file_uri'] = data['fileUri']
        super().__init__(**data)


# 多模态内容
class ContentPart(BaseModel):
    type: str  # "text", "image", "audio", "video", "document"
    text: Optional[str] = None

    # Gemini 2.5标准格式
    inlineData: Optional[InlineData] = None
    fileData: Optional[FileData] = None

    # 向后兼容的字段
    inline_data: Optional[InlineData] = None
    file_data: Optional[FileData] = None

    def __init__(self, **data):
        # 处理字段名兼容性
        if 'inline_data' in data and 'inlineData' not in data:
            data['inlineData'] = data['inline_data']
        elif 'inlineData' in data and 'inline_data' not in data:
            data['inline_data'] = data['inlineData']

        if 'file_data' in data and 'fileData' not in data:
            data['fileData'] = data['file_data']
        elif 'fileData' in data and 'file_data' not in data:
            data['file_data'] = data['fileData']

        super().__init__(**data)

# 请求/响应
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Union[str, Dict[str, Any], ContentPart]]]
    reasoning: Optional[str] = None

    class Config:
        extra = "allow"

    @validator('content')
    def validate_content(cls, v):
        """验证并标准化content字段"""
        if isinstance(v, str):
            return v
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("content must be string or array of content objects")

    def get_text_content(self) -> str:
        """获取纯文本内容"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            text_parts = []
            for item in self.content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    if item.get('type') == 'text' and 'text' in item:
                        text_parts.append(item['text'])
                    elif 'text' in item:
                        text_parts.append(item['text'])
            return ' '.join(text_parts) if text_parts else ""
        else:
            return str(self.content)

    def has_multimodal_content(self) -> bool:
        """检查是否包含多模态内容"""
        if isinstance(self.content, list):
            for item in self.content:
                if isinstance(item, dict) and item.get('type') in ['image', 'audio', 'video', 'document']:
                    return True
        return False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None
    thinking_config: Optional[ThinkingConfig] = None

    # OpenAI Compatible 工具调用字段
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    reasoning_effort: Optional[str] = None

    class Config:
        extra = "allow"

    def __init__(self, **data):
        # 参数范围验证
        if 'temperature' in data and data['temperature'] is not None:
            data['temperature'] = max(0.0, min(2.0, data['temperature']))
        if 'top_p' in data and data['top_p'] is not None:
            data['top_p'] = max(0.0, min(1.0, data['top_p']))
        if 'n' in data and data['n'] is not None:
            data['n'] = max(1, min(4, data['n']))
        if 'max_tokens' in data and data['max_tokens'] is not None:
            data['max_tokens'] = max(1, data['max_tokens'])

        super().__init__(**data)

        # reasoning_effort to thinking_budget mapping
        if self.reasoning_effort and not self.thinking_config:
            budget_map = {
                "low": 4096,
                "medium": 8192,
            }
            if self.reasoning_effort in budget_map:
                self.thinking_config = ThinkingConfig(thinking_budget=budget_map[self.reasoning_effort])

# Embedding Models
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None
    task_type: Optional[str] = None
    output_dimensionality: Optional[int] = None

    class Config:
        extra = "allow"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage

# Gemini Native Embedding Models
class EmbedContentConfig(BaseModel):
    task_type: Optional[str] = None
    output_dimensionality: Optional[int] = None

class GeminiEmbeddingRequest(BaseModel):
    contents: Union[str, List[str]]
    config: Optional[EmbedContentConfig] = None
    model: Optional[str] = None # Included for routing purposes, not part of official Gemini request body

    class Config:
        extra = "allow"

class EmbeddingValue(BaseModel):
    values: List[float]

class GeminiEmbeddingResponse(BaseModel):
    embeddings: List[EmbeddingValue]
