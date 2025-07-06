import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import os
import time
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# --- 页面配置 ---
st.set_page_config(
    page_title="Gemini API Proxy",
    page_icon="🌠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- API配置 ---
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

if 'streamlit.io' in os.getenv('STREAMLIT_SERVER_HEADLESS', ''):
    API_BASE_URL = os.getenv('API_BASE_URL', 'https://your-app.onrender.com')


# --- API调用函数 ---
def call_api(endpoint: str, method: str = 'GET', data: Any = None, timeout: int = 30) -> Optional[Dict]:
    """统一API调用函数"""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        spinner_message = "加载中..." if method == 'GET' else "保存中..."
        with st.spinner(spinner_message):
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, timeout=timeout)
            else:
                raise ValueError(f"不支持的方法: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API错误: {response.status_code}")
                return None

    except requests.exceptions.Timeout:
        st.error("请求超时，请重试。")
        return None
    except requests.exceptions.ConnectionError:
        st.error("无法连接到API服务。")
        return None
    except Exception as e:
        st.error(f"API错误: {str(e)}")
        return None


def wake_up_service():
    """唤醒服务"""
    try:
        response = requests.get(f"{API_BASE_URL}/wake", timeout=10)
        if response.status_code == 200:
            st.success("服务已激活")
            return True
    except:
        pass
    return False


def check_service_health():
    """检查服务健康状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


# --- 健康检测函数 ---
def check_all_keys_health():
    """一键检测所有Key健康状态"""
    result = call_api('/admin/health/check-all', 'POST', timeout=60)
    return result


def get_health_summary():
    """获取健康状态汇总"""
    result = call_api('/admin/health/summary')
    return result


# --- 缓存函数 ---
@st.cache_data(ttl=30)
def get_cached_stats():
    """获取缓存的统计数据"""
    return call_api('/admin/stats')


@st.cache_data(ttl=60)
def get_cached_status():
    """获取缓存的服务状态"""
    return call_api('/status')


@st.cache_data(ttl=30)
def get_cached_model_config(model_name: str):
    """获取缓存的模型配置"""
    return call_api(f'/admin/models/{model_name}')


@st.cache_data(ttl=30)
def get_cached_gemini_keys():
    """获取缓存的Gemini密钥列表"""
    return call_api('/admin/keys/gemini')


@st.cache_data(ttl=30)
def get_cached_user_keys():
    """获取缓存的用户密钥列表"""
    return call_api('/admin/keys/user')


@st.cache_data(ttl=30)
def get_cached_health_summary():
    """获取缓存的健康状态汇总"""
    return get_health_summary()


# --- 密钥管理函数 ---
def mask_key(key: str, show_full: bool = False) -> str:
    """密钥掩码处理"""
    if show_full:
        return key

    if key.startswith('sk-'):
        # 用户密钥格式: sk-xxxxxxxx...
        if len(key) > 10:
            return f"{key[:6]}{'•' * (len(key) - 10)}{key[-4:]}"
        return key
    elif key.startswith('AIzaSy'):
        # Gemini密钥格式: AIzaSyxxxxxxx...
        if len(key) > 12:
            return f"{key[:8]}{'•' * (len(key) - 12)}{key[-4:]}"
        return key
    else:
        # 其他格式
        if len(key) > 8:
            return f"{key[:4]}{'•' * (len(key) - 8)}{key[-4:]}"
        return key


def delete_key(key_type: str, key_id: int) -> bool:
    """删除密钥"""
    endpoint = f'/admin/keys/{key_type}/{key_id}'
    result = call_api(endpoint, 'DELETE')
    return result and result.get('success', False)


def toggle_key_status(key_type: str, key_id: int) -> bool:
    """切换密钥状态"""
    endpoint = f'/admin/keys/{key_type}/{key_id}/toggle'
    result = call_api(endpoint, 'POST')
    return result and result.get('success', False)


def get_health_status_color(health_status: str) -> str:
    """获取健康状态颜色"""
    status_colors = {
        'healthy': '#10b981',  # 绿色
        'unhealthy': '#ef4444',  # 红色
        'unknown': '#f59e0b'  # 黄色
    }
    return status_colors.get(health_status, '#6b7280')  # 默认灰色


def format_health_status(health_status: str) -> str:
    """格式化健康状态显示"""
    status_map = {
        'healthy': '正常',
        'unhealthy': '异常',
        'unknown': '未知'
    }
    return status_map.get(health_status, health_status)


# --- 移动端优化的玻璃拟态风格CSS ---
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, shrink-to-fit=no">
<style>
    /* 全局设置 - 防止水平滚动 */
    html, body {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro SC", "SF Pro Display", "Helvetica Neue", "PingFang SC", "Microsoft YaHei UI", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        -webkit-text-size-adjust: 100%;
        -ms-text-size-adjust: 100%;
    }

    /* Streamlit 根容器优化 */
    .stApp {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
        background: linear-gradient(135deg, 
            #e0e7ff 0%, 
            #f3e8ff 25%, 
            #fce7f3 50%, 
            #fef3c7 75%, 
            #dbeafe 100%
        );
        background-size: 400% 400%;
        animation: gradient-shift 20s ease infinite;
        min-height: 100vh;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 主内容区域 - 移动端优化 */
    .block-container {
        width: 100% !important;
        max-width: 100% !important;
        padding: 1rem !important;
        margin: 0 !important;
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.05),
            0 8px 32px rgba(0, 0, 0, 0.03),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        position: relative;
        overflow-x: hidden !important;
        box-sizing: border-box;
    }

    /* 移动端媒体查询 */
    @media (max-width: 768px) {
        .block-container {
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
            border-radius: 12px;
        }

        /* 调整列间距 */
        .stColumns {
            gap: 0.5rem !important;
        }

        .stColumns > div {
            padding: 0 0.25rem !important;
            min-width: 0 !important;
            flex-shrink: 1 !important;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding: 0.5rem !important;
            margin: 0.25rem 0 !important;
            border-radius: 8px;
        }

        .stColumns {
            gap: 0.25rem !important;
        }

        .stColumns > div {
            padding: 0 0.125rem !important;
        }
    }

    /* 确保所有容器不超出屏幕 */
    div[data-testid="stHorizontalBlock"],
    div[data-testid="stVerticalBlock"],
    .stTabs,
    .stForm,
    .stExpander {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
    }

    /* 度量卡片移动端优化 */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 1rem 1.25rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.05),
            0 4px 16px rgba(0, 0, 0, 0.03),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box;
    }

    @media (max-width: 768px) {
        [data-testid="metric-container"] {
            padding: 0.75rem 1rem;
            border-radius: 12px;
        }

        [data-testid="metric-container"] > div:nth-child(1) {
            font-size: 0.75rem;
        }

        [data-testid="metric-container"] > div:nth-child(2) {
            font-size: 1.5rem;
        }
    }

    /* 侧边栏移动端优化 */
    section[data-testid="stSidebar"] {
        width: 100% !important;
        max-width: 280px !important;
        background: linear-gradient(135deg, 
            rgba(99, 102, 241, 0.12) 0%,
            rgba(168, 85, 247, 0.08) 25%,
            rgba(59, 130, 246, 0.1) 50%,
            rgba(139, 92, 246, 0.08) 75%,
            rgba(99, 102, 241, 0.12) 100%
        );
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-right: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 4px 0 32px rgba(0, 0, 0, 0.08);
        overflow-x: hidden !important;
    }

    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            max-width: 240px !important;
        }

        section[data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(2) {
            padding: 1rem 1rem;
        }
    }

    /* 侧边栏Logo区域移动端优化 */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 1rem;
        margin-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @media (max-width: 768px) {
        .sidebar-logo {
            padding: 0.75rem 0.75rem;
            margin-bottom: 1rem;
            border-radius: 12px;
            gap: 0.5rem;
        }

        .sidebar-logo-icon {
            font-size: 2rem !important;
        }

        .sidebar-logo-title {
            font-size: 1.125rem !important;
        }

        .sidebar-logo-subtitle {
            font-size: 0.8125rem !important;
        }
    }

    /* 导航项移动端优化 */
    section[data-testid="stSidebar"] .stRadio > div > label {
        font-size: 0.9375rem !important;
        padding: 0.875rem 1.125rem !important;
        border-radius: 14px !important;
        width: 100% !important;
        box-sizing: border-box !important;
        word-wrap: break-word !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    @media (max-width: 768px) {
        section[data-testid="stSidebar"] .stRadio > div > label {
            font-size: 0.875rem !important;
            padding: 0.75rem 1rem !important;
            border-radius: 12px !important;
        }
    }

    /* 按钮移动端优化 */
    .stButton > button {
        width: 100% !important;
        max-width: 100% !important;
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        font-size: 0.875rem;
        padding: 0.625rem 1rem;
        letter-spacing: 0.02em;
        background: rgba(99, 102, 241, 0.1);
        color: #4f46e5;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 
            0 8px 24px rgba(99, 102, 241, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        position: relative;
        overflow: hidden;
        box-sizing: border-box;
        white-space: nowrap;
        text-overflow: ellipsis;
    }

    @media (max-width: 768px) {
        .stButton > button {
            font-size: 0.8125rem;
            padding: 0.5rem 0.75rem;
            border-radius: 10px;
        }
    }

    /* 输入框移动端优化 */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        width: 100% !important;
        max-width: 100% !important;
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px !important;
        font-size: 0.875rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
        color: #1f2937 !important;
        box-sizing: border-box !important;
    }

    @media (max-width: 768px) {
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select,
        .stTextArea > div > div > textarea {
            font-size: 0.8125rem !important;
            padding: 0.625rem 0.875rem !important;
            border-radius: 8px !important;
        }
    }

    /* 健康状态标签移动端优化 */
    .status-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 0.875rem;
        border-radius: 16px;
        font-size: 0.8125rem;
        font-weight: 500;
        line-height: 1;
        white-space: nowrap;
        min-width: 3rem;
        max-width: 100%;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-sizing: border-box;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    @media (max-width: 768px) {
        .status-badge {
            padding: 0.375rem 0.625rem;
            border-radius: 12px;
            font-size: 0.75rem;
            min-width: 2.5rem;
        }
    }

    /* 密钥卡片移动端优化 */
    div[data-testid="stHorizontalBlock"] {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.75rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 10px 32px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        position: relative;
        overflow: hidden;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box;
    }

    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 12px;
        }
    }

    /* 密钥代码显示移动端优化 */
    .key-code {
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        font-family: 'SF Mono', Monaco, 'Cascadia Mono', monospace;
        font-size: 0.875rem;
        color: #1f2937;
        overflow: hidden;
        text-overflow: ellipsis;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.3);
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        white-space: nowrap;
    }

    @media (max-width: 768px) {
        .key-code {
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            font-size: 0.8125rem;
        }
    }

    .key-id {
        font-weight: 600;
        color: #374151;
        min-width: 2rem;
        white-space: nowrap;
    }

    .key-meta {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.375rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    @media (max-width: 768px) {
        .key-meta {
            font-size: 0.6875rem;
        }
    }

    /* 标签页移动端优化 */
    .stTabs [data-testid="stTabBar"] {
        gap: 1rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
        padding: 0;
        margin-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px 12px 0 0;
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-bottom: none;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
    }

    @media (max-width: 768px) {
        .stTabs [data-testid="stTabBar"] {
            gap: 0.5rem;
            margin-bottom: 1rem;
            border-radius: 8px 8px 0 0;
        }

        .stTabs [data-testid="stTabBar"] button {
            font-size: 0.875rem;
            padding: 1rem 1.25rem;
            min-width: auto;
            white-space: nowrap;
        }
    }

    /* 图表容器移动端优化 */
    .js-plotly-plot .plotly {
        border-radius: 16px;
        overflow: hidden;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        background: rgba(255, 255, 255, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        margin: 0.25rem;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box;
    }

    @media (max-width: 768px) {
        .js-plotly-plot .plotly {
            border-radius: 12px;
            margin: 0.125rem;
        }
    }

    /* 表格移动端优化 */
    .stDataFrame {
        border-radius: 12px;
        overflow: auto;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        background: rgba(255, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box;
    }

    /* 标题移动端优化 */
    h1 {
        background: linear-gradient(135deg, #1f2937 0%, #4f46e5 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
        overflow-wrap: break-word;
        word-wrap: break-word;
    }

    @media (max-width: 768px) {
        h1 {
            font-size: 1.75rem;
        }

        h2 {
            font-size: 1.5rem;
        }

        h3 {
            font-size: 1.125rem;
        }
    }

    /* 页面副标题移动端优化 */
    .page-subtitle {
        font-size: 0.9375rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
        font-weight: 400;
        overflow-wrap: break-word;
        word-wrap: break-word;
    }

    @media (max-width: 768px) {
        .page-subtitle {
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }
    }

    /* Alert消息移动端优化 */
    [data-testid="stAlert"] {
        border: none !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
        padding: 0.875rem 1rem !important;
        margin: 0.75rem 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
    }

    @media (max-width: 768px) {
        [data-testid="stAlert"] {
            padding: 0.75rem 0.875rem !important;
            margin: 0.5rem 0 !important;
            border-radius: 10px !important;
        }
    }

    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 3px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }

    /* 禁用水平滚动 */
    * {
        box-sizing: border-box;
    }

    *:not(pre):not(code) {
        overflow-x: hidden !important;
    }

    /* 确保flex容器不超出 */
    .stColumns {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    .stColumns > div {
        min-width: 0 !important;
        overflow-x: hidden !important;
        flex-shrink: 1 !important;
    }

    /* Form容器优化 */
    .stForm {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
    }

    /* Expander优化 */
    .streamlit-expanderHeader {
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
    }

    /* 选择文本样式 */
    ::selection {
        background: rgba(99, 102, 241, 0.2);
        color: #1f2937;
    }

    ::-moz-selection {
        background: rgba(99, 102, 241, 0.2);
        color: #1f2937;
    }

    /* 防止图片和视频超出容器 */
    img, video, iframe {
        max-width: 100% !important;
        height: auto !important;
        box-sizing: border-box !important;
    }
</style>
""", unsafe_allow_html=True)


# --- 获取服务状态函数 ---
@st.cache_data(ttl=10)
def get_service_status():
    """获取服务状态，用于侧边栏显示"""
    try:
        health = check_service_health()
        stats = get_cached_stats()
        if health and stats:
            return {
                'online': True,
                'active_keys': stats.get('active_gemini_keys', 0),
                'healthy_keys': stats.get('healthy_gemini_keys', 0)
            }
    except:
        pass
    return {'online': False, 'active_keys': 0, 'healthy_keys': 0}


# --- 玻璃拟态侧边栏 ---
with st.sidebar:
    # Logo区域
    st.markdown('''
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🌠</div>
        <div class="sidebar-logo-text">
            <div class="sidebar-logo-title">Gemini Proxy</div>
            <div class="sidebar-logo-subtitle">多Key智能轮询系统</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # 导航标题
    st.markdown('<div class="sidebar-section-title">主菜单</div>', unsafe_allow_html=True)

    # 创建带图标的导航选项
    nav_options = {
        "🏠 控制台": "控制台",
        "⚙️ 模型配置": "模型配置",
        "🔑 密钥管理": "密钥管理",
        "🔧 系统设置": "系统设置"
    }

    # 使用自定义HTML为导航项添加图标
    page_display = st.radio(
        "导航",
        list(nav_options.keys()),
        label_visibility="collapsed",
        key="nav_radio"
    )

    # 转换显示值为实际页面值
    page = nav_options[page_display]

    # 添加状态指示器
    st.markdown('<div class="sidebar-status">', unsafe_allow_html=True)

    # 服务状态
    service_status = get_service_status()
    status_class = "online" if service_status['online'] else "offline"
    status_text = "在线" if service_status['online'] else "离线"

    st.markdown(f'''
    <div class="sidebar-status-card">
        <div class="sidebar-status-title">服务状态</div>
        <div class="sidebar-status-content">
            <div class="sidebar-status-indicator {status_class}"></div>
            <div class="sidebar-status-text">{status_text}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # API密钥状态
    if service_status['online']:
        st.markdown(f'''
        <div class="sidebar-status-card">
            <div class="sidebar-status-title">API 密钥</div>
            <div class="sidebar-status-content">
                <div class="sidebar-status-text">{service_status['healthy_keys']} / {service_status['active_keys']} 正常</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 底部信息
    st.markdown(f'''
    <div class="sidebar-footer">
        <div class="sidebar-footer-content">
            <div class="sidebar-footer-item">
                <span>版本 v1.1.0</span>
            </div>
            <div class="sidebar-footer-item">
                <a href="{API_BASE_URL}/docs" target="_blank" class="sidebar-footer-link">API 文档</a>
                <span>·</span>
                <a href="https://github.com/arain119" target="_blank" class="sidebar-footer-link">GitHub</a>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

# --- 主页面内容 ---
if page == "控制台":
    st.title("控制台")
    st.markdown('<p class="page-subtitle">实时监控服务运行状态和使用情况</p>', unsafe_allow_html=True)

    # 获取统计数据
    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("无法获取服务数据，请检查服务连接")
        st.stop()

    # 健康状态提示和刷新按钮
    col1, col2 = st.columns([11, 1])

    with col1:
        health_summary = stats_data.get('health_summary', {})
        if health_summary:
            total_active = health_summary.get('total_active', 0)
            healthy_count = health_summary.get('healthy', 0)
            unhealthy_count = health_summary.get('unhealthy', 0)

            if unhealthy_count > 0:
                st.error(f"发现 {unhealthy_count} 个异常密钥，共 {total_active} 个激活密钥")
            elif healthy_count > 0:
                st.success(f"所有 {healthy_count} 个密钥运行正常")
            else:
                st.info("暂无激活的密钥")

    with col2:
        if st.button("⟳", help="刷新数据", key="refresh_dashboard"):
            st.cache_data.clear()
            st.rerun()

    # 核心指标
    st.markdown("### 核心指标")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gemini_keys = stats_data.get('active_gemini_keys', 0)
        healthy_gemini = stats_data.get('healthy_gemini_keys', 0)
        st.metric(
            "Gemini密钥",
            gemini_keys,
            delta=f"{healthy_gemini} 正常"
        )

    with col2:
        user_keys = stats_data.get('active_user_keys', 0)
        total_user = stats_data.get('user_keys', 0)
        st.metric(
            "用户密钥",
            user_keys,
            delta=f"共 {total_user} 个"
        )

    with col3:
        models = stats_data.get('supported_models', [])
        st.metric("支持模型", len(models))

    with col4:
        thinking_status = "启用" if status_data.get('thinking_enabled', False) else "禁用"
        st.metric("思考功能", thinking_status)

    # 使用率分析
    st.markdown("### 使用率分析")

    usage_stats = stats_data.get('usage_stats', {})
    if usage_stats and models:
        # 准备数据
        model_data = []
        for model in models:
            stats = usage_stats.get(model, {'minute': {'requests': 0}, 'day': {'requests': 0}})

            model_config_data = get_cached_model_config(model)
            if not model_config_data:
                rpm_limit = 10 if 'flash' in model else 5
                rpd_limit = 250 if 'flash' in model else 100
            else:
                rpm_limit = model_config_data.get('total_rpm_limit', 10)
                rpd_limit = model_config_data.get('total_rpd_limit', 250)

            rpm_used = stats['minute']['requests']
            rpm_percent = (rpm_used / rpm_limit * 100) if rpm_limit > 0 else 0

            rpd_used = stats['day']['requests']
            rpd_percent = (rpd_used / rpd_limit * 100) if rpd_limit > 0 else 0

            model_data.append({
                'Model': model,
                'RPM Used': rpm_used,
                'RPM Limit': rpm_limit,
                'RPM %': rpm_percent,
                'RPD Used': rpd_used,
                'RPD Limit': rpd_limit,
                'RPD %': rpd_percent
            })

        if model_data:
            df = pd.DataFrame(model_data)

            # 移动端友好的图表布局
            # 在移动端使用单列布局，桌面端使用双列
            col1, col2 = st.columns(2)

            with col1:
                fig_rpm = go.Figure()
                fig_rpm.add_trace(go.Bar(
                    x=df['Model'],
                    y=df['RPM %'],
                    text=[f"{x:.1f}%" for x in df['RPM %']],
                    textposition='outside',
                    marker_color='rgba(99, 102, 241, 0.8)',
                    marker_line=dict(width=0),
                    hovertemplate='<b>%{x}</b><br>使用率: %{y:.1f}%<br>当前: %{customdata[0]:,}<br>限制: %{customdata[1]:,}<extra></extra>',
                    customdata=df[['RPM Used', 'RPM Limit']].values
                ))
                fig_rpm.update_layout(
                    title="每分钟请求数 (RPM)",
                    title_font=dict(size=14, color='#1f2937', family='-apple-system, BlinkMacSystemFont'),
                    yaxis_title="使用率 (%)",
                    yaxis_range=[0, max(100, df['RPM %'].max() * 1.2) if len(df) > 0 else 100],
                    height=280,
                    showlegend=False,
                    plot_bgcolor='rgba(255, 255, 255, 0.3)',
                    paper_bgcolor='rgba(255, 255, 255, 0.3)',
                    font=dict(family='-apple-system, BlinkMacSystemFont', color='#374151', size=10),
                    yaxis=dict(gridcolor='rgba(107, 114, 128, 0.2)', zerolinecolor='rgba(107, 114, 128, 0.3)',
                               color='#374151'),
                    xaxis=dict(linecolor='rgba(107, 114, 128, 0.3)', color='#374151'),
                    bargap=0.4,
                    margin=dict(l=30, r=30, t=40, b=30)
                )
                st.plotly_chart(fig_rpm, use_container_width=True)

            with col2:
                fig_rpd = go.Figure()
                fig_rpd.add_trace(go.Bar(
                    x=df['Model'],
                    y=df['RPD %'],
                    text=[f"{x:.1f}%" for x in df['RPD %']],
                    textposition='outside',
                    marker_color='rgba(16, 185, 129, 0.8)',
                    marker_line=dict(width=0),
                    hovertemplate='<b>%{x}</b><br>使用率: %{y:.1f}%<br>当前: %{customdata[0]:,}<br>限制: %{customdata[1]:,}<extra></extra>',
                    customdata=df[['RPD Used', 'RPD Limit']].values
                ))
                fig_rpd.update_layout(
                    title="每日请求数 (RPD)",
                    title_font=dict(size=14, color='#1f2937', family='-apple-system, BlinkMacSystemFont'),
                    yaxis_title="使用率 (%)",
                    yaxis_range=[0, max(100, df['RPD %'].max() * 1.2) if len(df) > 0 else 100],
                    height=280,
                    showlegend=False,
                    plot_bgcolor='rgba(255, 255, 255, 0.3)',
                    paper_bgcolor='rgba(255, 255, 255, 0.3)',
                    font=dict(family='-apple-system, BlinkMacSystemFont', color='#374151', size=10),
                    yaxis=dict(gridcolor='rgba(107, 114, 128, 0.2)', zerolinecolor='rgba(107, 114, 128, 0.3)',
                               color='#374151'),
                    xaxis=dict(linecolor='rgba(107, 114, 128, 0.3)', color='#374151'),
                    bargap=0.4,
                    margin=dict(l=30, r=30, t=40, b=30)
                )
                st.plotly_chart(fig_rpd, use_container_width=True)

            # 详细数据表
            with st.expander("查看详细数据"):
                display_df = df[['Model', 'RPM Used', 'RPM Limit', 'RPM %', 'RPD Used', 'RPD Limit', 'RPD %']].copy()
                display_df.columns = ['模型', '分钟请求', '分钟限制', '分钟使用率', '日请求', '日限制', '日使用率']
                display_df['分钟使用率'] = display_df['分钟使用率'].apply(lambda x: f"{x:.1f}%")
                display_df['日使用率'] = display_df['日使用率'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无使用数据")

elif page == "密钥管理":
    st.title("密钥管理")
    st.markdown('<p class="page-subtitle">管理 Gemini API 密钥和用户访问令牌</p>', unsafe_allow_html=True)

    # 刷新按钮
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("⟳", help="刷新数据", key="refresh_keys"):
            st.cache_data.clear()
            st.rerun()

    tab1, tab2 = st.tabs(["Gemini 密钥", "用户密钥"])

    with tab1:
        st.markdown("#### 添加新密钥")

        with st.form("add_gemini_key"):
            new_key = st.text_area(
                "Gemini API 密钥",
                height=120,
                placeholder="AIzaSy...\n\n支持批量添加：\n- 多个密钥可用逗号、分号或换行符分隔\n- 示例：AIzaSy123..., AIzaSy456...; AIzaSy789...",
                help="从 Google AI Studio 获取。支持批量添加：用逗号、分号、换行符或多个空格分隔多个密钥"
            )
            submitted = st.form_submit_button("添加密钥", type="primary")

            if submitted and new_key:
                result = call_api('/admin/config/gemini-key', 'POST', {'key': new_key})
                if result:
                    if result.get('success'):
                        # 显示成功消息
                        st.success(result.get('message', '密钥添加成功'))

                        # 如果是批量添加，显示详细结果
                        total_processed = result.get('total_processed', 1)
                        if total_processed > 1:
                            successful = result.get('successful_adds', 0)
                            failed = result.get('failed_adds', 0)

                            # 创建详细信息展开器
                            with st.expander(f"查看详细结果 (处理了 {total_processed} 个密钥)", expanded=failed > 0):
                                if successful > 0:
                                    st.markdown("**✅ 成功添加的密钥：**")
                                    success_details = [detail for detail in result.get('details', []) if '✅' in detail]
                                    for detail in success_details:
                                        st.markdown(f"- {detail}")

                                if result.get('duplicate_keys'):
                                    st.markdown("**⚠️ 重复的密钥（已跳过）：**")
                                    for duplicate in result.get('duplicate_keys', []):
                                        st.warning(f"- {duplicate}")

                                if result.get('invalid_keys'):
                                    st.markdown("**❌ 无效的密钥：**")
                                    for invalid in result.get('invalid_keys', []):
                                        st.error(f"- {invalid}")

                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        # 显示失败消息和详细信息
                        st.error(result.get('message', '添加失败'))

                        # 显示失败详情
                        if result.get('invalid_keys'):
                            with st.expander("查看失败详情"):
                                st.markdown("**格式错误的密钥：**")
                                for invalid in result.get('invalid_keys', []):
                                    st.write(f"- {invalid}")

                        if result.get('duplicate_keys'):
                            with st.expander("重复的密钥"):
                                for duplicate in result.get('duplicate_keys', []):
                                    st.write(f"- {duplicate}")
                else:
                    st.error("网络错误，请重试")

        st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)

        # 现有密钥
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.markdown("#### 现有密钥")
        with col2:
            if st.button("健康检测", help="检测所有密钥状态", key="health_check_gemini"):
                with st.spinner("检测中..."):
                    result = check_all_keys_health()
                    if result and result.get('success'):
                        st.success(result['message'])
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
        with col3:
            show_full_keys = st.checkbox("显示完整", key="show_gemini_full")

        # 获取密钥列表
        gemini_keys_data = get_cached_gemini_keys()
        if gemini_keys_data and gemini_keys_data.get('success'):
            gemini_keys = gemini_keys_data.get('keys', [])

            if gemini_keys:
                # 统计信息
                active_count = len([k for k in gemini_keys if k.get('status') == 1])
                healthy_count = len(
                    [k for k in gemini_keys if k.get('status') == 1 and k.get('health_status') == 'healthy'])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div style="color: #374151; font-weight: 500;">共 {len(gemini_keys)} 个密钥</div>',
                                unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div style="color: #374151; font-weight: 500;">激活 {active_count} 个</div>',
                                unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div style="color: #059669; font-weight: 500;">正常 {healthy_count} 个</div>',
                                unsafe_allow_html=True)

                valid_keys = []
                invalid_count = 0

                for key_info in gemini_keys:
                    # 验证数据完整性
                    if (isinstance(key_info, dict) and
                            'id' in key_info and
                            'key' in key_info and
                            'status' in key_info and
                            key_info['id'] is not None and
                            key_info['key'] is not None):
                        valid_keys.append(key_info)
                    else:
                        invalid_count += 1

                # 如果有无效数据，给出提示
                if invalid_count > 0:
                    st.warning(f"发现 {invalid_count} 个数据不完整的密钥，已跳过显示")

                # 渲染有效的密钥 - 移动端优化布局
                for key_info in valid_keys:
                    try:
                        # 创建一个容器来包含整个密钥卡片
                        container = st.container()
                        with container:
                            # 移动端友好的布局：使用响应式列
                            col1, col2, col3, col4, col5, col6 = st.columns([0.5, 3, 1, 1, 0.8, 0.8])

                            with col1:
                                st.markdown(f'<div class="key-id">#{key_info.get("id", "N/A")}</div>',
                                            unsafe_allow_html=True)

                            with col2:
                                st.markdown(f'''
                                <div>
                                    <div class="key-code">{mask_key(key_info.get('key', ''), show_full_keys)}</div>
                                    <div class="key-meta">
                                        {f"成功率 {key_info.get('success_rate', 1.0) * 100:.1f}% · 响应时间 {key_info.get('avg_response_time', 0.0):.2f}s · 请求数 {key_info.get('total_requests', 0)}"
                                if key_info.get('total_requests', 0) > 0 else "尚未使用"}
                                    </div>
                                </div>
                                ''', unsafe_allow_html=True)

                            with col3:
                                st.markdown(f'''
                                <span class="status-badge status-{key_info.get('health_status', 'unknown')}">
                                    {format_health_status(key_info.get('health_status', 'unknown'))}
                                </span>
                                ''', unsafe_allow_html=True)

                            with col4:
                                st.markdown(f'''
                                <span class="status-badge status-{'active' if key_info.get('status', 0) == 1 else 'inactive'}">
                                    {'激活' if key_info.get('status', 0) == 1 else '禁用'}
                                </span>
                                ''', unsafe_allow_html=True)

                            with col5:
                                key_id = key_info.get('id')
                                status = key_info.get('status', 0)
                                if key_id is not None:
                                    toggle_text = "禁用" if status == 1 else "激活"
                                    if st.button(toggle_text, key=f"toggle_g_{key_id}", use_container_width=True):
                                        if toggle_key_status('gemini', key_id):
                                            st.success("状态已更新")
                                            st.cache_data.clear()
                                            time.sleep(1)
                                            st.rerun()

                            with col6:
                                if key_id is not None:
                                    if st.button("删除", key=f"del_g_{key_id}", use_container_width=True):
                                        if delete_key('gemini', key_id):
                                            st.success("删除成功")
                                            st.cache_data.clear()
                                            time.sleep(1)
                                            st.rerun()

                    except Exception as e:
                        # 异常时显示错误信息而不是空白
                        st.error(f"渲染密钥 #{key_info.get('id', '?')} 时出错: {str(e)}")

                # 如果没有有效密钥
                if not valid_keys:
                    st.warning("所有密钥数据都不完整，请检查数据源")

            else:
                st.info("暂无密钥，请添加第一个 Gemini API 密钥")
        else:
            st.error("无法获取密钥列表")

    with tab2:
        st.markdown("#### 生成访问密钥")

        with st.form("generate_user_key"):
            key_name = st.text_input("密钥名称", placeholder="例如：生产环境、测试环境")
            submitted = st.form_submit_button("生成新密钥", type="primary")

            if submitted:
                name = key_name if key_name else '未命名'
                result = call_api('/admin/config/user-key', 'POST', {'name': name})
                if result and result.get('success'):
                    new_key = result.get('key')
                    st.success("密钥生成成功")
                    st.warning("请立即保存此密钥，它不会再次显示")
                    st.code(new_key, language=None)

                    with st.expander("使用示例"):
                        st.code(f"""
import openai

client = openai.OpenAI(
    api_key="{new_key}",
    base_url="{API_BASE_URL}/v1"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{{"role": "user", "content": "Hello"}}]
)
                        """, language="python")

                    st.cache_data.clear()

        st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)

        # 现有密钥
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("#### 现有密钥")
        with col2:
            show_full_user_keys = st.checkbox("显示完整", key="show_user_full")

        # 获取用户密钥
        user_keys_data = get_cached_user_keys()
        if user_keys_data and user_keys_data.get('success'):
            user_keys = user_keys_data.get('keys', [])

            if user_keys:
                active_count = len([k for k in user_keys if k['status'] == 1])
                st.markdown(
                    f'<div style="color: #6b7280; font-weight: 500; margin-bottom: 1rem;">共 {len(user_keys)} 个密钥，{active_count} 个激活</div>',
                    unsafe_allow_html=True)

                for key_info in user_keys:
                    container = st.container()
                    with container:
                        # 移动端友好的布局
                        col1, col2, col3, col4, col5 = st.columns([0.5, 3, 1, 0.8, 0.8])

                        with col1:
                            st.markdown(f'<div class="key-id">#{key_info["id"]}</div>', unsafe_allow_html=True)

                        with col2:
                            st.markdown(f'''
                            <div>
                                <div class="key-code">{mask_key(key_info['key'], show_full_user_keys)}</div>
                                <div class="key-meta">
                                    {f"名称: {key_info['name']}" if key_info.get('name') else "未命名"} · 
                                    {f"最后使用: {key_info['last_used'][:16]}" if key_info.get('last_used') else "从未使用"}
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)

                        with col3:
                            st.markdown(f'''
                            <span class="status-badge status-{'active' if key_info['status'] == 1 else 'inactive'}">
                                {'激活' if key_info['status'] == 1 else '停用'}
                            </span>
                            ''', unsafe_allow_html=True)

                        with col4:
                            toggle_text = "停用" if key_info['status'] == 1 else "激活"
                            if st.button(toggle_text, key=f"toggle_u_{key_info['id']}", use_container_width=True):
                                if toggle_key_status('user', key_info['id']):
                                    st.success("状态已更新")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()

                        with col5:
                            if st.button("删除", key=f"del_u_{key_info['id']}", use_container_width=True):
                                if delete_key('user', key_info['id']):
                                    st.success("删除成功")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()

            else:
                st.info("暂无用户密钥")

elif page == "模型配置":
    st.title("模型配置")
    st.markdown('<p class="page-subtitle">调整模型参数和使用限制</p>', unsafe_allow_html=True)

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("无法获取数据")
        st.stop()

    models = status_data.get('models', [])
    if not models:
        st.warning("暂无可用模型")
        st.stop()

    # 信息提示
    st.info('显示的限制针对单个 API Key，总限制会根据健康密钥数量自动倍增')

    for model in models:
        st.markdown(f"### {model}")

        current_config = get_cached_model_config(model)
        if not current_config or not current_config.get('success'):
            st.warning(f"无法加载模型配置")
            continue

        with st.form(f"model_config_{model}"):
            # 移动端友好的列布局
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                rpm = st.number_input(
                    "RPM (每分钟请求)",
                    min_value=1,
                    value=current_config.get('single_api_rpm_limit', 10 if 'flash' in model else 5),
                    key=f"rpm_{model}"
                )

            with col2:
                rpd = st.number_input(
                    "RPD (每日请求)",
                    min_value=1,
                    value=current_config.get('single_api_rpd_limit', 250 if 'flash' in model else 100),
                    key=f"rpd_{model}"
                )

            with col3:
                tpm = st.number_input(
                    "TPM (每分钟令牌)",
                    min_value=1000,
                    value=current_config.get('single_api_tpm_limit', 250000),
                    key=f"tpm_{model}"
                )

            with col4:
                status_options = {1: "激活", 0: "禁用"}
                current_status = current_config.get('status', 1)
                new_status = st.selectbox(
                    "状态",
                    options=list(status_options.values()),
                    index=0 if current_status == 1 else 1,
                    key=f"status_{model}"
                )

            if st.form_submit_button("保存配置", type="primary", use_container_width=True):
                update_data = {
                    "single_api_rpm_limit": rpm,
                    "single_api_rpd_limit": rpd,
                    "single_api_tpm_limit": tpm,
                    "status": 1 if new_status == "激活" else 0
                }

                result = call_api(f'/admin/models/{model}', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success("配置已保存")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("保存失败")

elif page == "系统设置":
    st.title("系统设置")
    st.markdown('<p class="page-subtitle">配置高级功能和系统参数</p>', unsafe_allow_html=True)

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("无法获取配置数据")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["思考模式", "提示词注入", "负载均衡", "系统信息"])

    with tab1:
        st.markdown("#### 思考模式配置")
        st.markdown("启用推理功能以提高复杂查询的响应质量")

        thinking_config = stats_data.get('thinking_config', {})

        with st.form("thinking_config_form"):
            col1, col2 = st.columns(2)

            with col1:
                thinking_enabled = st.checkbox(
                    "启用思考模式",
                    value=thinking_config.get('enabled', False)
                )

                include_thoughts = st.checkbox(
                    "在响应中包含思考过程",
                    value=thinking_config.get('include_thoughts', False)
                )

            with col2:
                budget_options = {
                    "自动": -1,
                    "禁用": 0,
                    "低 (4k)": 4096,
                    "中 (8k)": 8192,
                    "flash最大思考预算 (24k)": 24576,
                    "pro最大思考预算 (32k)": 32768
                }

                current_budget = thinking_config.get('budget', -1)
                selected_option = next((k for k, v in budget_options.items() if v == current_budget), "自动")

                budget_option = st.selectbox(
                    "思考预算",
                    options=list(budget_options.keys()),
                    index=list(budget_options.keys()).index(selected_option)
                )

            if st.form_submit_button("保存配置", type="primary", use_container_width=True):
                update_data = {
                    "enabled": thinking_enabled,
                    "budget": budget_options[budget_option],
                    "include_thoughts": include_thoughts
                }

                result = call_api('/admin/config/thinking', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success("配置已保存")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()

    with tab2:
        st.markdown("#### 提示词注入")
        st.markdown("为所有请求自动添加自定义指令")

        inject_config = stats_data.get('inject_config', {})

        with st.form("inject_prompt_form"):
            inject_enabled = st.checkbox(
                "启用提示词注入",
                value=inject_config.get('enabled', False)
            )

            position_options = {
                'system': '系统消息',
                'user_prefix': '用户消息前',
                'user_suffix': '用户消息后'
            }

            position = st.selectbox(
                "注入位置",
                options=list(position_options.keys()),
                format_func=lambda x: position_options[x],
                index=list(position_options.keys()).index(inject_config.get('position', 'system'))
            )

            content = st.text_area(
                "提示词内容",
                value=inject_config.get('content', ''),
                height=150,
                placeholder="输入自定义提示词..."
            )

            if st.form_submit_button("保存配置", type="primary", use_container_width=True):
                update_data = {
                    "enabled": inject_enabled,
                    "content": content,
                    "position": position
                }

                result = call_api('/admin/config/inject-prompt', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success("配置已保存")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()

    with tab3:
        st.markdown("#### 负载均衡策略")
        st.markdown("优化 API Key 选择策略")

        # 获取当前策略
        all_configs = call_api('/admin/config')
        current_strategy = 'adaptive'

        if all_configs and all_configs.get('success'):
            system_configs = all_configs.get('system_configs', [])
            for config in system_configs:
                if config['key'] == 'load_balance_strategy':
                    current_strategy = config['value']
                    break

        with st.form("load_balance_form"):
            strategy_options = {
                'adaptive': '自适应策略',
                'least_used': '最少使用策略',
                'round_robin': '轮流使用策略'
            }

            strategy_descriptions = {
                'adaptive': '根据成功率和响应时间智能选择',
                'least_used': '优先使用请求最少的密钥',
                'round_robin': '按顺序轮流使用'
            }

            strategy = st.selectbox(
                "选择策略",
                options=list(strategy_options.keys()),
                format_func=lambda x: strategy_options[x],
                index=list(strategy_options.keys()).index(current_strategy)
            )

            st.info(strategy_descriptions[strategy])

            if st.form_submit_button("保存策略", type="primary", use_container_width=True):
                st.success(f"策略已更新为: {strategy_options[strategy]}")

    with tab4:
        st.markdown("#### 系统信息")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 服务信息")
            st.text(f"Python: {status_data.get('python_version', 'Unknown').split()[0]}")
            st.text(f"版本: {status_data.get('version', '1.1.0')}")
            st.text(f"模型: {', '.join(status_data.get('models', []))}")

        with col2:
            st.markdown("##### 资源使用")
            st.text(f"内存: {status_data.get('memory_usage_mb', 0):.1f} MB")
            st.text(f"CPU: {status_data.get('cpu_percent', 0):.1f}%")
            st.text(f"运行: {status_data.get('uptime_seconds', 0) // 3600} 小时")

# --- 页脚 ---
st.markdown(
    f"""
    <div style='text-align: center; color: rgba(255, 255, 255, 0.7); font-size: 0.8125rem; margin-top: 4rem; padding: 2rem 1rem; border-top: 1px solid rgba(255, 255, 255, 0.15); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); background: rgba(255, 255, 255, 0.05); border-radius: 16px 16px 0 0; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3); width: 100%; max-width: 100%; box-sizing: border-box; overflow-wrap: break-word; word-wrap: break-word;'>
        <a href='{API_BASE_URL}/health' target='_blank' style='color: rgba(255, 255, 255, 0.8); text-decoration: none; transition: all 0.3s ease; padding: 0.25rem 0.5rem; border-radius: 6px; backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px);' onmouseover='this.style.color="white"; this.style.background="rgba(255, 255, 255, 0.1)"; this.style.textShadow="0 0 8px rgba(255, 255, 255, 0.5)";' onmouseout='this.style.color="rgba(255, 255, 255, 0.8)"; this.style.background="transparent"; this.style.textShadow="none";'>健康检查</a> · 
        <span style='color: rgba(255, 255, 255, 0.6); word-break: break-all;'>{API_BASE_URL}</span> ·
        <span style='color: rgba(255, 255, 255, 0.6);'>v1.2</span>
    </div>
    """,
    unsafe_allow_html=True
)