import streamlit as st
import requests
import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

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


# --- 自动清理功能函数 ---
def get_cleanup_status():
    """获取自动清理状态"""
    return call_api('/admin/cleanup/status')


def update_cleanup_config(config_data):
    """更新自动清理配置"""
    return call_api('/admin/cleanup/config', 'POST', config_data)


def manual_cleanup():
    """手动执行清理"""
    return call_api('/admin/cleanup/manual', 'POST')


# --- 故障转移配置函数 ---
def get_failover_config():
    """获取故障转移配置"""
    return call_api('/admin/config/failover')


def update_failover_config(config_data):
    """更新故障转移配置"""
    return call_api('/admin/config/failover', 'POST', config_data)


def get_failover_stats():
    """获取故障转移统计信息"""
    return call_api('/admin/failover/stats')


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


@st.cache_data(ttl=60)
def get_cached_cleanup_status():
    """获取缓存的自动清理状态"""
    return get_cleanup_status()


@st.cache_data(ttl=30)
def get_cached_failover_config():
    """获取缓存的故障转移配置"""
    return get_failover_config()


@st.cache_data(ttl=60)
def get_cached_failover_stats():
    """获取缓存的故障转移统计"""
    return get_failover_stats()


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


def update_user_key_config(key_id: int, config_data: Dict) -> bool:
    """更新用户密钥配置"""
    endpoint = f'/admin/keys/user/{key_id}/config'
    result = call_api(endpoint, 'POST', data=config_data)
    return result and result.get('success', False)


def delete_unhealthy_gemini_keys() -> Optional[Dict]:
    """一键删除所有异常的Gemini密钥"""
    endpoint = '/admin/keys/gemini/unhealthy'
    result = call_api(endpoint, 'DELETE', timeout=60)
    return result


def get_health_status_color(health_status: str) -> str:
    """获取健康状态颜色"""
    status_colors = {
        'healthy': '#10b981',  # 绿色
        'unhealthy': '#ef4444',  # 红色
        'unknown': '#f59e0b'  # 黄色
    }
    return status_colors.get(health_status, '#6b7280')  # 默认灰色


def format_health_status(status):
    status_map = {
        'healthy': '健康',
        'unhealthy': '异常',
        'unknown': '未知',
        'rate_limited': '速率限制',
        'auto_removed': '自动移除'
    }
    return status_map.get(status, '未知')

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


@st.cache_data(ttl=60)
def get_hourly_stats():
    """获取过去24小时的每小时统计数据"""
    return call_api('/admin/stats/hourly')


@st.cache_data(ttl=60)
def get_recent_logs(limit: int = 100):
    """获取最近的请求日志"""
    return call_api(f'/admin/logs/recent?limit={limit}')


@st.cache_data(ttl=60)
def get_cached_deepthink_config():
    return call_api('/admin/config/deepthink', 'GET')

def update_deepthink_config(data):
    return call_api('/admin/config/deepthink', 'POST', data)

@st.cache_data(ttl=60)
def get_cached_search_config():
    return call_api('/admin/config/search', 'GET')

def update_search_config(data):
    return call_api('/admin/config/search', 'POST', data)
