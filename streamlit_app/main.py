import streamlit as st
from app_styling import apply_styling
from app_pages import (
    render_dashboard_page,
    render_key_management_page,
    render_model_config_page,
    render_system_settings_page,
)
from app_utils import get_service_status, API_BASE_URL

# --- 页面配置 ---
st.set_page_config(
    page_title="Gemini API Proxy",
    page_icon="🌠",
    layout="wide",
    initial_sidebar_state="auto"
)

def render_sidebar():
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
            st.markdown(f'''\
            <div class="sidebar-status-card">\
                <div class="sidebar-status-title">API 密钥</div>\
                <div class="sidebar-status-content">\
                    <div class="sidebar-status-text">{service_status['healthy_keys']} / {service_status['active_keys']} 正常</div>\
                </div>\
            </div>\
            ''', unsafe_allow_html=True)


        st.markdown('</div>', unsafe_allow_html=True)

        # 底部信息
        st.markdown(f'''
        <div class="sidebar-footer">
            <div class="sidebar-footer-content">
                <div class="sidebar-footer-item">
                    <span>版本 v2.0</span>
                </div>
                <div class="sidebar-footer-item">
                    <a href="{API_BASE_URL}/docs" target="_blank" class="sidebar-footer-link">API 文档</a>
                    <span>·</span>
                    <a href="https://github.com/arain119" target="_blank" class="sidebar-footer-link">GitHub</a>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    return page

def render_footer():
    st.markdown(
        f"""
        <div style='text-align: center; color: rgba(255, 255, 255, 0.7); font-size: 0.8125rem; margin-top: 4rem; padding: 2rem 0; border-top: 1px solid rgba(255, 255, 255, 0.15); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); background: rgba(255, 255, 255, 0.05); border-radius: 16px 16px 0 0; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);'>
            <a href='{API_BASE_URL}/health' target='_blank' style='color: rgba(255, 255, 255, 0.8); text-decoration: none; transition: all 0.3s ease; padding: 0.25rem 0.5rem; border-radius: 6px; backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px);' onmouseover='this.style.color="white"; this.style.background="rgba(255, 255, 255, 0.1)"; this.style.textShadow="0 0 8px rgba(255, 255, 255, 0.5)";' onmouseout='this.style.color="rgba(255, 255, 255, 0.8)"; this.style.background="transparent"; this.style.textShadow="none";'>健康检查</a> · 
            <span style='color: rgba(255, 255, 255, 0.6);'>{API_BASE_URL}</span> ·
            <span style='color: rgba(255, 255, 255, 0.6);'>v2.0</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# 应用CSS样式
apply_styling()

# 渲染侧边栏并获取当前页面
page = render_sidebar()

# 根据选择的页面渲染主内容
if page == "控制台":
    render_dashboard_page()
elif page == "密钥管理":
    render_key_management_page()
elif page == "模型配置":
    render_model_config_page()
elif page == "系统设置":
    render_system_settings_page()

# 渲染页脚
render_footer()
