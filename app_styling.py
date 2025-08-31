import streamlit as st

def apply_styling():
    # --- 玻璃拟态风格CSS ---
    st.markdown("""
    <style>
    /* 移动端检测脚本 */
    script {
        display: none;
    }

    /* 全局字体和基础设置 */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro SC", "SF Pro Display", "Helvetica Neue", "PingFang SC", "Microsoft YaHei UI", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* 页面背景 */
    .stApp {
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
        overflow-x: hidden;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 主内容区域 */
    .block-container {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.05),
            0 8px 32px rgba(0, 0, 0, 0.03),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        padding: 1.5rem;
        margin: 1rem;
        max-width: 1440px;
        position: relative;
        overflow: visible;
        min-height: auto;
    }

    /* 媒体查询 */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
            margin: 0.5rem;
            border-radius: 16px;
        }

        /* 隐藏侧边栏按钮 */
        .stSidebar .stButton {
            margin-bottom: 0.5rem;
        }

        /* 标题 */
        h1 {
            font-size: 1.875rem !important;
            margin-bottom: 1rem !important;
        }

        h2 {
            font-size: 1.5rem !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
        }

        h3 {
            font-size: 1.125rem !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }

        /* 卡片间距 */
        [data-testid="metric-container"] {
            margin-bottom: 0.75rem;
            padding: 1rem 1.25rem;
        }

        /* 按钮 */
        .stButton > button {
            width: 100%;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            padding: 0.625rem 1rem;
        }

        /* 表单间距 */
        .stForm {
            margin-bottom: 1rem;
        }

        /* 输入框 */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select,
        .stTextArea > div > div > textarea {
            font-size: 16px !important; /* 防止iOS缩放 */
            padding: 0.75rem 1rem !important;
        }

        /* 标签页 */
        .stTabs [data-testid="stTabBar"] {
            gap: 0.5rem;
            padding: 0;
            margin-bottom: 1rem;
            overflow-x: auto;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }

        .stTabs [data-testid="stTabBar"]::-webkit-scrollbar {
            display: none;
        }

        .stTabs [data-testid="stTabBar"] button {
            padding: 0.875rem 1.25rem;
            font-size: 0.875rem;
            white-space: nowrap;
            min-width: auto;
        }

        /* Alert */
        [data-testid="stAlert"] {
            padding: 0.75rem 1rem !important;
            margin: 0.5rem 0 !important;
            border-radius: 12px !important;
            font-size: 0.875rem !important;
        }
    }

    /* 超小屏幕 */
    @media (max-width: 480px) {
        .block-container {
            padding: 0.75rem;
            margin: 0.25rem;
            border-radius: 12px;
        }

        /* 超小屏幕下的度量卡片 */
        [data-testid="metric-container"] {
            padding: 0.875rem 1rem;
        }

        [data-testid="metric-container"] > div:nth-child(2) {
            font-size: 1.875rem;
        }

        /* 超小屏幕下的按钮 */
        .stButton > button {
            font-size: 0.8125rem;
            padding: 0.5rem 0.875rem;
        }

        /* 超小屏幕下的标题 */
        h1 {
            font-size: 1.5rem !important;
        }

        h2 {
            font-size: 1.25rem !important;
        }

        h3 {
            font-size: 1rem !important;
        }
    }

    .block-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.6) 50%, 
            transparent
        );
    }

    /* 度量卡片玻璃效果 */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 1.5rem 1.75rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.05),
            0 4px 16px rgba(0, 0, 0, 0.03),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.8) 50%, 
            transparent
        );
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.08),
            0 8px 32px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.7);
        border-color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.5);
    }

    /* 移动端触摸 */
    @media (max-width: 768px) {
        /* 卡片悬停动画 */
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px) scale(1.01);
        }

        [data-testid="metric-container"]:active {
            transform: scale(0.98);
            transition: transform 0.15s ease;
        }

        /* 按钮动画 */
        .stButton > button:hover {
            transform: translateY(-1px);
        }

        .stButton > button:active {
            transform: scale(0.98);
            transition: transform 0.1s ease;
        }

        /* 导航项动画 */
        section[data-testid="stSidebar"] .stRadio > div > label:hover {
            transform: translateX(3px);
        }

        section[data-testid="stSidebar"] .stRadio > div > label:active {
            transform: scale(0.98);
            transition: transform 0.1s ease;
        }

        /* Logo动画 */
        .sidebar-logo:hover {
            transform: translateY(-1px) scale(1.01);
        }

        .sidebar-logo:active {
            transform: scale(0.98);
            transition: transform 0.1s ease;
        }

        /* 状态卡片动画 */
        .sidebar-status-card:hover {
            transform: translateY(-1px);
        }

        .sidebar-status-card:active {
            transform: scale(0.98);
            transition: transform 0.1s ease;
        }

        /* 密钥卡片动画 */
        div[data-testid="stHorizontalBlock"]:hover {
            transform: translateY(-1px) scale(1.005);
        }

        div[data-testid="stHorizontalBlock"]:active {
            transform: scale(0.98);
            transition: transform 0.1s ease;
        }

        /* 链接动画 */
        .sidebar-footer-link:hover {
            transform: translateY(-0.5px);
        }

        .sidebar-footer-link:active {
            background: rgba(255, 255, 255, 0.15);
            transform: scale(0.98);
            transition: all 0.1s ease;
        }

        /* 标签页动画 */
        .stTabs [data-testid="stTabBar"] button:hover {
            transform: translateY(-0.5px);
        }

        .stTabs [data-testid="stTabBar"] button:active {
            background: rgba(255, 255, 255, 0.5);
            transform: scale(0.98);
            transition: all 0.1s ease;
        }

        /* 输入框聚焦动画 */
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stTextArea > div > div > textarea:focus {
            transform: translateY(-0.5px);
        }

        /* 状态标签保留动画 */
        .status-badge:hover {
            transform: translateY(-1px) scale(1.02);
        }

        .status-badge:active {
            transform: scale(0.98);
            transition: transform 0.1s ease;
        }
    }

    /* 度量值样式 */
    [data-testid="metric-container"] > div:nth-child(1) {
        font-size: 0.8125rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    [data-testid="metric-container"] > div:nth-child(2) {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1f2937;
        line-height: 1.1;
        background: linear-gradient(135deg, #1f2937 0%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    [data-testid="metric-container"] > div:nth-child(3) {
        font-size: 0.8125rem;
        font-weight: 500;
        margin-top: 0.75rem;
        color: #6b7280;
    }

    /* 侧边栏设计 */
    section[data-testid="stSidebar"] {
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
        box-shadow: 
            4px 0 32px rgba(0, 0, 0, 0.08),
            0 0 0 1px rgba(255, 255, 255, 0.08) inset;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* 移动端侧边栏宽度调整 */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 280px !important;
            z-index: 999;
            transform: translateX(0);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        section[data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(2) {
            padding: 1.5rem 1rem;
            position: relative;
        }
    }

    /* 侧边栏动态背景 */
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 20%, rgba(99, 102, 241, 0.2) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(168, 85, 247, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 60%, rgba(59, 130, 246, 0.18) 0%, transparent 50%);
        opacity: 0.7;
        animation: float 20s ease-in-out infinite alternate;
        pointer-events: none;
    }

    @keyframes float {
        0% { transform: translate(0px, 0px) rotate(0deg); opacity: 0.7; }
        50% { transform: translate(-10px, -10px) rotate(1deg); opacity: 0.9; }
        100% { transform: translate(5px, -5px) rotate(-1deg); opacity: 0.7; }
    }

    /* 侧边栏内容区域 */
    section[data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(2) {
        padding: 2rem 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        position: relative;
        z-index: 2;
    }

    /* Logo区域玻璃效果 */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.875rem;
        padding: 1.25rem 1rem;
        margin-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* 移动端Logo调整 */
    @media (max-width: 768px) {
        .sidebar-logo {
            padding: 1rem 0.875rem;
            margin-bottom: 1rem;
            border-radius: 16px;
        }

        .sidebar-logo-icon {
            font-size: 2rem !important;
        }

        .sidebar-logo-title {
            font-size: 1.125rem !important;
        }

        .sidebar-logo-subtitle {
            font-size: 0.75rem !important;
        }
    }

    .sidebar-logo::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.15) 50%, 
            transparent
        );
        transition: left 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .sidebar-logo:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 
            0 16px 48px rgba(0, 0, 0, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.25);
        background: rgba(255, 255, 255, 0.12);
    }


    .sidebar-logo:hover::before {
        left: 100%;
    }

    .sidebar-logo-icon {
        font-size: 2.5rem;
        line-height: 1;
        filter: drop-shadow(0 0 12px rgba(99, 102, 241, 0.8));
        animation: pulse-glow 3s ease-in-out infinite;
    }

    @keyframes pulse-glow {
        0%, 100% { filter: drop-shadow(0 0 12px rgba(99, 102, 241, 0.8)); }
        50% { filter: drop-shadow(0 0 24px rgba(99, 102, 241, 1)); }
    }

    .sidebar-logo-title {
        font-size: 1.375rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: white;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }

    .sidebar-logo-subtitle {
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.8);
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    }

    /* 玻璃分割线 */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.25) 20%, 
            rgba(255, 255, 255, 0.5) 50%, 
            rgba(255, 255, 255, 0.25) 80%, 
            transparent
        );
        margin: 1.5rem 0;
        position: relative;
    }

    /* 移动端分割线调整 */
    @media (max-width: 768px) {
        .sidebar-divider {
            margin: 1rem 0;
        }
    }

    .sidebar-divider::after {
        content: '';
        position: absolute;
        top: 1px;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.15) 50%, 
            transparent
        );
    }

    /* 导航区域标题 */
    .sidebar-section-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        padding: 0 1rem 0.75rem 1rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
        position: relative;
    }

    /* 移动端导航标题调整 */
    @media (max-width: 768px) {
        .sidebar-section-title {
            font-size: 0.75rem;
            padding: 0 0.75rem 0.5rem 0.75rem;
            margin-bottom: 0.25rem;
        }
    }

    .sidebar-section-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 1rem;
        right: 1rem;
        height: 1px;
        background: linear-gradient(90deg, 
            rgba(255, 255, 255, 0.25), 
            rgba(255, 255, 255, 0.08)
        );
    }

    /* 导航容器 */
    section[data-testid="stSidebar"] .stRadio {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0.5rem !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* 移动端导航间距调整 */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.375rem !important;
        }
    }

    /* 导航项玻璃效果 */
    section[data-testid="stSidebar"] .stRadio > div > label {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.9) !important;
        padding: 1rem 1.25rem !important;
        border-radius: 16px !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.875rem !important;
        margin: 0.375rem 0 !important;
        position: relative !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        width: 100% !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
        -webkit-tap-highlight-color: transparent !important; /* 移除iOS点击高亮 */
    }

    /* 移动端导航项调整 */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] .stRadio > div > label {
            font-size: 0.875rem !important;
            padding: 0.875rem 1rem !important;
            margin: 0.25rem 0 !important;
            border-radius: 14px !important;
        }
    }

    /* 导航项内容发光边框 */
    section[data-testid="stSidebar"] .stRadio > div > label::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 16px;
        padding: 1px;
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.25) 0%, 
            rgba(255, 255, 255, 0.08) 25%,
            transparent 50%,
            rgba(255, 255, 255, 0.08) 75%,
            rgba(255, 255, 255, 0.25) 100%
        );
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: exclude;
        opacity: 0;
        transition: opacity 0.4s ease;
    }

    /* 悬停效果 */
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        color: white !important;
        transform: translateX(6px) translateY(-2px) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        box-shadow: 
            0 12px 32px rgba(0, 0, 0, 0.1),
            0 4px 16px rgba(99, 102, 241, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }


    section[data-testid="stSidebar"] .stRadio > div > label:hover::before {
        opacity: 1;
    }

    /* 选中状态玻璃效果 */
    section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label {
        background: linear-gradient(135deg, 
            rgba(99, 102, 241, 0.3) 0%, 
            rgba(168, 85, 247, 0.25) 50%,
            rgba(99, 102, 241, 0.3) 100%
        ) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        color: white !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 
            0 12px 40px rgba(99, 102, 241, 0.25),
            0 6px 20px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.25),
            inset 0 -1px 0 rgba(0, 0, 0, 0.1) !important;
        transform: translateX(4px) !important;
    }

    /* 移动端选中状态不移动 */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label {
            transform: none !important;
        }
    }

    /* 选中状态发光边框 */
    section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label::after {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        border-radius: 0 2px 2px 0;
        background: linear-gradient(180deg, 
            #6366f1 0%, 
            #a855f7 50%,
            #6366f1 100%
        );
        box-shadow: 
            0 0 16px rgba(99, 102, 241, 1),
            0 0 32px rgba(99, 102, 241, 0.6);
        animation: glow-pulse 2s ease-in-out infinite;
    }

    @keyframes glow-pulse {
        0%, 100% { 
            box-shadow: 
                0 0 16px rgba(99, 102, 241, 1),
                0 0 32px rgba(99, 102, 241, 0.6);
        }
        50% { 
            box-shadow: 
                0 0 24px rgba(99, 102, 241, 1),
                0 0 48px rgba(99, 102, 241, 0.8),
                0 0 64px rgba(99, 102, 241, 0.4);
        }
    }

    /* 隐藏radio按钮 */
    section[data-testid="stSidebar"] .stRadio input[type="radio"] {
        display: none !important;
    }

    /* 状态指示器玻璃卡片 */
    .sidebar-status {
        margin-top: auto;
        padding-top: 1.5rem;
    }

    /* 移动端状态调整 */
    @media (max-width: 768px) {
        .sidebar-status {
            padding-top: 1rem;
        }
    }

    .sidebar-status-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.12);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    /* 移动端状态卡片调整 */
    @media (max-width: 768px) {
        .sidebar-status-card {
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 14px;
        }
    }

    .sidebar-status-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.4) 50%, 
            transparent
        );
    }

    .sidebar-status-card:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
        box-shadow: 
            0 12px 32px rgba(0, 0, 0, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.18);
    }


    .sidebar-status-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.75);
        margin-bottom: 0.5rem;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }

    /* 移动端状态标题调整 */
    @media (max-width: 768px) {
        .sidebar-status-title {
            font-size: 0.75rem;
            margin-bottom: 0.375rem;
        }
    }

    .sidebar-status-content {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .sidebar-status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
        position: relative;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.25);
    }

    .sidebar-status-indicator.online {
        background: #10b981;
        box-shadow: 
            0 0 16px rgba(16, 185, 129, 0.8),
            0 0 0 2px rgba(255, 255, 255, 0.25);
        animation: online-pulse 2s ease-in-out infinite;
    }

    .sidebar-status-indicator.offline {
        background: #ef4444;
        box-shadow: 
            0 0 16px rgba(239, 68, 68, 0.8),
            0 0 0 2px rgba(255, 255, 255, 0.25);
    }

    @keyframes online-pulse {
        0%, 100% { 
            box-shadow: 
                0 0 16px rgba(16, 185, 129, 0.8),
                0 0 0 2px rgba(255, 255, 255, 0.25);
        }
        50% { 
            box-shadow: 
                0 0 24px rgba(16, 185, 129, 1),
                0 0 40px rgba(16, 185, 129, 0.6),
                0 0 0 2px rgba(255, 255, 255, 0.35);
        }
    }

    .sidebar-status-text {
        font-size: 1rem;
        color: white;
        font-weight: 500;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }

    /* 移动端状态文字调整 */
    @media (max-width: 768px) {
        .sidebar-status-text {
            font-size: 0.875rem;
        }
    }

    /* 版本信息 */
    .sidebar-footer {
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        margin-top: 1rem;
        position: relative;
    }

    /* 移动端版本信息调整 */
    @media (max-width: 768px) {
        .sidebar-footer {
            padding-top: 0.75rem;
            margin-top: 0.75rem;
        }
    }

    .sidebar-footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.25) 50%, 
            transparent
        );
    }

    .sidebar-footer-content {
        display: flex;
        flex-direction: column;
        gap: 0.375rem;
        padding: 0 0.5rem;
    }

    /* 移动端版本信息内容调整 */
    @media (max-width: 768px) {
        .sidebar-footer-content {
            gap: 0.25rem;
            padding: 0 0.25rem;
        }
    }

    .sidebar-footer-item {
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.6);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }

    /* 移动端版本信息项调整 */
    @media (max-width: 768px) {
        .sidebar-footer-item {
            font-size: 0.75rem;
            gap: 0.375rem;
        }
    }

    .sidebar-footer-link {
        color: rgba(255, 255, 255, 0.75);
        text-decoration: none;
        transition: all 0.3s ease;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        -webkit-tap-highlight-color: transparent; 
    }

    .sidebar-footer-link:hover {
        color: white;
        background: rgba(255, 255, 255, 0.12);
        text-shadow: 0 0 12px rgba(255, 255, 255, 0.6);
        transform: translateY(-1px);
    }


    /* 按钮玻璃效果 */
    .stButton > button {
        border-radius: 14px;
        font-weight: 500;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        font-size: 0.9375rem;
        padding: 0.75rem 1.5rem;
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
        -webkit-tap-highlight-color: transparent; 
        min-height: 44px; 
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.3) 50%, 
            transparent
        );
        transition: left 0.6s ease;
    }

    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.2);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 12px 36px rgba(99, 102, 241, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        border-color: rgba(99, 102, 241, 0.4);
        color: #4338ca;
    }


    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }

    /* 输入框 */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        background: rgba(249, 250, 251, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        font-size: 0.9375rem !important;
        padding: 0.875rem 1.25rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
        color: #1f2937 !important;
        min-height: 44px !important; 
    }

    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #6b7280 !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        background: rgba(255, 255, 255, 0.8) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 
            0 0 0 3px rgba(99, 102, 241, 0.1),
            0 12px 32px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
        outline: none !important;
        transform: translateY(-1px);
    }


    /* 健康状态标签玻璃效果 */
    .status-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8125rem;
        font-weight: 500;
        line-height: 1;
        white-space: nowrap;
        min-width: 3.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* 移动端状态标签调整 */
    @media (max-width: 768px) {
        .status-badge {
            padding: 0.375rem 0.75rem;
            font-size: 0.75rem;
            min-width: 3rem;
            border-radius: 16px;
        }
    }

    .status-badge:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 
            0 12px 32px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
    }


    .status-healthy {
        background: rgba(16, 185, 129, 0.15);
        color: #065f46;
        border-color: rgba(16, 185, 129, 0.3);
    }

    .status-unhealthy {
        background: rgba(239, 68, 68, 0.15);
        color: #991b1b;
        border-color: rgba(239, 68, 68, 0.3);
    }

    .status-tripped {
        background: rgba(245, 158, 11, 0.15);
        color: #92400e;
        border-color: rgba(245, 158, 11, 0.3);
    }

    .status-unknown {
        background: rgba(107, 114, 128, 0.15);
        color: #374151;
        border-color: rgba(107, 114, 128, 0.3);
    }

    .status-active {
        background: rgba(59, 130, 246, 0.15);
        color: #1e40af;
        border-color: rgba(59, 130, 246, 0.3);
    }

    .status-inactive {
        background: rgba(107, 114, 128, 0.15);
        color: #374151;
        border-color: rgba(107, 114, 128, 0.3);
    }

    /* 密钥卡片玻璃效果 */
    div[data-testid="stHorizontalBlock"] {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 10px 32px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        position: relative;
        overflow: hidden;
    }

    /* 移动端密钥卡片调整 */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 14px;
        }
    }

    div[data-testid="stHorizontalBlock"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.8) 50%, 
            transparent
        );
    }

    div[data-testid="stHorizontalBlock"]:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 
            0 16px 48px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        border-color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.5);
    }


    /* 密钥代码显示 */
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
        word-break: break-all; 
    }

    /* 移动端密钥代码调整 */
    @media (max-width: 768px) {
        .key-code {
            font-size: 0.75rem;
            padding: 0.625rem 0.875rem;
            border-radius: 8px;
        }
    }

    .key-id {
        font-weight: 600;
        color: #374151;
        min-width: 2.5rem;
    }

    /* 移动端密钥ID调整 */
    @media (max-width: 768px) {
        .key-id {
            min-width: 2rem;
            font-size: 0.875rem;
        }
    }

    .key-meta {
        font-size: 0.8125rem;
        color: #6b7280;
        margin-top: 0.375rem;
    }

    /* 移动端密钥元数据调整 */
    @media (max-width: 768px) {
        .key-meta {
            font-size: 0.75rem;
            margin-top: 0.25rem;
        }
    }

    /* 标签页玻璃效果 */
    .stTabs [data-testid="stTabBar"] {
        gap: 1.5rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
        padding: 0;
        margin-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px 16px 0 0;
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-bottom: none;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        overflow-x: auto; 
        scrollbar-width: none;
        -ms-overflow-style: none;
    }

    .stTabs [data-testid="stTabBar"]::-webkit-scrollbar {
        display: none;
    }

    .stTabs [data-testid="stTabBar"] button {
        font-weight: 500;
        color: #6b7280;
        padding: 1rem 1.5rem;
        border-bottom: 2px solid transparent;
        font-size: 0.9375rem;
        letter-spacing: 0.02em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 12px 12px 0 0;
        background: transparent;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        white-space: nowrap; 
        min-width: auto;
        flex-shrink: 0; 
    }

    .stTabs [data-testid="stTabBar"] button:hover {
        background: rgba(255, 255, 255, 0.4);
        color: #374151;
        transform: translateY(-1px);
    }

    /* 移动端标签页悬停优化 */
    @media (max-width: 768px) {
        .stTabs [data-testid="stTabBar"] button:hover {
            transform: none;
        }

        .stTabs [data-testid="stTabBar"] button:active {
            background: rgba(255, 255, 255, 0.5);
        }
    }

    .stTabs [data-testid="stTabBar"] button[aria-selected="true"] {
        color: #1f2937;
        border-bottom-color: #6366f1;
        background: rgba(255, 255, 255, 0.5);
        box-shadow: 
            0 -4px 12px rgba(99, 102, 241, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }

    /* Alert消息玻璃效果 */
    [data-testid="stAlert"] {
        border: none !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
        padding: 1rem 1.25rem !important;
        margin: 0.75rem 0 !important;
    }

    [data-testid="stAlert"][kind="info"] {
        background: rgba(59, 130, 246, 0.1) !important;
        color: #1e40af !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
    }

    [data-testid="stAlert"][kind="success"] {
        background: rgba(16, 185, 129, 0.1) !important;
        color: #065f46 !important;
        border-color: rgba(16, 185, 129, 0.3) !important;
    }

    [data-testid="stAlert"][kind="warning"] {
        background: rgba(245, 158, 11, 0.1) !important;
        color: #92400e !important;
        border-color: rgba(245, 158, 11, 0.3) !important;
    }

    [data-testid="stAlert"][kind="error"] {
        background: rgba(239, 68, 68, 0.1) !important;
        color: #991b1b !important;
        border-color: rgba(239, 68, 68, 0.3) !important;
    }

    /* 图表容器玻璃效果 */
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
        pointer-events: none;
        user-select: none; 
    }

    /* 禁用图表内部元素的交互 */
    .js-plotly-plot .plotly svg,
    .js-plotly-plot .plotly canvas,
    .js-plotly-plot .plotly .plotly-plot,
    .js-plotly-plot .plotly .svg-container {
        pointer-events: none !important;
        touch-action: none !important;
        user-select: none !important;
    }

    /* 移动端图表调整 */
    @media (max-width: 768px) {
        .js-plotly-plot .plotly {
            border-radius: 14px;
        }
    }

    /* 表格玻璃效果 */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        background: rgba(255, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }

    /* 移动端表格调整 */
    @media (max-width: 768px) {
        .stDataFrame {
            border-radius: 12px;
            font-size: 0.875rem;
        }
    }

    /* 标题样式 */
    h1, h2, h3 {
        color: #1f2937;
    }

    h1 {
        background: linear-gradient(135deg, #1f2937 0%, #4f46e5 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.875rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    h3 {
        font-size: 1.25rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* 页面副标题 */
    .page-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* 移动端页面副标题调整 */
    @media (max-width: 768px) {
        .page-subtitle {
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }
    }

    /* 分割线玻璃效果 */
    hr {
        margin: 1.5rem 0 !important;
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(99, 102, 241, 0.3) 20%, 
            rgba(99, 102, 241, 0.5) 50%, 
            rgba(99, 102, 241, 0.3) 80%, 
            transparent
        ) !important;
        position: relative;
    }

    /* 移动端分割线调整 */
    @media (max-width: 768px) {
        hr {
            margin: 1rem 0 !important;
        }
    }

    hr::after {
        content: '';
        position: absolute;
        top: 1px;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.3) 50%, 
            transparent
        );
    }

    .main .block-container {
        max-height: none !important;
        overflow: visible !important;
    }

    .stApp > div {
        overflow: visible !important;
    }

    body {
        overflow-x: hidden;
        overflow-y: auto;
    }

    .stApp {
        overflow-x: hidden;
        overflow-y: auto;
    }

    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    /* 移动端隐藏滚动条 */
    @media (max-width: 768px) {
        ::-webkit-scrollbar {
            width: 0px;
            height: 0px;
        }
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

    /* 选择文本样式 */
    ::selection {
        background: rgba(99, 102, 241, 0.2);
        color: #1f2937;
    }

    ::-moz-selection {
        background: rgba(99, 102, 241, 0.2);
        color: #1f2937;
    }

    /* 移动端顶部导航栏隐藏按钮优化 */
    @media (max-width: 768px) {
        /* 侧边栏切换按钮样式 */
        button[kind="secondary"] {
            background: rgba(99, 102, 241, 0.1) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(99, 102, 241, 0.3) !important;
            border-radius: 12px !important;
            box-shadow: 
                0 4px 16px rgba(99, 102, 241, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
            color: #4f46e5 !important;
            font-weight: 500 !important;
            min-height: 44px !important;
        }
    }


    /* 移动端横屏适配 */
    @media (max-width: 1024px) and (orientation: landscape) {
        .block-container {
            padding: 1rem;
            margin: 0.5rem;
        }

        [data-testid="metric-container"] {
            padding: 1rem 1.25rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        h2 {
            font-size: 1.5rem !important;
        }
    }

    /* 状态卡片样式 */
    .status-card-style {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 
            0 10px 32px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)
