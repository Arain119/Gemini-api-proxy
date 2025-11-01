import sqlite3
import json
import secrets
import string
import os
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import threading
from contextlib import contextmanager
import logging

# 配置日志
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = None, db_queue=None):
        self.db_queue = db_queue
        # Render 环境下使用持久化路径
        if db_path is None:
            if os.getenv('RENDER_EXTERNAL_URL'):
                # Render 环境
                db_path = "/opt/render/project/src/gemini_proxy.db"
                # 确保目录存在
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
            else:
                # 本地环境
                db_path = "gemini_proxy.db"

        self.db_path = db_path

        # 缓存结构，减少高频查询的数据库压力
        self._config_cache: Dict[str, Optional[str]] = {}
        self._config_cache_lock = threading.RLock()

        self._model_config_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._model_configs_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
        self._model_cache_lock = threading.RLock()
        self._model_cache_ttl = 5.0  # 秒

        self._available_keys_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
        self._available_keys_cache_lock = threading.RLock()
        self._available_keys_cache_ttl = 1.0  # 秒

        # 初始化数据库
        self.init_db()

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _invalidate_config_cache(self, key: Optional[str] = None) -> None:
        with self._config_cache_lock:
            if key is None:
                self._config_cache.clear()
            else:
                self._config_cache.pop(key, None)

    def _invalidate_model_cache(self, model_name: Optional[str] = None) -> None:
        with self._model_cache_lock:
            if model_name is None:
                self._model_config_cache.clear()
            else:
                self._model_config_cache.pop(model_name, None)
            self._model_configs_cache = None

    def _invalidate_available_keys_cache(self) -> None:
        with self._available_keys_cache_lock:
            self._available_keys_cache = None

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # 设置WAL模式以提高并发性能
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=1000")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 系统配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Gemini Keys表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gemini_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    source_type TEXT DEFAULT 'cli_api_key' NOT NULL,
                    metadata TEXT DEFAULT '{}' NOT NULL,
                    status INTEGER DEFAULT 1,
                    health_status TEXT DEFAULT 'unknown',
                    consecutive_failures INTEGER DEFAULT 0,
                    last_check_time TIMESTAMP,
                    success_rate REAL DEFAULT 1.0,
                    avg_response_time REAL DEFAULT 0.0,
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    breaker_status TEXT DEFAULT 'active' NOT NULL,
                    last_failure_timestamp INTEGER DEFAULT 0 NOT NULL,
                    ema_success_rate REAL DEFAULT 1.0 NOT NULL,
                    ema_response_time REAL DEFAULT 0.0 NOT NULL
                )
            ''')

            # 健康检测历史记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_check_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gemini_key_id INTEGER NOT NULL,
                    check_date DATE NOT NULL,
                    is_healthy BOOLEAN NOT NULL,
                    total_checks INTEGER DEFAULT 1,
                    failed_checks INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0,
                    avg_response_time REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (gemini_key_id) REFERENCES gemini_keys (id),
                    UNIQUE(gemini_key_id, check_date)
                )
            ''')

            # 模型配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE NOT NULL,
                    display_name TEXT UNIQUE,
                    single_api_rpm_limit INTEGER NOT NULL,
                    single_api_tpm_limit INTEGER NOT NULL,
                    single_api_rpd_limit INTEGER NOT NULL,
                    status INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 用户API Keys表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    name TEXT,
                    status INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    tpm_limit INTEGER DEFAULT -1,
                    rpd_limit INTEGER DEFAULT -1,
                    rpm_limit INTEGER DEFAULT -1,
                    valid_until TIMESTAMP DEFAULT NULL,
                    max_concurrency INTEGER DEFAULT -1
                )
            ''')

            # 使用记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gemini_key_id INTEGER,
                    user_key_id INTEGER,
                    model_name TEXT,
                    requests INTEGER DEFAULT 0,
                    tokens INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'success',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (gemini_key_id) REFERENCES gemini_keys (id),
                    FOREIGN KEY (user_key_id) REFERENCES user_keys (id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cli_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    account_email TEXT,
                    credentials TEXT NOT NULL,
                    status INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                )
            ''')

            # 检查并迁移旧表结构
            self._migrate_database(cursor)

            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_logs(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_gemini_key_status ON gemini_keys(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_gemini_key_health ON gemini_keys(health_status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_configs_name ON model_configs(model_name)')

            # 新增索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_history_date ON health_check_history(check_date)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_health_history_key_date ON health_check_history(gemini_key_id, check_date)')

            # 初始化系统配置
            self._init_system_config(cursor)

            # 初始化模型配置
            self._init_model_configs(cursor)

            conn.commit()

    def _format_gemini_row(self, row: sqlite3.Row) -> Dict:
        data = dict(row)
        metadata = data.get('metadata')
        if isinstance(metadata, str):
            try:
                data['metadata'] = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON for gemini key %s", data.get('id'))
                data['metadata'] = {}
        elif metadata is None:
            data['metadata'] = {}
        return data

    def _migrate_database(self, cursor):
        """迁移旧的数据库结构并添加新字段"""
        try:
            # 检查gemini_keys表是否有新字段
            cursor.execute("PRAGMA table_info(gemini_keys)")
            columns = [column[1] for column in cursor.fetchall()]

            # 添加健康检测相关字段
            if 'health_status' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN health_status TEXT DEFAULT 'unknown'")
            if 'consecutive_failures' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN consecutive_failures INTEGER DEFAULT 0")
            if 'last_check_time' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN last_check_time TIMESTAMP")
            if 'success_rate' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN success_rate REAL DEFAULT 1.0")
            if 'avg_response_time' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN avg_response_time REAL DEFAULT 0.0")
            if 'total_requests' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN total_requests INTEGER DEFAULT 0")
            if 'successful_requests' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN successful_requests INTEGER DEFAULT 0")
            if 'source_type' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN source_type TEXT DEFAULT 'cli_api_key' NOT NULL")
            else:
                cursor.execute("UPDATE gemini_keys SET source_type='cli_api_key' WHERE source_type='api_key'")
            if 'metadata' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN metadata TEXT DEFAULT '{}' NOT NULL")
            if 'breaker_status' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN breaker_status TEXT DEFAULT 'active' NOT NULL")
            if 'last_failure_timestamp' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN last_failure_timestamp INTEGER DEFAULT 0 NOT NULL")
            if 'ema_success_rate' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN ema_success_rate REAL DEFAULT 1.0 NOT NULL")
            if 'ema_response_time' not in columns:
                cursor.execute("ALTER TABLE gemini_keys ADD COLUMN ema_response_time REAL DEFAULT 0.0 NOT NULL")

            # 检查usage_logs表是否有status字段
            cursor.execute("PRAGMA table_info(usage_logs)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'status' not in columns:
                cursor.execute("ALTER TABLE usage_logs ADD COLUMN status TEXT DEFAULT 'success'")

            # 检查user_keys表是否有新字段
            cursor.execute("PRAGMA table_info(user_keys)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'tpm_limit' not in columns:
                cursor.execute("ALTER TABLE user_keys ADD COLUMN tpm_limit INTEGER DEFAULT -1")
            if 'rpd_limit' not in columns:
                cursor.execute("ALTER TABLE user_keys ADD COLUMN rpd_limit INTEGER DEFAULT -1")
            if 'rpm_limit' not in columns:
                cursor.execute("ALTER TABLE user_keys ADD COLUMN rpm_limit INTEGER DEFAULT -1")
            if 'valid_until' not in columns:
                cursor.execute("ALTER TABLE user_keys ADD COLUMN valid_until TIMESTAMP DEFAULT NULL")
            if 'max_concurrency' not in columns:
                cursor.execute("ALTER TABLE user_keys ADD COLUMN max_concurrency INTEGER DEFAULT -1")


            # 检查model_configs表结构
            cursor.execute("PRAGMA table_info(model_configs)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'display_name' not in columns:
                cursor.execute("ALTER TABLE model_configs ADD COLUMN display_name TEXT")
                cursor.execute("UPDATE model_configs SET display_name = model_name WHERE display_name IS NULL")

            if 'rpm_limit' in columns:
                # 需要迁移到新的单API限制结构
                logger.info("正在迁移模型配置数据库结构...")

                # 获取旧数据
                cursor.execute("SELECT * FROM model_configs")
                old_configs = cursor.fetchall()

                # 备份旧表
                cursor.execute("ALTER TABLE model_configs RENAME TO model_configs_old")

                # 创建新表
                cursor.execute('''
                    CREATE TABLE model_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT UNIQUE NOT NULL,
                        single_api_rpm_limit INTEGER NOT NULL,
                        single_api_tpm_limit INTEGER NOT NULL,
                        single_api_rpd_limit INTEGER NOT NULL,
                        status INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # 迁移数据
                for old_config in old_configs:
                    cursor.execute('''
                        INSERT INTO model_configs (id, model_name, single_api_rpm_limit, single_api_tpm_limit, single_api_rpd_limit, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (old_config['id'], old_config['model_name'],
                          old_config.get('rpm_limit', 1000), old_config.get('tpm_limit', 2000000),
                          old_config.get('rpd_limit', 50000),
                          old_config['status'], old_config['created_at'], old_config['updated_at']))

                logger.info("模型配置数据库迁移完成")

            # 移除旧的conversations表（如果存在）
            cursor.execute("DROP TABLE IF EXISTS conversations")

        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            # 继续执行，不让迁移失败阻止服务启动

    @staticmethod
    def _env_keep_alive_default() -> str:
        """获取 Keep-Alive 的环境变量默认值"""
        value = os.getenv('ENABLE_KEEP_ALIVE')
        if value is None:
            value = 'true' if os.getenv('RENDER') else 'false'
        return str(value).lower()

    def _init_system_config(self, cursor):
        """初始化系统配置"""
        default_keep_alive = self._env_keep_alive_default()
        default_configs = [
            ('keep_alive_enabled', default_keep_alive, '是否启用 Keep-Alive 与后台调度任务'),
            ('default_model_name', 'gemini-2.5-flash-lite', '默认模型名称'),
            ('max_retries', '3', 'API请求最大重试次数（已废弃，保留兼容性）'),
            ('request_timeout', '60', 'API请求超时时间（秒）'),
            ('load_balance_strategy', 'adaptive', '负载均衡策略: least_used, round_robin, adaptive'),

            # 快速故障转移配置
            ('fast_failover_enabled', 'true', '是否启用快速故障转移'),

            ('single_key_retry', 'false', '单个Key是否进行重试'),
            ('background_health_check', 'true', '是否启用后台健康检测'),
            ('health_check_delay', '5', '失败后健康检测延迟时间（秒）'),

            # 健康检测配置
            ('health_check_enabled', 'true', '是否启用健康检测'),
            ('health_check_interval', '300', '健康检测间隔（秒）'),
            ('failure_threshold', '3', '连续失败阈值'),

            # 思考功能配置
            ('thinking_enabled', 'true', '是否启用思考功能'),
            ('thinking_budget', '-1', '思考预算（token数）：-1=自动，0=禁用，1-32768=固定预算'),
            ('include_thoughts', 'true', '是否在响应中包含思考过程'),

            # 注入prompt配置
            ('inject_prompt_enabled', 'false', '是否启用注入prompt'),
            ('inject_prompt_content', '', '注入的prompt内容'),
            ('inject_prompt_position', 'system', '注入位置: system, user_prefix, user_suffix'),

            # 自动清理配置
            ('auto_cleanup_enabled', 'false', '是否启用自动清理异常API key'),
            ('auto_cleanup_days', '3', '连续异常天数阈值'),
            ('min_checks_per_day', '5', '每日最少检测次数'),

            # 防自动化检测配置
            ('anti_detection_enabled', 'true', '是否启用防自动化检测'),

            # 防截断配置
            ('anti_truncation_enabled', 'false', '是否启用防截断功能'),

            # 响应解密配置
            ('enable_response_decryption', 'false', '是否启用响应自动解密'),
            
            # 流式模式配置
            ('stream_mode', 'auto', 'Stream mode setting: auto, stream, non_stream'),
            ('deepthink_enabled', 'false', 'Enable DeepThink mode for multi-step reasoning'),
            ('deepthink_concurrency', '3', 'Default concurrency for DeepThink mode (3-7)'),

            # 搜索功能配置
            ('search_enabled', 'false', '启用搜索模式进行网页抓取'),
            ('search_num_queries', '3', '搜索查询次数'),
            ('search_num_pages_per_query', '3', '每次查询的页面数'),
        ]

        for key, value, description in default_configs:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO system_config (key, value, description)
                    VALUES (?, ?, ?)
                ''', (key, value, description))
            except Exception as e:
                logger.error(f"Failed to insert config {key}: {e}")

    def _init_model_configs(self, cursor):
        """初始化模型配置（单个API限制）"""
        default_models = [
            ('gemini-2.5-flash', 10, 250000, 250),  # 单API: RPM, TPM, RPD
            ('gemini-2.5-flash-lite', 15, 250000, 1000),  # 单API: RPM, TPM, RPD
            ('gemini-2.5-pro', 5, 250000, 100),  # 单API: RPM, TPM, RPD
            ('gemini-embedding-001', 100, 30000, 1000), # 单API: RPM, TPM, RPD
        ]

        for model_name, rpm, tpm, rpd in default_models:
            try:
                cursor.execute('''
                    INSERT INTO model_configs (model_name, single_api_rpm_limit, single_api_tpm_limit, single_api_rpd_limit)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_name) DO UPDATE SET
                        single_api_rpm_limit=excluded.single_api_rpm_limit,
                        single_api_tpm_limit=excluded.single_api_tpm_limit,
                        single_api_rpd_limit=excluded.single_api_rpd_limit
                ''', (model_name, rpm, tpm, rpd))
            except Exception as e:
                logger.error(f"Failed to insert model config {model_name}: {e}")

    # 系统配置管理
    def get_config(self, key: str, default: str = None) -> str:
        """获取系统配置"""
        with self._config_cache_lock:
            if key in self._config_cache:
                return self._config_cache[key]

        value = default
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM system_config WHERE key = ?', (key,))
                row = cursor.fetchone()
                if row:
                    value = row['value']
        except Exception as e:
            logger.error(f"Failed to get config {key}: {e}")

        with self._config_cache_lock:
            self._config_cache[key] = value

        return value

    def set_config(self, key: str, value: str) -> bool:
        """设置系统配置"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO system_config (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (key, value))
                conn.commit()
                with self._config_cache_lock:
                    self._config_cache[key] = value
                return True
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False

    def get_all_configs(self) -> List[Dict]:
        """获取所有系统配置"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM system_config ORDER BY key')
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get all configs: {e}")
            return []

    def get_thinking_config(self) -> Dict[str, any]:
        """获取思考配置"""
        return {
            'enabled': self.get_config('thinking_enabled', 'true').lower() == 'true',
            'budget': int(self.get_config('thinking_budget', '-1')),
            'include_thoughts': self.get_config('include_thoughts', 'true').lower() == 'true'
        }

    def set_thinking_config(self, enabled: bool = None, budget: int = None, include_thoughts: bool = None) -> bool:
        """设置思考配置"""
        try:
            if enabled is not None:
                self.set_config('thinking_enabled', 'true' if enabled else 'false')

            if budget is not None:
                if not (-1 <= budget <= 32768):
                    raise ValueError("thinking_budget must be between -1 and 32768")
                self.set_config('thinking_budget', str(budget))

            if include_thoughts is not None:
                self.set_config('include_thoughts', 'true' if include_thoughts else 'false')

            return True
        except Exception as e:
            logger.error(f"Failed to set thinking config: {e}")
            return False

    def get_inject_prompt_config(self) -> Dict[str, any]:
        """获取注入prompt配置"""
        return {
            'enabled': self.get_config('inject_prompt_enabled', 'false').lower() == 'true',
            'content': self.get_config('inject_prompt_content', ''),
            'position': self.get_config('inject_prompt_position', 'system')
        }

    def set_inject_prompt_config(self, enabled: bool = None, content: str = None, position: str = None) -> bool:
        """设置注入prompt配置"""
        try:
            if enabled is not None:
                self.set_config('inject_prompt_enabled', 'true' if enabled else 'false')

            if content is not None:
                self.set_config('inject_prompt_content', content)

            if position is not None:
                if position not in ['system', 'user_prefix', 'user_suffix']:
                    raise ValueError("position must be one of: system, user_prefix, user_suffix")
                self.set_config('inject_prompt_position', position)

            return True
        except Exception as e:
            logger.error(f"Failed to set inject prompt config: {e}")
            return False

    # 自动清理配置方法
    def get_auto_cleanup_config(self) -> Dict[str, any]:
        """获取自动清理配置"""
        try:
            return {
                'enabled': self.get_config('auto_cleanup_enabled', 'false').lower() == 'true',
                'days_threshold': int(self.get_config('auto_cleanup_days', '3')),
                'min_checks_per_day': int(self.get_config('min_checks_per_day', '5'))
            }
        except Exception as e:
            logger.error(f"Failed to get auto cleanup config: {e}")
            return {
                'enabled': False,
                'days_threshold': 3,
                'min_checks_per_day': 5
            }

    def set_auto_cleanup_config(self, enabled: bool = None, days_threshold: int = None,
                                min_checks_per_day: int = None) -> bool:
        """设置自动清理配置"""
        try:
            if enabled is not None:
                self.set_config('auto_cleanup_enabled', 'true' if enabled else 'false')

            if days_threshold is not None:
                if not (1 <= days_threshold <= 30):
                    raise ValueError("days_threshold must be between 1 and 30")
                self.set_config('auto_cleanup_days', str(days_threshold))

            if min_checks_per_day is not None:
                if not (1 <= min_checks_per_day <= 100):
                    raise ValueError("min_checks_per_day must be between 1 and 100")
                self.set_config('min_checks_per_day', str(min_checks_per_day))

            return True
        except Exception as e:
            logger.error(f"Failed to set auto cleanup config: {e}")
            return False

    # 流式模式配置方法
    def get_stream_mode_config(self) -> Dict[str, any]:
        """获取流式模式配置"""
        try:
            return {
                'mode': self.get_config('stream_mode', 'auto')
            }
        except Exception as e:
            logger.error(f"Failed to get stream mode config: {e}")
            return {
                'mode': 'auto'
            }

    def set_stream_mode_config(self, mode: str = None) -> bool:
        """设置流式模式配置"""
        try:
            if mode is not None:
                if mode not in ['auto', 'stream', 'non_stream']:
                    raise ValueError("mode must be one of: auto, stream, non_stream")
                self.set_config('stream_mode', mode)

            return True
        except Exception as e:
            logger.error(f"Failed to set stream mode config: {e}")
            return False

    # 与 Gemini 服务通信时的流式模式配置
    def get_stream_to_gemini_mode_config(self) -> Dict[str, any]:
        """获取与 Gemini 通信的流式模式配置"""
        try:
            return {
                'mode': self.get_config('stream_to_gemini_mode', 'auto')
            }
        except Exception as e:
            logger.error(f"Failed to get stream_to_gemini_mode config: {e}")
            return {
                'mode': 'auto'
            }

    def set_stream_to_gemini_mode_config(self, mode: str = None) -> bool:
        """设置与 Gemini 通信的流式模式配置"""
        try:
            if mode is not None:
                if mode not in ['auto', 'stream', 'non_stream']:
                    raise ValueError("mode must be one of: auto, stream, non_stream")
                self.set_config('stream_to_gemini_mode', mode)

            return True
        except Exception as e:
            logger.error(f"Failed to set stream_to_gemini_mode config: {e}")
            return False

    # 故障转移配置方法
    def get_failover_config(self) -> Dict[str, any]:
        """获取故障转移配置"""
        try:
            return {
                'fast_failover_enabled': self.get_config('fast_failover_enabled', 'true').lower() == 'true',

                'background_health_check': self.get_config('background_health_check', 'true').lower() == 'true',
                'health_check_delay': int(self.get_config('health_check_delay', '5')),
            }
        except Exception as e:
            logger.error(f"Failed to get failover config: {e}")
            return {
                'fast_failover_enabled': True,

                'background_health_check': True,
                'health_check_delay': 5,
            }

    def set_failover_config(self, fast_failover_enabled: bool = None,
                            background_health_check: bool = None, health_check_delay: int = None) -> bool:
        """设置故障转移配置"""
        try:
            if fast_failover_enabled is not None:
                self.set_config('fast_failover_enabled', 'true' if fast_failover_enabled else 'false')



            if background_health_check is not None:
                self.set_config('background_health_check', 'true' if background_health_check else 'false')

            if health_check_delay is not None:
                if not (1 <= health_check_delay <= 60):
                    raise ValueError("health_check_delay must be between 1 and 60 seconds")
                self.set_config('health_check_delay', str(health_check_delay))

            return True
        except Exception as e:
            logger.error(f"Failed to set failover config: {e}")
            return False

    # Keep-Alive 配置方法
    def get_keep_alive_config(self) -> Dict[str, any]:
        """获取 Keep-Alive 配置"""
        try:
            default_value = self._env_keep_alive_default()
            return {
                'enabled': self.get_config('keep_alive_enabled', default_value).lower() == 'true'
            }
        except Exception as e:
            logger.error(f"Failed to get keep-alive config: {e}")
            default_enabled = self._env_keep_alive_default() == 'true'
            return {
                'enabled': default_enabled
            }

    def set_keep_alive_config(self, enabled: bool = None) -> bool:
        """设置 Keep-Alive 配置"""
        try:
            if enabled is not None:
                self.set_config('keep_alive_enabled', 'true' if enabled else 'false')
            return True
        except Exception as e:
            logger.error(f"Failed to set keep-alive config: {e}")
            return False

    # 防自动化检测配置方法
    def get_anti_detection_config(self) -> Dict[str, any]:
        """获取防自动化检测配置"""
        try:
            return {
                'enabled': self.get_config('anti_detection_enabled', 'true').lower() == 'true'
            }
        except Exception as e:
            logger.error(f"Failed to get anti detection config: {e}")
            return {
                'enabled': True
            }

    def set_anti_detection_config(self, enabled: bool = None) -> bool:
        """设置防自动化检测配置"""
        try:
            if enabled is not None:
                self.set_config('anti_detection_enabled', 'true' if enabled else 'false')

            return True
        except Exception as e:
            logger.error(f"Failed to set anti detection config: {e}")
            return False

    # 防截断配置方法
    def get_anti_truncation_config(self) -> Dict[str, any]:
        """获取防截断配置"""
        try:
            return {
                'enabled': self.get_config('anti_truncation_enabled', 'false').lower() == 'true'
            }
        except Exception as e:
            logger.error(f"Failed to get anti truncation config: {e}")
            return {
                'enabled': False
            }

    def set_anti_truncation_config(self, enabled: bool = None) -> bool:
        """设置防截断配置"""
        try:
            if enabled is not None:
                self.set_config('anti_truncation_enabled', 'true' if enabled else 'false')

            return True
        except Exception as e:
            logger.error(f"Failed to set anti truncation config: {e}")
            return False

    # 响应解密配置方法
    def get_response_decryption_config(self) -> Dict[str, any]:
        """获取响应解密配置"""
        try:
            return {
                'enabled': self.get_config('enable_response_decryption', 'false').lower() == 'true'
            }
        except Exception as e:
            logger.error(f"Failed to get response decryption config: {e}")
            return {'enabled': False}

    def set_response_decryption_config(self, enabled: bool = None) -> bool:
        """设置响应解密配置"""
        try:
            if enabled is not None:
                self.set_config('enable_response_decryption', 'true' if enabled else 'false')
            return True
        except Exception as e:
            logger.error(f"Failed to set response decryption config: {e}")
            return False

    # DeepThink配置方法
    def get_deepthink_config(self) -> Dict[str, any]:
        """获取DeepThink配置"""
        try:
            return {
                'enabled': self.get_config('deepthink_enabled', 'false').lower() == 'true',
                'concurrency': int(self.get_config('deepthink_concurrency', '3'))
            }
        except Exception as e:
            logger.error(f"Failed to get deepthink config: {e}")
            return {'enabled': False, 'concurrency': 3}

    def set_deepthink_config(self, enabled: bool = None, concurrency: int = None) -> bool:
        """设置DeepThink配置"""
        try:
            if enabled is not None:
                self.set_config('deepthink_enabled', 'true' if enabled else 'false')
            
            if concurrency is not None:
                if not (3 <= concurrency <= 7):
                    raise ValueError("Concurrency must be between 3 and 7")
                self.set_config('deepthink_concurrency', str(concurrency))

            return True
        except Exception as e:
            logger.error(f"Failed to set deepthink config: {e}")
            return False

    # 搜索配置方法
    def get_search_config(self) -> Dict[str, any]:
        """获取搜索配置"""
        try:
            return {
                'enabled': self.get_config('search_enabled', 'false').lower() == 'true',
                'num_queries': int(self.get_config('search_num_queries', '3')),
                'num_pages_per_query': int(self.get_config('search_num_pages_per_query', '3')),
            }
        except Exception as e:
            logger.error(f"Failed to get search config: {e}")
            return {'enabled': False, 'num_queries': 3, 'num_pages_per_query': 3}

    def set_search_config(self, enabled: bool = None, num_queries: int = None, num_pages_per_query: int = None) -> bool:
        """设置搜索配置"""
        try:
            if enabled is not None:
                self.set_config('search_enabled', 'true' if enabled else 'false')
            if num_queries is not None:
                self.set_config('search_num_queries', str(num_queries))
            if num_pages_per_query is not None:
                self.set_config('search_num_pages_per_query', str(num_pages_per_query))

            return True
        except Exception as e:
            logger.error(f"Failed to set search config: {e}")
            return False
            
    def _get_effective_active_key_count(self) -> int:
        """根据当前状态计算可用于容量估算的API Key数量"""
        try:
            healthy_count = len(self.get_healthy_gemini_keys())
            if healthy_count > 0:
                return healthy_count

            available_count = len(self.get_available_gemini_keys())
            if available_count > 0:
                return available_count

            # 最后退回到所有仍处于启用状态的 Key 数量，避免出现配置为 0 的情况
            return sum(1 for key in self.get_all_gemini_keys() if key.get('status') == 1)
        except Exception as e:
            logger.error(f"Failed to calculate effective key count: {e}")
            return 0

    # 模型配置管理
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro', 'gemini-embedding-001']

    def get_model_config(self, model_name: str) -> Optional[Dict]:
        """获取模型配置（包含计算的总限制）"""
        now = time.time()
        with self._model_cache_lock:
            cached = self._model_config_cache.get(model_name)
            if cached and now - cached[0] < self._model_cache_ttl:
                return dict(cached[1])

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM model_configs 
                    WHERE model_name = ? AND status = 1
                ''', (model_name,))
                row = cursor.fetchone()

                if not row:
                    return None

                config = dict(row)

                effective_keys_count = self._get_effective_active_key_count()

                # 计算总限制
                config['total_rpm_limit'] = config['single_api_rpm_limit'] * effective_keys_count
                config['total_tpm_limit'] = config['single_api_tpm_limit'] * effective_keys_count
                config['total_rpd_limit'] = config['single_api_rpd_limit'] * effective_keys_count

                # 为了兼容原有代码，保留旧字段名
                config['rpm_limit'] = config['total_rpm_limit']
                config['tpm_limit'] = config['total_tpm_limit']
                config['rpd_limit'] = config['total_rpd_limit']

                with self._model_cache_lock:
                    self._model_config_cache[model_name] = (time.time(), dict(config))

                return config
        except Exception as e:
            logger.error(f"Failed to get model config for {model_name}: {e}")
            return None

    def get_all_model_configs(self) -> List[Dict]:
        """获取所有模型配置（包含计算的总限制）"""
        now = time.time()
        with self._model_cache_lock:
            if self._model_configs_cache and now - self._model_configs_cache[0] < self._model_cache_ttl:
                return [dict(item) for item in self._model_configs_cache[1]]

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM model_configs ORDER BY model_name')
                configs = [dict(row) for row in cursor.fetchall()]

                effective_keys_count = self._get_effective_active_key_count()

                # 为每个配置添加总限制
                for config in configs:
                    config['total_rpm_limit'] = config['single_api_rpm_limit'] * effective_keys_count
                    config['total_tpm_limit'] = config['single_api_tpm_limit'] * effective_keys_count
                    config['total_rpd_limit'] = config['single_api_rpd_limit'] * effective_keys_count

                    # 为了兼容原有代码，保留旧字段名
                    config['rpm_limit'] = config['total_rpm_limit']
                    config['tpm_limit'] = config['total_tpm_limit']
                    config['rpd_limit'] = config['total_rpd_limit']

                with self._model_cache_lock:
                    self._model_configs_cache = (time.time(), [dict(item) for item in configs])

                return configs
        except Exception as e:
            logger.error(f"Failed to get all model configs: {e}")
            return []

    def update_model_config(self, model_name: str, **kwargs) -> bool:
        """更新模型配置"""
        try:
            allowed_fields = ['display_name', 'single_api_rpm_limit', 'single_api_tpm_limit', 'single_api_rpd_limit', 'status']
            fields = []
            values = []

            for field, value in kwargs.items():
                if field in allowed_fields:
                    fields.append(f"{field} = ?")
                    values.append(value)

            if not fields:
                return False

            values.append(model_name)
            query = f"UPDATE model_configs SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE model_name = ?"

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
                updated = cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update model config for {model_name}: {e}")
            return False

        if updated:
            self._invalidate_model_cache(model_name)
        return updated

    def is_thinking_model(self, model_name: str) -> bool:
        """检查模型是否支持思考功能"""
        return '2.5' in model_name

    # Gemini Key管理 - 增强版
    def add_gemini_key(self, key: str, *, source_type: str = 'cli_api_key', metadata: Optional[Dict] = None) -> bool:
        """添加Gemini Key"""
        success = False
        try:
            metadata_json = json.dumps(metadata or {})
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO gemini_keys (key, source_type, metadata)
                    VALUES (?, ?, ?)
                ''', (key, source_type, metadata_json))
                conn.commit()
                success = cursor.rowcount > 0
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            logger.error(f"Failed to add Gemini key: {e}")
            return False

        if success:
            self._invalidate_available_keys_cache()
            self._invalidate_model_cache()
        return success

    def update_gemini_key(self, key_id: int, **kwargs):
        """更新Gemini Key"""
        try:
            allowed_fields = ['status', 'health_status', 'consecutive_failures',
                              'last_check_time', 'success_rate', 'avg_response_time',
                              'total_requests', 'successful_requests', 'breaker_status',
                              'last_failure_timestamp', 'ema_success_rate', 'ema_response_time',
                              'source_type', 'metadata', 'key']
            fields = []
            values = []

            for field, value in kwargs.items():
                if field in allowed_fields:
                    if field == 'metadata' and isinstance(value, dict):
                        fields.append(f"{field} = ?")
                        values.append(json.dumps(value))
                    else:
                        fields.append(f"{field} = ?")
                        values.append(value)

            if not fields:
                return False

            values.append(key_id)
            query = f"UPDATE gemini_keys SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
                updated = cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update Gemini key {key_id}: {e}")
            return False

        if updated:
            self._invalidate_available_keys_cache()
            self._invalidate_model_cache()
        return updated

    def delete_gemini_key(self, key_id: int) -> bool:
        """删除Gemini Key"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM gemini_keys WHERE id = ?", (key_id,))
                conn.commit()
                deleted = cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete Gemini key {key_id}: {e}")
            return False

        if deleted:
            self._invalidate_available_keys_cache()
            self._invalidate_model_cache()
        return deleted

    def get_all_gemini_keys(self) -> List[Dict]:
        """获取所有Gemini Keys"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM gemini_keys ORDER BY success_rate DESC, avg_response_time ASC, id ASC")
                return [self._format_gemini_row(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get all Gemini keys: {e}")
            return []

    def get_available_gemini_keys(self) -> List[Dict]:
        """获取所有可用的Gemini Keys (排除了熔断的key)"""
        now = time.time()
        with self._available_keys_cache_lock:
            if self._available_keys_cache and now - self._available_keys_cache[0] < self._available_keys_cache_ttl:
                return [dict(item) for item in self._available_keys_cache[1]]

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # 增加了 breaker_status != 'tripped' 的过滤条件
                # 增加了 ema_success_rate 和 ema_response_time 用于排序和返回
                cursor.execute("""
                    SELECT id, key, source_type, metadata, health_status, success_rate, avg_response_time,
                           ema_success_rate, ema_response_time, consecutive_failures, last_failure_timestamp, status, breaker_status
                    FROM gemini_keys
                    WHERE status = 1 AND breaker_status != 'tripped'
                    ORDER BY
                        CASE health_status
                            WHEN 'healthy' THEN 1
                            WHEN 'untested' THEN 2
                            WHEN 'rate_limited' THEN 3
                            ELSE 4
                        END,
                        ema_success_rate DESC,
                        ema_response_time ASC
                """)
                rows = [self._format_gemini_row(row) for row in cursor.fetchall()]

                with self._available_keys_cache_lock:
                    self._available_keys_cache = (time.time(), [dict(item) for item in rows])

                return rows
        except Exception as e:
            logger.error(f"Failed to get available Gemini keys: {e}")
            return []

    def get_healthy_gemini_keys(self) -> List[Dict]:
        """获取健康的Gemini Keys"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM gemini_keys
                    WHERE status = 1 AND health_status = 'healthy'
                    ORDER BY success_rate DESC, avg_response_time ASC, id ASC
                ''')
                return [self._format_gemini_row(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get healthy Gemini keys: {e}")
            return []

    def get_unhealthy_gemini_keys(self) -> List[Dict]:
        """获取所有异常的Gemini Keys"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM gemini_keys WHERE health_status = 'unhealthy' AND status = 1")
                return [self._format_gemini_row(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get unhealthy Gemini keys: {e}")
            return []

    def toggle_gemini_key_status(self, key_id: int) -> bool:
        """切换Gemini Key状态"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE gemini_keys 
                    SET status = CASE WHEN status = 1 THEN 0 ELSE 1 END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (key_id,))
                conn.commit()
                toggled = cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to toggle Gemini key {key_id} status: {e}")
            return False

        if toggled:
            self._invalidate_available_keys_cache()
            self._invalidate_model_cache()
        return toggled

    def update_gemini_key_status(self, key_id: int, new_status: str):
        """更新Gemini Key的健康状态"""
        allowed_statuses = ['healthy', 'unhealthy', 'untested', 'rate_limited']
        if new_status not in allowed_statuses:
            raise ValueError(f"Invalid status: {new_status}")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE gemini_keys
                SET health_status = ?, last_check_time = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_status, key_id))
            conn.commit()
        self._invalidate_available_keys_cache()
        self._invalidate_model_cache()

    def get_gemini_key_by_id(self, key_id: int) -> Optional[Dict]:
        """根据ID获取Gemini Key"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM gemini_keys WHERE id = ?', (key_id,))
                row = cursor.fetchone()
                return self._format_gemini_row(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get Gemini key {key_id}: {e}")
            return None

    def get_gemini_key_by_value(self, key: str) -> Optional[Dict]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM gemini_keys WHERE key = ?', (key,))
                row = cursor.fetchone()
                return self._format_gemini_row(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get Gemini key by value: {e}")
            return None

    def update_key_performance(self, key_id: int, success: bool, response_time: float = 0.0):
        """更新Key性能指标"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # 获取当前统计
                cursor.execute('''
                    SELECT total_requests, successful_requests, avg_response_time, consecutive_failures
                    FROM gemini_keys WHERE id = ?
                ''', (key_id,))
                row = cursor.fetchone()

                if not row:
                    return False

                total_requests = row['total_requests'] + 1
                successful_requests = row['successful_requests'] + (1 if success else 0)
                success_rate = successful_requests / total_requests if total_requests > 0 else 0.0

                # 计算平均响应时间（简单移动平均）
                current_avg = row['avg_response_time']
                if current_avg == 0:
                    new_avg = response_time
                else:
                    # 使用滑动平均，权重为0.1
                    new_avg = current_avg * 0.9 + response_time * 0.1

                # 更新连续失败次数
                if success:
                    consecutive_failures = 0
                    health_status = 'healthy'
                else:
                    consecutive_failures = row['consecutive_failures'] + 1
                    health_status = 'unhealthy'

                # 更新数据库
                cursor.execute('''
                    UPDATE gemini_keys 
                    SET total_requests = ?, successful_requests = ?, success_rate = ?,
                        avg_response_time = ?, consecutive_failures = ?, health_status = ?,
                        last_check_time = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (total_requests, successful_requests, success_rate, new_avg,
                      consecutive_failures, health_status, key_id))

                conn.commit()
                updated = cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update key performance for {key_id}: {e}")
            return False

        if updated:
            self._invalidate_available_keys_cache()
        return updated

    def get_thinking_models(self) -> List[str]:
        """获取支持思考功能的模型列表"""
        return [model for model in self.get_supported_models() if self.is_thinking_model(model)]

    # 健康检测历史记录方法
    def record_daily_health_status(self, key_id: int, is_healthy: bool, response_time: float = 0.0):
        """记录每日健康状态"""
        try:
            today = datetime.now().date()

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # 检查当日是否已有记录
                cursor.execute('''
                    SELECT total_checks, failed_checks, avg_response_time
                    FROM health_check_history 
                    WHERE gemini_key_id = ? AND check_date = ?
                ''', (key_id, today))

                existing = cursor.fetchone()

                if existing:
                    # 更新现有记录
                    new_total = existing['total_checks'] + 1
                    new_failed = existing['failed_checks'] + (0 if is_healthy else 1)
                    new_success_rate = (new_total - new_failed) / new_total

                    # 计算新的平均响应时间
                    old_avg = existing['avg_response_time']
                    new_avg = (old_avg * existing['total_checks'] + response_time) / new_total

                    cursor.execute('''
                        UPDATE health_check_history 
                        SET total_checks = ?, failed_checks = ?, success_rate = ?,
                            avg_response_time = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE gemini_key_id = ? AND check_date = ?
                    ''', (new_total, new_failed, new_success_rate, new_avg, key_id, today))
                else:
                    # 插入新记录
                    cursor.execute('''
                        INSERT INTO health_check_history 
                        (gemini_key_id, check_date, is_healthy, total_checks, failed_checks, success_rate, avg_response_time)
                        VALUES (?, ?, ?, 1, ?, 1.0, ?)
                    ''', (key_id, today, is_healthy, 0 if is_healthy else 1, response_time))

                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record daily health status for key {key_id}: {e}")

    def get_consecutive_unhealthy_days(self, key_id: int, days_threshold: int = 3) -> int:
        """获取连续异常天数"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # 使用参数化查询避免SQL注入
                cursor.execute('''
                    SELECT check_date, success_rate 
                    FROM health_check_history 
                    WHERE gemini_key_id = ? 
                    AND check_date >= date('now', ? || ' days')
                    ORDER BY check_date DESC
                ''', (key_id, -(days_threshold + 2)))

                records = cursor.fetchall()

                # 如果没有历史记录，返回0（表示没有连续异常天数）
                if not records:
                    logger.debug(f"No health history found for key {key_id}")
                    return 0

                consecutive_days = 0
                for record in records:
                    # 健康阈值：成功率低于10%视为异常
                    if record['success_rate'] < 0.1:
                        consecutive_days += 1
                    else:
                        break

                logger.debug(f"Key {key_id} has {consecutive_days} consecutive unhealthy days")
                return consecutive_days

        except Exception as e:
            logger.error(f"Error getting consecutive unhealthy days for key {key_id}: {e}")
            return 0  # 出错时返回0，不影响功能

    def get_at_risk_keys(self, days_threshold: int = None) -> List[Dict]:
        """获取有风险的API keys"""
        try:
            if days_threshold is None:
                days_threshold = int(self.get_config('auto_cleanup_days', '3'))

            at_risk_keys = []
            available_keys = self.get_all_gemini_keys()

            for key_info in available_keys:
                if key_info['status'] != 1:  # 只检查激活的key
                    continue

                try:
                    consecutive_days = self.get_consecutive_unhealthy_days(key_info['id'], days_threshold)
                    if consecutive_days > 0:
                        at_risk_keys.append({
                            'id': key_info['id'],
                            'key': key_info['key'][:10] + '...',
                            'consecutive_unhealthy_days': consecutive_days,
                            'days_until_removal': max(0, days_threshold - consecutive_days)
                        })
                except Exception as e:
                    logger.error(f"Error checking risk for key {key_info['id']}: {e}")
                    continue

            logger.debug(f"Found {len(at_risk_keys)} at-risk keys")
            return at_risk_keys

        except Exception as e:
            logger.error(f"Error getting at-risk keys: {e}")
            return []  # 出错时返回空列表

    def auto_remove_failed_keys(self, days_threshold: int = None, min_checks_per_day: int = None) -> List[Dict]:
        """自动移除连续异常的API key"""
        try:
            if days_threshold is None:
                days_threshold = int(self.get_config('auto_cleanup_days', '3'))
            if min_checks_per_day is None:
                min_checks_per_day = int(self.get_config('min_checks_per_day', '5'))

            removed_keys = []

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # 获取所有激活的key
                cursor.execute('SELECT id, key FROM gemini_keys WHERE status = 1')
                active_keys = cursor.fetchall()

                # 确保至少保留一个健康的key
                healthy_keys = []
                for key_info in active_keys:
                    try:
                        consecutive_days = self.get_consecutive_unhealthy_days(key_info['id'], days_threshold)
                        if consecutive_days == 0:
                            healthy_keys.append(key_info)
                    except Exception as e:
                        logger.error(f"Error checking health for key {key_info['id']}: {e}")
                        continue

                for key_info in active_keys:
                    try:
                        key_id = key_info['id']

                        # 检查连续异常天数
                        consecutive_days = self.get_consecutive_unhealthy_days(key_id, days_threshold)

                        if consecutive_days >= days_threshold:
                            # 确保不会移除所有key
                            if len(healthy_keys) <= 1 and key_info in healthy_keys:
                                logger.warning(f"Skipping removal of key {key_id} to maintain at least one healthy key")
                                continue

                            # 验证每天都有足够的检测次数（避免因检测不足导致误删）
                            cursor.execute('''
                                SELECT check_date, total_checks 
                                FROM health_check_history 
                                WHERE gemini_key_id = ? 
                                AND check_date >= date('now', ? || ' days')
                                ORDER BY check_date DESC
                            ''', (key_id, -days_threshold))

                            recent_records = cursor.fetchall()

                            # 检查是否每天都有足够的检测次数
                            sufficient_checks = all(
                                record['total_checks'] >= min_checks_per_day
                                for record in recent_records[:days_threshold]
                            )

                            if sufficient_checks and len(recent_records) >= days_threshold:
                                # 标记为删除（软删除）
                                cursor.execute('''
                                    UPDATE gemini_keys
                                    SET status = 0, health_status = 'auto_removed',
                                        updated_at = CURRENT_TIMESTAMP
                                    WHERE id = ?
                                ''', (key_id,))

                                removed_keys.append({
                                    'id': key_id,
                                    'key': key_info['key'][:10] + '...',
                                    'consecutive_days': consecutive_days
                                })

                                logger.info(
                                    f"Auto-removed key {key_id} after {consecutive_days} consecutive unhealthy days")

                    except Exception as e:
                        logger.error(f"Error processing key {key_info['id']} for auto removal: {e}")
                        continue

                conn.commit()
                return removed_keys

        except Exception as e:
            logger.error(f"Auto cleanup failed: {e}")
            return []

    # Gemini CLI account management
    def create_cli_account(self, credentials_json: str, account_email: Optional[str] = None, label: Optional[str] = None) -> int:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO cli_accounts (label, account_email, credentials)
                    VALUES (?, ?, ?)
                    ''',
                    (label, account_email, credentials_json),
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to create CLI account: {e}")
            raise

    def update_cli_account_credentials(self, account_id: int, credentials_json: str, account_email: Optional[str] = None) -> bool:
        try:
            fields = ["credentials = ?", "updated_at = CURRENT_TIMESTAMP"]
            values: List[Any] = [credentials_json]
            if account_email:
                fields.insert(1, "account_email = ?")
                values.append(account_email)

            values.append(account_id)
            query = f"UPDATE cli_accounts SET {', '.join(fields)} WHERE id = ?"

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update CLI account {account_id}: {e}")
            return False

    def touch_cli_account(self, account_id: int):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE cli_accounts SET last_used = CURRENT_TIMESTAMP WHERE id = ?",
                    (account_id,),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to update last_used for CLI account {account_id}: {e}")

    def get_cli_account(self, account_id: int) -> Optional[Dict]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM cli_accounts WHERE id = ?', (account_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to load CLI account {account_id}: {e}")
            return None

    def list_cli_accounts(self) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM cli_accounts ORDER BY created_at DESC')
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list CLI accounts: {e}")
            return []

    # 用户Key管理
    def generate_user_key(self, name: str = None) -> str:
        """生成用户Key，自动填补删除的ID"""
        prefix = "sk-"
        length = 48
        characters = string.ascii_letters + string.digits
        random_part = ''.join(secrets.choice(characters) for _ in range(length))
        key = prefix + random_part

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # 查找最小的可用ID（填补空缺）
                cursor.execute('''
                    WITH RECURSIVE seq(x) AS (
                        SELECT 1
                        UNION ALL
                        SELECT x + 1 FROM seq
                        WHERE x < (SELECT COALESCE(MAX(id), 0) + 1 FROM user_keys)
                    )
                    SELECT MIN(x) as next_id
                    FROM seq
                    WHERE x NOT IN (SELECT id FROM user_keys)
                ''')

                result = cursor.fetchone()
                if result and result['next_id']:
                    next_id = result['next_id']
                else:
                    # 如果没有空缺，则使用下一个最大ID
                    cursor.execute('SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM user_keys')
                    next_id = cursor.fetchone()['next_id']

                try:
                    # 使用指定的ID插入
                    cursor.execute('''
                        INSERT INTO user_keys (id, key, name) VALUES (?, ?, ?)
                    ''', (next_id, key, name))
                    conn.commit()
                except sqlite3.IntegrityError:
                    # 如果ID冲突，则使用自动递增
                    cursor.execute('''
                        INSERT INTO user_keys (key, name) VALUES (?, ?)
                    ''', (key, name))
                    conn.commit()

                return key
        except Exception as e:
            logger.error(f"Failed to generate user key: {e}")
            return ""

    def validate_user_key(self, key: str) -> Optional[Dict]:
        """验证用户Key"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM user_keys 
                    WHERE key = ? AND status = 1
                ''', (key,))
                row = cursor.fetchone()

                if row:
                    # 更新最后使用时间
                    cursor.execute('''
                        UPDATE user_keys SET last_used = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    ''', (row['id'],))
                    conn.commit()
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Failed to validate user key: {e}")
            return None

    def get_all_user_keys(self) -> List[Dict]:
        """获取所有用户Keys"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM user_keys ORDER BY id ASC")
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get all user keys: {e}")
            return []

    def toggle_user_key_status(self, key_id: int) -> bool:
        """切换用户Key状态"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE user_keys 
                    SET status = CASE WHEN status = 1 THEN 0 ELSE 1 END 
                    WHERE id = ?
                ''', (key_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to toggle user key {key_id} status: {e}")
            return False

    def delete_user_key(self, key_id: int) -> bool:
        """删除用户Key"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM user_keys WHERE id = ?", (key_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete user key {key_id}: {e}")
            return False

    def get_user_key_by_id(self, key_id: int) -> Optional[Dict]:
        """根据ID获取用户Key"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM user_keys WHERE id = ?', (key_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get user key {key_id}: {e}")
            return None

    def get_user_key_usage_stats(self, user_key_id: int, time_window: str) -> Dict[str, int]:
        """获取指定用户密钥在指定时间窗口内的使用统计"""
        try:
            time_deltas = {
                'minute': timedelta(minutes=1),
                'day': timedelta(days=1)
            }

            if time_window not in time_deltas:
                raise ValueError(f"Invalid time window: {time_window}")

            cutoff_time = datetime.now() - time_deltas[time_window]

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT
                        COALESCE(SUM(requests), 0) as total_requests,
                        COALESCE(SUM(tokens), 0) as total_tokens
                    FROM usage_logs
                    WHERE user_key_id = ? AND timestamp > ?
                ''', (user_key_id, cutoff_time))

                row = cursor.fetchone()
                return {
                    'requests': row['total_requests'],
                    'tokens': row['total_tokens']
                }
        except Exception as e:
            logger.error(f"Failed to get user key usage stats for {user_key_id}: {e}")
            return {'requests': 0, 'tokens': 0}

    def update_user_key(self, key_id: int, **kwargs) -> bool:
        """更新用户Key信息"""
        try:
            allowed_fields = ['name', 'status', 'tpm_limit', 'rpd_limit', 'rpm_limit', 'valid_until', 'max_concurrency']
            fields = []
            values = []

            for field, value in kwargs.items():
                if field in allowed_fields:
                    fields.append(f"{field} = ?")
                    values.append(value)

            if not fields:
                return False

            values.append(key_id)
            query = f"UPDATE user_keys SET {', '.join(fields)} WHERE id = ?"

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update user key {key_id}: {e}")
            return False

    def get_key_usage_stats(self, key_id: int, key_type: str = 'gemini', days: int = 7) -> Dict:
        """获取密钥的使用统计"""
        try:
            column = 'gemini_key_id' if key_type == 'gemini' else 'user_key_id'

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as total_requests,
                        SUM(tokens) as total_tokens,
                        DATE(timestamp) as date
                    FROM usage_logs 
                    WHERE {column} = ? 
                    AND timestamp > datetime('now', '-{days} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                ''', (key_id,))

                daily_stats = [dict(row) for row in cursor.fetchall()]

                # 总计
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as total_requests,
                        SUM(tokens) as total_tokens
                    FROM usage_logs 
                    WHERE {column} = ? 
                    AND timestamp > datetime('now', '-{days} days')
                ''', (key_id,))

                total_stats = dict(cursor.fetchone())

                return {
                    'daily_stats': daily_stats,
                    'total_stats': total_stats
                }
        except Exception as e:
            logger.error(f"Failed to get key usage stats for {key_id}: {e}")
            return {'daily_stats': [], 'total_stats': {'total_requests': 0, 'total_tokens': 0}}

    # 使用记录管理
    def log_usage(self, gemini_key_id: int, user_key_id: int, model_name: str, status: str = 'success', requests: int = 1, tokens: int = 0):
        """Asynchronously log usage by putting it in a queue."""
        if self.db_queue:
            task = {
                "gemini_key_id": gemini_key_id,
                "user_key_id": user_key_id,
                "model_name": model_name,
                "status": status,
                "requests": requests,
                "tokens": tokens,
            }
            try:
                self.db_queue.put_nowait(("log_usage", task))
            except asyncio.QueueFull:
                logger.warning("Database queue is full. Falling back to synchronous logging.")
                self.log_usage_sync(**task)
        else:
            self.log_usage_sync(gemini_key_id, user_key_id, model_name, status, requests, tokens)

    def log_usage_sync(self, gemini_key_id: int, user_key_id: int, model_name: str, status: str = 'success', requests: int = 1, tokens: int = 0):
        """Synchronously log usage to the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO usage_logs (gemini_key_id, user_key_id, model_name, status, requests, tokens)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (gemini_key_id, user_key_id, model_name, status, requests, tokens))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log usage synchronously: {e}")

    def get_usage_stats(self, model_name: str, time_window: str = 'minute') -> Dict[str, int]:
        """获取指定模型在指定时间窗口内的使用统计"""
        try:
            time_deltas = {
                'minute': timedelta(minutes=1),
                'day': timedelta(days=1)
            }

            if time_window not in time_deltas:
                raise ValueError(f"Invalid time window: {time_window}")

            cutoff_time = datetime.now() - time_deltas[time_window]

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        COALESCE(SUM(requests), 0) as total_requests,
                        COALESCE(SUM(tokens), 0) as total_tokens
                    FROM usage_logs
                    WHERE model_name = ? AND timestamp > ?
                ''', (model_name, cutoff_time))

                row = cursor.fetchone()
                return {
                    'requests': row['total_requests'],
                    'tokens': row['total_tokens']
                }
        except Exception as e:
            logger.error(f"Failed to get usage stats for {model_name}: {e}")
            return {'requests': 0, 'tokens': 0}

    def get_all_usage_stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """获取所有模型的使用统计"""
        try:
            stats = {}
            models = self.get_supported_models()

            for model in models:
                stats[model] = {
                    'minute': self.get_usage_stats(model, 'minute'),
                    'day': self.get_usage_stats(model, 'day')
                }

            return stats
        except Exception as e:
            logger.error(f"Failed to get all usage stats: {e}")
            return {}

    def get_hourly_stats_for_last_24_hours(self) -> List[Dict]:
        """获取过去24小时每小时的请求统计"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT
                        strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                        COUNT(*) as total_requests,
                        SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END) as failed_requests
                    FROM usage_logs
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY hour
                    ORDER BY hour
                ''')
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get hourly stats: {e}")
            return []

    def get_recent_usage_logs(self, limit: int = 100) -> List[Dict]:
        """获取最近的使用记录"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT
                        ul.timestamp,
                        ul.model_name,
                        ul.tokens,
                        ul.status,
                        uk.name as user_key_name
                    FROM usage_logs ul
                    LEFT JOIN user_keys uk ON ul.user_key_id = uk.id
                    WHERE ul.timestamp >= datetime('now', '-24 hours')
                    ORDER BY ul.timestamp DESC
                    LIMIT ?
                ''', (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get recent usage logs: {e}")
            return []

    def get_model_usage_rate(self, model_name: str) -> float:
        """获取模型使用率（基于RPM）"""
        try:
            stats = self.get_usage_stats(model_name, 'minute')
            model_config = self.get_model_config(model_name)

            if not model_config or model_config['rpm_limit'] == 0:
                return 0.0

            return stats['requests'] / model_config['rpm_limit']
        except Exception as e:
            logger.error(f"Failed to get model usage rate for {model_name}: {e}")
            return 0.0

    def get_database_stats(self) -> Dict:
        """获取数据库统计信息"""
        stats = {}

        try:
            # 获取数据库文件大小
            if os.path.exists(self.db_path):
                stats['database_size_mb'] = os.path.getsize(self.db_path) / 1024 / 1024
            else:
                stats['database_size_mb'] = 0

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # 获取各表的记录数
                tables = ['system_config', 'gemini_keys', 'model_configs', 'user_keys', 'usage_logs',
                          'health_check_history']
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                        stats[f'{table}_count'] = cursor.fetchone()['count']
                    except Exception as e:
                        logger.error(f"Failed to get count for table {table}: {e}")
                        stats[f'{table}_count'] = 0

                # 获取最近的使用记录
                try:
                    cursor.execute('''
                        SELECT COUNT(*) as recent_usage 
                        FROM usage_logs 
                        WHERE timestamp > datetime('now', '-1 hour')
                    ''')
                    stats['recent_usage_count'] = cursor.fetchone()['recent_usage']
                except Exception as e:
                    logger.error(f"Failed to get recent usage count: {e}")
                    stats['recent_usage_count'] = 0

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats['error'] = str(e)

        return stats

    def cleanup_old_logs(self, days: int = 1) -> int:
        """清理旧的使用日志"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM usage_logs 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                deleted_count = cursor.rowcount
                conn.commit()

            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
            return 0

    def cleanup_old_health_history(self, days: int = 1) -> int:
        """清理旧的健康检测历史"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM health_check_history 
                    WHERE check_date < ?
                ''', (cutoff_date.date(),))
                deleted_count = cursor.rowcount
                conn.commit()

            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old health history: {e}")
            return 0

    def backup_database(self, backup_path: str = None) -> bool:
        """备份数据库"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"gemini_proxy_backup_{timestamp}.db"

            import shutil
            shutil.copy2(self.db_path, backup_path)
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            'database_path': self.db_path,
            'database_exists': os.path.exists(self.db_path),
            'environment': 'render' if os.getenv('RENDER_EXTERNAL_URL') else 'local',
            'stats': self.get_database_stats()
        }

    # 健康检测相关方法
    def get_keys_health_summary(self) -> Dict:
        """获取Keys健康状态汇总"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        health_status,
                        COUNT(*) as count
                    FROM gemini_keys 
                    WHERE status = 1
                    GROUP BY health_status
                ''')

                health_counts = {row['health_status']: row['count'] for row in cursor.fetchall()}

                return {
                    'healthy': health_counts.get('healthy', 0),
                    'unhealthy': health_counts.get('unhealthy', 0),
                    'unknown': health_counts.get('unknown', 0),
                    'total_active': sum(health_counts.values())
                }
        except Exception as e:
            logger.error(f"Failed to get keys health summary: {e}")
            return {
                'healthy': 0,
                'unhealthy': 0,
                'unknown': 0,
                'total_active': 0
            }

    def mark_keys_for_health_check(self) -> List[Dict]:
        """标记需要健康检查的Keys"""
        try:
            health_check_interval = int(self.get_config('health_check_interval', '300'))  # 5分钟
            cutoff_time = datetime.now() - timedelta(seconds=health_check_interval)

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM gemini_keys 
                    WHERE status = 1 
                    AND (last_check_time IS NULL OR last_check_time < ?)
                    ORDER BY last_check_time ASC NULLS FIRST
                ''', (cutoff_time,))

                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to mark keys for health check: {e}")
            return []
