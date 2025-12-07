from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import aiohttp
import json
import os
import numpy as np
import soundfile as sf
from io import BytesIO
import logging
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import requests
import pygame
import io

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="TTS API Router", description="分发TTS请求到多个服务器并汇总结果", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务器配置文件路径
SERVERS_CONFIG_FILE = "servers_config.json"
SETTINGS_FILE = "settings.json"
REFERENCE_AUDIO_CONFIG_FILE = "reference_audio_config.json"
TTSTEXT_CONFIG_FILE = "tts_text_config.json"
AUTO_SAVE_CONFIG_FILE = "auto_save_config.json"

# 初始化默认配置
def init_default_config():
    # 初始化服务器配置
    if not os.path.exists(SERVERS_CONFIG_FILE):
        default_servers = {
            "servers": [
                {"id": 1, "url": "http://127.0.0.1:9880", "name": "Server 1", "enabled": True},
            ]
        }
        with open(SERVERS_CONFIG_FILE, "w") as f:
            json.dump(default_servers, f, indent=2)
    
    # 初始化设置
    if not os.path.exists(SETTINGS_FILE):
        default_settings = {
            "sentence_split_length": 50,
            "timeout": 30,
            "retry_count": 3,
            "smart_split": False
        }
        with open(SETTINGS_FILE, "w") as f:
            json.dump(default_settings, f, indent=2)
    
    # 初始化参考音频配置
    if not os.path.exists(REFERENCE_AUDIO_CONFIG_FILE):
        default_reference_audio = {
            "reference_audio_path": ""
        }
        with open(REFERENCE_AUDIO_CONFIG_FILE, "w") as f:
            json.dump(default_reference_audio, f, indent=2)
    
    # 初始化TTS文本配置
    if not os.path.exists(TTSTEXT_CONFIG_FILE):
        default_tts_text = {
            "input_text": "",
            "prompt_text": ""
        }
        with open(TTSTEXT_CONFIG_FILE, "w") as f:
            json.dump(default_tts_text, f, indent=2)
    
    # 初始化自动保存配置
    if not os.path.exists(AUTO_SAVE_CONFIG_FILE):
        default_auto_save = {
            "auto_save": False,
            "auto_play": False
        }
        with open(AUTO_SAVE_CONFIG_FILE, "w") as f:
            json.dump(default_auto_save, f, indent=2)

# 加载配置
def load_config():
    init_default_config()
    
    with open(SERVERS_CONFIG_FILE, "r") as f:
        servers_config = json.load(f)
    
    with open(SETTINGS_FILE, "r") as f:
        settings = json.load(f)
    
    return servers_config, settings

# 保存服务器配置
def save_servers_config(servers):
    with open(SERVERS_CONFIG_FILE, "w") as f:
        json.dump({"servers": servers}, f, indent=2)

# 保存设置
def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

# 保存参考音频路径
def save_reference_audio_path(path):
    with open(REFERENCE_AUDIO_CONFIG_FILE, "w") as f:
        json.dump({"reference_audio_path": path}, f, indent=2)

# 加载参考音频路径
def load_reference_audio_path():
    init_default_config()
    with open(REFERENCE_AUDIO_CONFIG_FILE, "r") as f:
        config = json.load(f)
    return config.get("reference_audio_path", "")

# 保存TTS文本配置
def save_tts_text_config(input_text, prompt_text):
    config = {
        "input_text": input_text,
        "prompt_text": prompt_text
    }
    with open(TTSTEXT_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

# 加载TTS文本配置
def load_tts_text_config():
    init_default_config()
    with open(TTSTEXT_CONFIG_FILE, "r") as f:
        config = json.load(f)
    return config.get("input_text", ""), config.get("prompt_text", "")

# 保存自动保存配置
def save_auto_save_config(auto_save, auto_play):
    config = {
        "auto_save": auto_save,
        "auto_play": auto_play
    }
    with open(AUTO_SAVE_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

# 加载自动保存配置
def load_auto_save_config():
    init_default_config()
    with open(AUTO_SAVE_CONFIG_FILE, "r") as f:
        config = json.load(f)
    return config.get("auto_save", False), config.get("auto_play", False)

# TTS请求模型
class TTSRequest(BaseModel):
    text: str
    text_lang: str
    ref_audio_path: str
    aux_ref_audio_paths: list = []
    prompt_text: str = ""
    prompt_lang: str
    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    smart_split: bool = False

# 服务器模型
class Server(BaseModel):
    id: int
    url: str
    name: str
    enabled: bool

# 设置模型
class Settings(BaseModel):
    sentence_split_length: int
    timeout: int
    retry_count: int
    smart_split: bool = False

# 句子拆分函数
def split_sentence(text, max_length, smart_split=False):
    """根据最大长度拆分句子，支持智能切分
    
    Args:
        text: 要拆分的文本
        max_length: 最大长度（当smart_split=False时使用）
        smart_split: 是否使用智能切分（完全基于标点符号）
    """
    # 定义标点符号
    punctuation = ["。", "！", "？", ".", "!", "?", "；", ";"]
    minor_punctuation = ["，", ",", "、"]
    
    # 如果是智能切分，基于标点符号，同时考虑最大长度
    if smart_split:
        sentences = []
        current = ""
        
        for char in text:
            current += char
            
            # 遇到主要标点符号，直接拆分
            if char in punctuation:
                sentences.append(current)
                current = ""
            # 遇到逗号，检查当前片段长度，如果超过最大长度则拆分
            elif char in minor_punctuation:
                # 逗号作为次要标点符号，只有当当前片段长度超过最大长度时才拆分
                if len(current) > max_length:
                    sentences.append(current)
                    current = ""
        
        if current.strip():
            sentences.append(current)
        
        # 过滤掉空句子
        result = [s.strip() for s in sentences if s.strip()]
        
        # 如果结果只有一个片段，且长度超过最大长度，强制拆分为多个片段
        if len(result) == 1 and len(result[0]) > max_length:
            logger.info(f"Forcing split of single large segment ({len(result[0])} chars) into smaller parts")
            # 使用普通切分逻辑处理这个长片段
            forced_split = split_sentence(result[0], max_length, smart_split=False)
            result = forced_split
        
        logger.info(f"Smart split result: {len(result)} parts")
        return result
    
    # 原始的基于长度的拆分逻辑
    if len(text) <= max_length:
        return [text]
    
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if len(current) >= max_length:
            # 查找最近的标点符号
            for i in range(len(current)-1, -1, -1):
                if current[i] in punctuation or current[i] in minor_punctuation:
                    sentences.append(current[:i+1])
                    current = current[i+1:]
                    break
            else:
                # 没有找到标点符号，直接按最大长度拆分
                sentences.append(current[:max_length])
                current = current[max_length:]
    
    if current:
        sentences.append(current)
    
    return sentences

# 异步发送请求到TTS服务器
async def send_tts_request(server_url, request_data):
    """发送TTS请求到指定服务器"""
    logger.info(f"Sending TTS request to server: {server_url}")
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{server_url}/tts", json=request_data) as response:
            if response.status == 200:
                logger.info(f"TTS request successful from server: {server_url}")
                return await response.read()
            else:
                error_text = await response.text()
                logger.error(f"TTS request failed from server {server_url}: {error_text}")
                raise HTTPException(status_code=response.status, detail=f"Server {server_url} failed: {error_text}")

# 合并音频文件
def merge_audio_files(audio_bytes_list, media_type="wav"):
    """合并多个音频文件"""
    if not audio_bytes_list:
        return b""
    
    # 如果是单个音频，直接返回
    if len(audio_bytes_list) == 1:
        return audio_bytes_list[0]
    
    # 对于wav格式，需要合并音频数据
    if media_type == "wav":
        # 读取所有音频数据
        audio_data_list = []
        sample_rate = None
        
        for audio_bytes in audio_bytes_list:
            with sf.SoundFile(BytesIO(audio_bytes)) as f:
                current_sr = f.samplerate
                if sample_rate is None:
                    sample_rate = current_sr
                elif sample_rate != current_sr:
                    logger.error(f"Sample rate mismatch: {sample_rate} vs {current_sr}")
                    raise HTTPException(status_code=500, detail="Sample rate mismatch")
                
                audio_data = f.read(dtype="float32")
                audio_data_list.append(audio_data)
        
        # 合并音频数据
        merged_audio = np.concatenate(audio_data_list)
        
        # 写入到BytesIO
        output = BytesIO()
        sf.write(output, merged_audio, sample_rate, format="wav")
        output.seek(0)
        
        return output.read()
    
    # 对于其他格式，暂时不支持合并，返回第一个音频
    logger.warning(f"Merge not supported for {media_type}, returning first audio")
    return audio_bytes_list[0]

# 主TTS路由
@app.post("/tts")
async def tts_router(request: TTSRequest):
    """TTS请求路由，分发到多个服务器并汇总结果"""
    servers_config, settings = load_config()
    
    # 获取启用的服务器
    enabled_servers = [s for s in servers_config["servers"] if s["enabled"]]
    if not enabled_servers:
        raise HTTPException(status_code=500, detail="No enabled servers")
    
    # 记录启用的服务器详情
    for i, server in enumerate(enabled_servers):
        logger.info(f"Enabled server {i}: {server['name']} - {server['url']} (enabled: {server['enabled']})")
    
    # 拆分句子
    split_texts = split_sentence(request.text, settings["sentence_split_length"], request.smart_split)
    if not split_texts:
        raise HTTPException(status_code=400, detail="Empty text after split")
    
    logger.info(f"Split text into {len(split_texts)} parts")
    logger.info(f"Using {len(enabled_servers)} enabled servers")
    logger.info(f"Smart split enabled: {request.smart_split}")
    
    # 准备请求数据
    request_data = request.dict()
    
    # 并行发送请求
    tasks = []
    for i, text_part in enumerate(split_texts):
        server_index = i % len(enabled_servers)
        server = enabled_servers[server_index]
        logger.info(f"Assigning part {i+1}/{len(split_texts)} to server {server_index} ({server['name']} - {server['url']})")
        
        part_request = request_data.copy()
        part_request["text"] = text_part
        part_request["streaming_mode"] = False  # 关闭流式模式，因为需要合并结果
        
        task = asyncio.create_task(send_tts_request(server["url"], part_request))
        tasks.append(task)
    
    # 等待所有请求完成
    try:
        results = await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Error in TTS requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # 合并结果
    merged_audio = merge_audio_files(results, request.media_type)
    
    # 返回结果
    return StreamingResponse(
        BytesIO(merged_audio),
        media_type=f"audio/{request.media_type}"
    )

# 获取服务器列表
@app.get("/api/servers")
async def get_servers():
    """获取服务器列表"""
    servers_config, _ = load_config()
    return servers_config["servers"]

# 添加服务器
@app.post("/api/servers")
async def add_server(server: Server):
    """添加服务器"""
    servers_config, _ = load_config()
    servers = servers_config["servers"]
    
    # 检查ID是否已存在
    if any(s["id"] == server.id for s in servers):
        raise HTTPException(status_code=400, detail="Server ID already exists")
    
    servers.append(server.dict())
    save_servers_config(servers)
    return {"message": "Server added successfully"}

# 更新服务器
@app.put("/api/servers/{server_id}")
async def update_server(server_id: int, server: Server):
    """更新服务器"""
    servers_config, _ = load_config()
    servers = servers_config["servers"]
    
    # 查找服务器
    for i, s in enumerate(servers):
        if s["id"] == server_id:
            servers[i] = server.dict()
            save_servers_config(servers)
            return {"message": "Server updated successfully"}
    
    raise HTTPException(status_code=404, detail="Server not found")

# 删除服务器
@app.delete("/api/servers/{server_id}")
async def delete_server(server_id: int):
    """删除服务器"""
    servers_config, _ = load_config()
    servers = servers_config["servers"]
    
    # 查找并删除服务器
    for i, s in enumerate(servers):
        if s["id"] == server_id:
            del servers[i]
            save_servers_config(servers)
            return {"message": "Server deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Server not found")

# 获取设置
@app.get("/api/settings")
async def get_settings():
    """获取设置"""
    _, settings = load_config()
    return settings

# 更新设置
@app.put("/api/settings")
async def update_settings(settings: Settings):
    """更新设置"""
    save_settings(settings.dict())
    return {"message": "Settings updated successfully"}

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

# 启动时初始化配置
init_default_config()

# ------------------------------
# GUI实现
# ------------------------------

class TTSRouterGUI:
    def __init__(self):
        # 后端API地址
        self.API_BASE_URL = "http://localhost:8888"
        
        # 存储生成的音频数据
        self.generated_audio = None
        
        # 创建主窗口
        self.root = None
        self.main_frame = None
        self.tree = None
        self.split_length_var = None
        self.timeout_var = None
        self.retry_var = None
        self.text_text = None
        self.text_lang_var = None
        self.prompt_lang_var = None
        self.prompt_text_var = None
        self.ref_audio_var = None
        self.status_var = None
        self.health_text = None
        # 添加自动保存和自动播放变量
        self.auto_save_var = None
        self.auto_play_var = None
        # 添加智能切分变量
        self.smart_split_var = None
        self.split_length_entry = None
        self.tts_smart_split_var = None
        
        # 保存按钮引用
        self.generate_btn = None
        self.play_btn = None
        self.save_btn = None
        self.check_health_btn = None
        
        # 初始化pygame用于音频播放
        pygame.mixer.init()
    
    def create_gui(self):
        """创建GUI界面"""
        # 设置窗口主题样式
        try:
            from ttkthemes import ThemedTk, THEMES
            self.root = ThemedTk(theme="arc")
        except ImportError:
            # 如果没有ttkthemes，使用默认主题
            self.root = tk.Tk()
        
        self.root.title("TTS API Router - GUI")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 设置样式风格
        style = ttk.Style()
        style.configure("TButton", padding=6, font=('Segoe UI', 10))
        style.configure("TLabel", font=('Segoe UI', 10))
        style.configure("TEntry", padding=6, font=('Segoe UI', 10))
        style.configure("TText", font=('Segoe UI', 10))
        style.configure("Treeview", font=('Segoe UI', 10))
        style.configure("Heading1.TLabel", font=('Segoe UI', 14, 'bold'), padding=10)
        
        # 创建主框架
        self.main_frame = ttk.Notebook(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 服务器管理模块
        self.create_server_management_tab()
        
        # 设置管理模块
        self.create_settings_tab()
        
        # TTS测试模块
        self.create_tts_test_tab()
        
        # 健康检查模块
        self.create_health_check_tab()
        
        # 创建菜单
        self.create_menu()
    
    def create_server_management_tab(self):
        """创建服务器管理标签页"""
        server_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(server_frame, text="服务器管理")
        
        # 服务器管理标题
        server_title = ttk.Label(server_frame, text="服务器管理", style="Heading1.TLabel")
        server_title.pack(anchor=tk.W, pady=5)
        
        # 服务器列表框架
        server_list_frame = ttk.LabelFrame(server_frame, text="服务器列表")
        server_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建服务器列表树
        columns = ("id", "name", "url", "enabled")
        self.tree = ttk.Treeview(server_list_frame, columns=columns, show="headings")
        self.tree.heading("id", text="ID")
        self.tree.heading("name", text="名称")
        self.tree.heading("url", text="URL")
        self.tree.heading("enabled", text="状态")
        
        # 设置列宽
        self.tree.column("id", width=50, anchor=tk.CENTER)
        self.tree.column("name", width=150, anchor=tk.W)
        self.tree.column("url", width=300, anchor=tk.W)
        self.tree.column("enabled", width=80, anchor=tk.CENTER)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(server_list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # 服务器操作按钮框架
        server_buttons_frame = ttk.Frame(server_frame)
        server_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 服务器操作按钮
        add_btn = ttk.Button(server_buttons_frame, text="添加服务器", command=self.add_server)
        edit_btn = ttk.Button(server_buttons_frame, text="编辑服务器", command=self.edit_server)
        delete_btn = ttk.Button(server_buttons_frame, text="删除服务器", command=self.delete_server)
        enable_btn = ttk.Button(server_buttons_frame, text="启用服务器", command=self.enable_server)
        disable_btn = ttk.Button(server_buttons_frame, text="禁用服务器", command=self.disable_server)
        refresh_btn = ttk.Button(server_buttons_frame, text="刷新列表", command=self.load_servers)
        
        # 布局按钮
        add_btn.pack(side=tk.LEFT, padx=5)
        edit_btn.pack(side=tk.LEFT, padx=5)
        delete_btn.pack(side=tk.LEFT, padx=5)
        enable_btn.pack(side=tk.LEFT, padx=5)
        disable_btn.pack(side=tk.LEFT, padx=5)
        refresh_btn.pack(side=tk.RIGHT, padx=5)
    
    def create_settings_tab(self):
        """创建设置管理标签页"""
        settings_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(settings_frame, text="系统设置")
        
        # 设置标题
        settings_title = ttk.Label(settings_frame, text="系统设置", style="Heading1.TLabel")
        settings_title.pack(anchor=tk.W, pady=5)
        
        # 设置表单框架
        settings_form_frame = ttk.LabelFrame(settings_frame, text="设置")
        settings_form_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 智能切分选项
        self.smart_split_var = tk.BooleanVar()
        smart_split_check = ttk.Checkbutton(settings_form_frame, text="智能切分（基于标点符号）", 
                                           variable=self.smart_split_var, command=self.toggle_smart_split)
        smart_split_check.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky=tk.W)
        
        # 句子拆分长度
        split_length_label = ttk.Label(settings_form_frame, text="句子拆分长度:")
        split_length_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.split_length_var = tk.IntVar()
        self.split_length_entry = ttk.Entry(settings_form_frame, textvariable=self.split_length_var, width=20)
        self.split_length_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        
        # 超时时间
        timeout_label = ttk.Label(settings_form_frame, text="超时时间 (秒):")
        timeout_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.timeout_var = tk.IntVar()
        timeout_entry = ttk.Entry(settings_form_frame, textvariable=self.timeout_var, width=20)
        timeout_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)
        
        # 重试次数
        retry_label = ttk.Label(settings_form_frame, text="重试次数:")
        retry_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        self.retry_var = tk.IntVar()
        retry_entry = ttk.Entry(settings_form_frame, textvariable=self.retry_var, width=20)
        retry_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)
        
        # 保存设置按钮
        save_settings_btn = ttk.Button(settings_form_frame, text="保存设置", command=self.save_settings)
        save_settings_btn.grid(row=4, column=0, columnspan=2, pady=20)
    
    def create_tts_test_tab(self):
        """创建TTS测试标签页"""
        tts_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(tts_frame, text="TTS测试")
        
        # TTS标题
        tts_title = ttk.Label(tts_frame, text="TTS测试", style="Heading1.TLabel")
        tts_title.pack(anchor=tk.W, pady=5)
        
        # TTS表单框架
        tts_form_frame = ttk.LabelFrame(tts_frame, text="TTS参数")
        tts_form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 文本输入
        text_label = ttk.Label(tts_form_frame, text="输入文本:")
        text_label.pack(anchor=tk.W, padx=10, pady=5)
        self.text_text = tk.Text(tts_form_frame, height=10, wrap=tk.WORD)
        self.text_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 语言设置框架
        lang_frame = ttk.Frame(tts_form_frame)
        lang_frame.pack(fill=tk.X, padx=10, pady=5)
        
        text_lang_label = ttk.Label(lang_frame, text="文本语言:")
        text_lang_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.text_lang_var = tk.StringVar(value="zh")
        text_lang_entry = ttk.Entry(lang_frame, textvariable=self.text_lang_var, width=10)
        text_lang_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        
        prompt_lang_label = ttk.Label(lang_frame, text="提示语言:")
        prompt_lang_label.grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)
        self.prompt_lang_var = tk.StringVar(value="zh")
        prompt_lang_entry = ttk.Entry(lang_frame, textvariable=self.prompt_lang_var, width=10)
        prompt_lang_entry.grid(row=0, column=3, padx=10, pady=5, sticky=tk.W)
        
        # 提示文本输入框
        prompt_text_frame = ttk.Frame(tts_form_frame)
        prompt_text_frame.pack(fill=tk.X, padx=10, pady=5)
        
        prompt_text_label = ttk.Label(prompt_text_frame, text="提示文本:")
        prompt_text_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.prompt_text_var = tk.StringVar(value="")
        prompt_text_entry = ttk.Entry(prompt_text_frame, textvariable=self.prompt_text_var, width=50)
        prompt_text_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        
        # 参考音频路径框架
        ref_audio_frame = ttk.Frame(tts_form_frame)
        ref_audio_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ref_audio_label = ttk.Label(ref_audio_frame, text="参考音频路径:")
        ref_audio_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.ref_audio_var = tk.StringVar()
        ref_audio_entry = ttk.Entry(ref_audio_frame, textvariable=self.ref_audio_var, width=40)
        ref_audio_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        
        # 添加浏览按钮
        browse_btn = ttk.Button(ref_audio_frame, text="浏览", command=self.browse_reference_audio)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 自动保存和自动播放设置
        auto_settings_frame = ttk.Frame(tts_form_frame)
        auto_settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.auto_save_var = tk.BooleanVar()
        auto_save_check = ttk.Checkbutton(auto_settings_frame, text="自动保存音频", variable=self.auto_save_var, command=self.toggle_auto_settings)
        auto_save_check.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.auto_play_var = tk.BooleanVar()
        auto_play_check = ttk.Checkbutton(auto_settings_frame, text="自动播放音频", variable=self.auto_play_var, command=self.toggle_auto_settings)
        auto_play_check.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 生成和播放按钮框架
        tts_buttons_frame = ttk.Frame(tts_frame)
        tts_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 保存按钮引用
        self.generate_btn = ttk.Button(tts_buttons_frame, text="生成音频", command=self.generate_tts)
        self.play_btn = ttk.Button(tts_buttons_frame, text="播放音频", command=self.play_audio, state="disabled")
        self.save_btn = ttk.Button(tts_buttons_frame, text="保存音频", command=self.save_audio, state="disabled")
        clear_btn = ttk.Button(tts_buttons_frame, text="清空", command=self.clear_text)
        
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态显示
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(tts_buttons_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=10)
        
    def browse_reference_audio(self):
        """浏览按钮：打开文件选择对话框选择参考音频"""
        file_path = filedialog.askopenfilename(
            title="选择参考音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac"), ("所有文件", "*.*")]
        )
        if file_path:
            self.ref_audio_var.set(file_path)
            # 保存到配置文件
            config = {
                "reference_audio_path": file_path
            }
            with open(REFERENCE_AUDIO_CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
    
    def load_reference_audio_path(self):
        """从配置文件加载参考音频路径"""
        try:
            with open(REFERENCE_AUDIO_CONFIG_FILE, "r") as f:
                config = json.load(f)
                self.ref_audio_var.set(config.get("reference_audio_path", ""))
        except FileNotFoundError:
            # 配置文件不存在，使用默认值
            self.ref_audio_var.set("")
        except Exception as e:
            logger.error(f"Failed to load reference audio path: {e}")
            self.ref_audio_var.set("")
    
    def create_health_check_tab(self):
        """创建健康检查标签页"""
        health_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(health_frame, text="健康检查")
        
        # 健康检查标题
        health_title = ttk.Label(health_frame, text="健康检查", style="Heading1.TLabel")
        health_title.pack(anchor=tk.W, pady=5)
        
        # 健康检查结果框架
        health_result_frame = ttk.LabelFrame(health_frame, text="检查结果")
        health_result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.health_text = tk.Text(health_result_frame, height=20, wrap=tk.WORD)
        self.health_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 健康检查按钮
        self.check_health_btn = ttk.Button(health_frame, text="执行健康检查", command=self.check_health)
        self.check_health_btn.pack(pady=10)
    
    def create_menu(self):
        """创建菜单"""
        # 创建菜单
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 操作菜单
        operation_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="操作", menu=operation_menu)
        operation_menu.add_command(label="刷新服务器列表", command=self.load_servers)
        operation_menu.add_command(label="执行健康检查", command=self.check_health)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=lambda: messagebox.showinfo("关于", "TTS API Router GUI\n版本: 1.0\n用于管理TTS API Router后端服务"))
    
    # 加载服务器列表
    def load_servers(self):
        try:
            response = requests.get(f"{self.API_BASE_URL}/api/servers")
            if response.status_code == 200:
                servers = response.json()
                # 清空树
                for item in self.tree.get_children():
                    self.tree.delete(item)
                # 添加服务器
                for server in servers:
                    status = "启用" if server["enabled"] else "禁用"
                    self.tree.insert("", tk.END, values=(server["id"], server["name"], server["url"], status), tags=("enabled" if server["enabled"] else "disabled",))
                # 设置标签样式
                self.tree.tag_configure("enabled", background="#e8f5e8")
                self.tree.tag_configure("disabled", background="#ffebee")
            else:
                messagebox.showerror("错误", f"获取服务器列表失败: {response.text}")
        except Exception as e:
            messagebox.showerror("错误", f"无法连接到后端: {str(e)}")
    
    # 加载设置
    def load_settings(self):
        try:
            response = requests.get(f"{self.API_BASE_URL}/api/settings")
            if response.status_code == 200:
                settings = response.json()
                self.split_length_var.set(settings["sentence_split_length"])
                self.timeout_var.set(settings["timeout"])
                self.retry_var.set(settings["retry_count"])
                # 加载智能切分配置
                smart_split = settings.get("smart_split", False)
                self.smart_split_var.set(smart_split)
                # 更新句子拆分长度输入框的可用性
                if smart_split:
                    self.split_length_entry.config(state="disabled")
                else:
                    self.split_length_entry.config(state="normal")
            else:
                messagebox.showerror("错误", f"获取设置失败: {response.text}")
        except Exception as e:
            messagebox.showerror("错误", f"无法连接到后端: {str(e)}")
    
    # 添加服务器
    def add_server(self):
        # 创建添加服务器对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("添加服务器")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 名称
        name_label = ttk.Label(dialog, text="名称:")
        name_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        
        # URL
        url_label = ttk.Label(dialog, text="URL:")
        url_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        url_var = tk.StringVar(value="http://")
        url_entry = ttk.Entry(dialog, textvariable=url_var, width=30)
        url_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        
        # 启用
        enabled_var = tk.BooleanVar(value=True)
        enabled_check = ttk.Checkbutton(dialog, text="启用", variable=enabled_var)
        enabled_check.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        # 保存按钮
        def save_server():
            name = name_var.get().strip()
            url = url_var.get().strip()
            if not name or not url:
                messagebox.showwarning("警告", "名称和URL不能为空")
                return
            
            try:
                server_data = {
                    "id": int(time.time()),  # 使用时间戳作为临时ID
                    "name": name,
                    "url": url,
                    "enabled": enabled_var.get()
                }
                response = requests.post(f"{self.API_BASE_URL}/api/servers", json=server_data)
                if response.status_code == 200:
                    messagebox.showinfo("成功", "服务器添加成功")
                    dialog.destroy()
                    self.load_servers()
                else:
                    messagebox.showerror("错误", f"添加服务器失败: {response.text}")
            except Exception as e:
                messagebox.showerror("错误", f"无法连接到后端: {str(e)}")
        
        save_btn = ttk.Button(dialog, text="保存", command=save_server, style="Primary.TButton")
        save_btn.grid(row=3, column=0, columnspan=2, pady=20)
        dialog.bind("<Return>", lambda event: save_server())
    
    # 编辑服务器
    def edit_server(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请选择要编辑的服务器")
            return
        
        item = selected[0]
        values = self.tree.item(item, "values")
        server_id = values[0]
        
        # 创建编辑对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("编辑服务器")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 名称
        name_label = ttk.Label(dialog, text="名称:")
        name_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        name_var = tk.StringVar(value=values[1])
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        
        # URL
        url_label = ttk.Label(dialog, text="URL:")
        url_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        url_var = tk.StringVar(value=values[2])
        url_entry = ttk.Entry(dialog, textvariable=url_var, width=30)
        url_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        
        # 启用
        enabled_var = tk.BooleanVar(value=(values[3] == "启用"))
        enabled_check = ttk.Checkbutton(dialog, text="启用", variable=enabled_var)
        enabled_check.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        # 保存按钮
        def save_server():
            name = name_var.get().strip()
            url = url_var.get().strip()
            if not name or not url:
                messagebox.showwarning("警告", "名称和URL不能为空")
                return
            
            try:
                server_data = {
                    "id": int(server_id),
                    "name": name,
                    "url": url,
                    "enabled": enabled_var.get()
                }
                response = requests.put(f"{self.API_BASE_URL}/api/servers/{server_id}", json=server_data)
                if response.status_code == 200:
                    messagebox.showinfo("成功", "服务器更新成功")
                    dialog.destroy()
                    self.load_servers()
                else:
                    messagebox.showerror("错误", f"更新服务器失败: {response.text}")
            except Exception as e:
                messagebox.showerror("错误", f"无法连接到后端: {str(e)}")
        
        save_btn = ttk.Button(dialog, text="保存", command=save_server, style="Primary.TButton")
        save_btn.grid(row=3, column=0, columnspan=2, pady=20)
        dialog.bind("<Return>", lambda event: save_server())
    
    # 删除服务器
    def delete_server(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请选择要删除的服务器")
            return
        
        item = selected[0]
        values = self.tree.item(item, "values")
        server_id = values[0]
        
        if messagebox.askyesno("确认", f"确定要删除服务器 '{values[1]}' 吗?"):
            try:
                response = requests.delete(f"{self.API_BASE_URL}/api/servers/{server_id}")
                if response.status_code == 200:
                    messagebox.showinfo("成功", "服务器删除成功")
                    self.load_servers()
                else:
                    messagebox.showerror("错误", f"删除服务器失败: {response.text}")
            except Exception as e:
                messagebox.showerror("错误", f"无法连接到后端: {str(e)}")
    
    # 启用服务器
    def enable_server(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请选择要启用的服务器")
            return
        
        item = selected[0]
        values = self.tree.item(item, "values")
        server_id = values[0]
        
        try:
            server_data = {
                "id": int(server_id),
                "name": values[1],
                "url": values[2],
                "enabled": True
            }
            response = requests.put(f"{self.API_BASE_URL}/api/servers/{server_id}", json=server_data)
            if response.status_code == 200:
                messagebox.showinfo("成功", "服务器已启用")
                self.load_servers()
            else:
                messagebox.showerror("错误", f"启用服务器失败: {response.text}")
        except Exception as e:
            messagebox.showerror("错误", f"无法连接到后端: {str(e)}")
    
    # 禁用服务器
    def disable_server(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请选择要禁用的服务器")
            return
        
        item = selected[0]
        values = self.tree.item(item, "values")
        server_id = values[0]
        
        try:
            server_data = {
                "id": int(server_id),
                "name": values[1],
                "url": values[2],
                "enabled": False
            }
            response = requests.put(f"{self.API_BASE_URL}/api/servers/{server_id}", json=server_data)
            if response.status_code == 200:
                messagebox.showinfo("成功", "服务器已禁用")
                self.load_servers()
            else:
                messagebox.showerror("错误", f"禁用服务器失败: {response.text}")
        except Exception as e:
            messagebox.showerror("错误", f"无法连接到后端: {str(e)}")
    
    # 切换智能切分状态
    def toggle_smart_split(self):
        """切换智能切分状态，更新句子拆分长度输入框的可用性"""
        is_smart = self.smart_split_var.get()
        # 如果启用了智能切分，禁用句子拆分长度输入框
        if is_smart:
            self.split_length_entry.config(state="disabled")
        else:
            self.split_length_entry.config(state="normal")
    
    # 保存设置
    def save_settings(self):
        settings = {
            "sentence_split_length": self.split_length_var.get(),
            "timeout": self.timeout_var.get(),
            "retry_count": self.retry_var.get(),
            "smart_split": self.smart_split_var.get()
        }
        
        try:
            response = requests.put(f"{self.API_BASE_URL}/api/settings", json=settings)
            if response.status_code == 200:
                messagebox.showinfo("成功", "设置保存成功")
            else:
                messagebox.showerror("错误", f"保存设置失败: {response.text}")
        except Exception as e:
            messagebox.showerror("错误", f"无法连接到后端: {str(e)}")
    
    # 生成音频
    def generate_tts(self):
        text = self.text_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "请输入要生成的文本")
            return
        
        text_lang = self.text_lang_var.get().strip()
        prompt_lang = self.prompt_lang_var.get().strip()
        ref_audio_path = self.ref_audio_var.get().strip()
        
        if not text_lang or not prompt_lang:
            messagebox.showwarning("警告", "语言设置不能为空")
            return
        
        if not ref_audio_path:
            messagebox.showwarning("警告", "参考音频路径不能为空")
            return
        
        # 异步生成音频，避免界面冻结
        def generate():
            # 直接使用保存的按钮引用
            self.root.after(0, lambda: self.status_var.set("生成中..."))
            self.root.after(0, lambda: self.generate_btn.config(state="disabled"))
            
            try:
                # 获取系统设置中的智能切分配置
                try:
                    settings_response = requests.get(f"{self.API_BASE_URL}/api/settings")
                    settings = settings_response.json() if settings_response.status_code == 200 else {}
                    smart_split = settings.get("smart_split", False)
                except:
                    smart_split = False
                
                data = {
                    "text": text,
                    "text_lang": text_lang,
                    "ref_audio_path": ref_audio_path,
                    "prompt_lang": prompt_lang,
                    "prompt_text": self.prompt_text_var.get(),
                    "media_type": "wav",
                    "smart_split": smart_split
                }
                response = requests.post(f"{self.API_BASE_URL}/tts", json=data)
                if response.status_code == 200:
                    self.generated_audio = response.content
                    self.root.after(0, lambda: self.status_var.set("生成成功"))
                    self.root.after(0, lambda: self.play_btn.config(state="normal"))
                    self.root.after(0, lambda: self.save_btn.config(state="normal"))
                    
                    # 保存TTS文本配置
                    self.save_tts_text()
                    
                    # 如果启用了自动保存，自动保存音频（不显示提示框）
                    if self.auto_save_var.get():
                        self.root.after(0, lambda: self.save_audio(show_message=False))
                    
                    # 如果启用了自动播放，自动播放音频
                    if self.auto_play_var.get():
                        self.root.after(0, lambda: self.play_audio())
                else:
                    self.root.after(0, lambda: self.status_var.set("生成失败"))
                    self.root.after(0, lambda: messagebox.showerror("错误", f"生成音频失败: {response.text}"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set("生成失败"))
                self.root.after(0, lambda: messagebox.showerror("错误", f"无法连接到后端: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.generate_btn.config(state="normal"))
        
        threading.Thread(target=generate, daemon=True).start()
    
    # 播放音频
    def play_audio(self):
        if not self.generated_audio:
            messagebox.showwarning("警告", "请先生成音频")
            return
        
        try:
            # 使用pygame播放音频
            pygame.mixer.music.load(io.BytesIO(self.generated_audio))
            pygame.mixer.music.play()
            self.status_var.set("播放中...")
            
            # 播放完成后更新状态
            def check_playing():
                if pygame.mixer.music.get_busy():
                    self.root.after(100, check_playing)
                else:
                    self.status_var.set("就绪")
            
            self.root.after(100, check_playing)
        except Exception as e:
            messagebox.showerror("错误", f"播放音频失败: {str(e)}")
            self.status_var.set("就绪")
    
    # 保存音频到文件
    def save_audio(self, show_message=True):
        if not self.generated_audio:
            if show_message:
                messagebox.showwarning("警告", "请先生成音频")
            return
        
        try:
            # 获取输入文本，用于生成文件名
            input_text = self.text_text.get(1.0, tk.END).strip()
            
            # 如果文本太长，只取前20个字符
            if len(input_text) > 20:
                input_text = input_text[:20] + "..."
            
            # 替换文件名中的非法字符
            input_text = input_text.replace("\\", "_").replace("/", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            
            # 获取当前时间，格式化为：年-月-日_时-分-秒
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            
            # 生成文件名
            filename = f"{input_text}_{current_time}.wav"
            
            # 创建缓存目录
            cache_dir = "audio_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            # 完整文件路径
            file_path = os.path.join(cache_dir, filename)
            
            # 保存音频文件
            with open(file_path, "wb") as f:
                f.write(self.generated_audio)
            
            if show_message:
                messagebox.showinfo("成功", f"音频已保存到：\n{file_path}")
        except Exception as e:
            if show_message:
                messagebox.showerror("错误", f"保存音频失败: {str(e)}")
            logger.error(f"Failed to save audio: {e}")
    
    # 清空文本
    def clear_text(self):
        self.text_text.delete(1.0, tk.END)
        self.text_text.focus()
    
    # 执行健康检查
    def check_health(self):
        self.health_text.delete(1.0, tk.END)
        
        # 直接使用保存的健康检查按钮引用
        if not self.check_health_btn:
            messagebox.showerror("错误", "找不到健康检查按钮")
            return
        
        # 设置按钮状态为禁用
        self.check_health_btn.config(state="disabled")
        self.health_text.insert(tk.END, "正在执行健康检查...\n\n")
        
        def check():
            try:
                # 检查后端API健康状态
                response = requests.get(f"{self.API_BASE_URL}/health")
                if response.status_code == 200:
                    self.root.after(0, lambda: self.health_text.insert(tk.END, f"✅ 后端API: 正常\n"))
                    self.root.after(0, lambda: self.health_text.insert(tk.END, f"   状态: {response.json()['status']}\n\n"))
                else:
                    self.root.after(0, lambda: self.health_text.insert(tk.END, f"❌ 后端API: 异常\n"))
                    self.root.after(0, lambda: self.health_text.insert(tk.END, f"   状态码: {response.status_code}\n\n"))
                
                # 检查服务器列表
                response = requests.get(f"{self.API_BASE_URL}/api/servers")
                if response.status_code == 200:
                    servers = response.json()
                    self.root.after(0, lambda: self.health_text.insert(tk.END, f"✅ 服务器列表: 正常\n"))
                    self.root.after(0, lambda: self.health_text.insert(tk.END, f"   服务器数量: {len(servers)}\n"))
                    for server in servers:
                        status = "启用" if server["enabled"] else "禁用"
                        self.root.after(0, lambda: self.health_text.insert(tk.END, f"   - {server['name']}: {server['url']} ({status})\n"))
                    self.root.after(0, lambda: self.health_text.insert(tk.END, "\n"))
                else:
                    self.root.after(0, lambda: self.health_text.insert(tk.END, f"❌ 服务器列表: 异常\n"))
                    self.root.after(0, lambda: self.health_text.insert(tk.END, f"   状态码: {response.status_code}\n\n"))
                
                self.root.after(0, lambda: self.health_text.insert(tk.END, "✅ 健康检查完成\n"))
            except Exception as e:
                self.root.after(0, lambda: self.health_text.insert(tk.END, f"❌ 健康检查失败: {str(e)}\n"))
            finally:
                self.root.after(0, lambda: self.check_health_btn.config(state="normal"))
        
        threading.Thread(target=check, daemon=True).start()
    
    def run(self):
        """运行GUI"""
        self.create_gui()
        self.load_servers()
        self.load_settings()
        # 加载参考音频路径
        self.load_reference_audio_path()
        # 加载TTS文本配置
        self.load_tts_text_config()
        # 加载自动保存配置
        self.load_auto_save_config()
        self.root.mainloop()
    
    def toggle_auto_settings(self):
        """切换自动保存和自动播放状态"""
        auto_save = self.auto_save_var.get()
        auto_play = self.auto_play_var.get()
        save_auto_save_config(auto_save, auto_play)
    
    def load_tts_text_config(self):
        """加载TTS文本配置"""
        try:
            with open(TTSTEXT_CONFIG_FILE, "r") as f:
                config = json.load(f)
                input_text = config.get("input_text", "")
                prompt_text = config.get("prompt_text", "")
                # 设置文本框内容
                self.text_text.delete(1.0, tk.END)
                self.text_text.insert(tk.END, input_text)
                self.prompt_text_var.set(prompt_text)
        except Exception as e:
            logger.error(f"Failed to load TTS text config: {e}")
    
    def save_tts_text(self):
        """保存TTS文本"""
        input_text = self.text_text.get(1.0, tk.END).strip()
        prompt_text = self.prompt_text_var.get().strip()
        save_tts_text_config(input_text, prompt_text)
    
    def load_auto_save_config(self):
        """加载自动保存和自动播放配置"""
        try:
            with open(AUTO_SAVE_CONFIG_FILE, "r") as f:
                config = json.load(f)
                auto_save = config.get("auto_save", False)
                auto_play = config.get("auto_play", False)
                self.auto_save_var.set(auto_save)
                self.auto_play_var.set(auto_play)
        except Exception as e:
            logger.error(f"Failed to load auto settings config: {e}")
            self.auto_save_var.set(False)
            self.auto_play_var.set(False)

# ------------------------------
# 后端启动函数
# ------------------------------

def start_backend():
    """启动FastAPI后端"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888, workers=1)

# ------------------------------
# 主函数
# ------------------------------

if __name__ == "__main__":
    # 在后台线程启动FastAPI后端
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # 等待后端服务启动
    time.sleep(2)
    
    # 启动GUI
    gui = TTSRouterGUI()
    gui.run()