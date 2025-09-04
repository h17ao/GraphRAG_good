#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from datetime import datetime
import os
from loguru import logger as _logger
from Option.Config2 import default_config
_print_level = "INFO"


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")

    
    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    
    if name:
        log_dir = os.path.join(name, "Logs")
        os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
        log_name = os.path.join(log_dir, f"{formatted_date}.log")
        _logger.add(f"{log_name}", level=logfile_level)
    # 如果没有指定name，只输出到stderr，不创建日志文件
    
    return _logger


def reconfigure_logger_for_dataset(dataset_name: str, exp_name: str = "rag_experiments", working_dir: str = "./", print_level="INFO", logfile_level="DEBUG"):
    """根据dataset_name重新配置日志路径到dataset_name/rag_experiments/Logs/"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")

    # 构建日志目录路径: dataset_name/rag_experiments/Logs/
    log_dir = os.path.join(working_dir, dataset_name, exp_name, "Logs")
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    log_name = os.path.join(log_dir, f"{formatted_date}.log")

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(f"{log_name}", level=logfile_level)
    return _logger


def reconfigure_logger_for_method_dataset(method_name: str, dataset_name: str, exp_name: str = "rag_experiments", working_dir: str = "./", print_level="INFO", logfile_level="DEBUG"):
    """根据method_name和dataset_name重新配置日志路径到dataset_name/method_name/rag_experiments/Logs/"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")

    # 构建日志目录路径
    if method_name:
        # dataset_name/method_name/rag_experiments/Logs/
        log_dir = os.path.join(working_dir, dataset_name, method_name, exp_name, "Logs")
    else:
        # dataset_name/rag_experiments/Logs/
        log_dir = os.path.join(working_dir, dataset_name, exp_name, "Logs")
    
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    log_name = os.path.join(log_dir, f"{formatted_date}.log")

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(f"{log_name}", level=logfile_level)
    return _logger


# 初始化默认logger（不创建日志文件，等到有具体配置时再创建）
logger = define_log_level()


def log_llm_stream(msg):
    _llm_stream_log(msg)


def set_llm_stream_logfunc(func):
    global _llm_stream_log
    _llm_stream_log = func


def _llm_stream_log(msg):
    if _print_level in ["INFO"]:
        print(msg, end="")