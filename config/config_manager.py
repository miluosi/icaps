"""
配置管理模块 - 加载和管理训练参数
"""
import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认使用config/config.json
        """
        if config_path is None:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'config_adp.json')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 配置加载成功: {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ 配置文件未找到: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ 配置文件格式错误: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "training": {
                "learning_rate": 0.001,
                "batch_size": 256,
                "gamma": 0.95
            },
            "sampling": {
                "default_method": "balanced"
            },
            "logging": {
                "debug_frequency": 100
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键路径
        
        Args:
            key_path: 键路径，如 'training.learning_rate'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key_path: 键路径，如 'training.learning_rate'
            value: 新值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到父级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
    
    def save(self, save_path: str = None) -> None:
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径，默认覆盖原文件
        """
        if save_path is None:
            save_path = self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✅ 配置保存成功: {save_path}")
        except Exception as e:
            print(f"❌ 配置保存失败: {e}")
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.get('training', {})
    
    def get_sampling_config(self) -> Dict[str, Any]:
        """获取采样配置"""
        return self.get('sampling', {})
    
    def get_network_config(self) -> Dict[str, Any]:
        """获取网络配置"""
        return self.get('network', {})
    
    def get_environment_config(self) -> Dict[str, Any]:
        """获取环境配置"""
        return self.get('environment', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.get('logging', {})
    
    def print_config(self, section: str = None) -> None:
        """
        打印配置信息
        
        Args:
            section: 特定部分，如果为None则打印全部
        """
        if section:
            config_to_print = self.get(section, {})
            print(f"\n=== {section.upper()} 配置 ===")
        else:
            config_to_print = self.config
            print(f"\n=== 完整配置 ===")
        
        print(json.dumps(config_to_print, indent=2, ensure_ascii=False))
    
    def get_sampling_method(self, training_step: int) -> str:
        """
        根据训练步数获取采样方法
        
        Args:
            training_step: 当前训练步数
            
        Returns:
            采样方法名称
        """
        schedule = self.get('sampling.adaptive_schedule', {})
        
        if training_step < schedule.get('stage_1', {}).get('steps', 1000):
            return schedule.get('stage_1', {}).get('method', 'balanced')
        elif training_step < schedule.get('stage_2', {}).get('steps', 3000):
            return schedule.get('stage_2', {}).get('method', 'importance')
        elif training_step < schedule.get('stage_3', {}).get('steps', 5000):
            return schedule.get('stage_3', {}).get('method', 'thompson')
        else:
            return schedule.get('stage_4', {}).get('method', 'prioritized')


# 全局配置实例
config_manager = ConfigManager()

# 便捷函数
def get_config(key_path: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return config_manager.get(key_path, default)

def get_training_config() -> Dict[str, Any]:
    """获取训练配置的便捷函数"""
    return config_manager.get_training_config()

def get_sampling_config() -> Dict[str, Any]:
    """获取采样配置的便捷函数"""
    return config_manager.get_sampling_config()

def get_sampling_method(training_step: int) -> str:
    """获取采样方法的便捷函数"""
    return config_manager.get_sampling_method(training_step)


if __name__ == "__main__":
    # 测试配置管理器
    print("🔧 配置管理器测试")
    
    # 打印所有配置
    config_manager.print_config()
    
    # 测试获取特定配置
    lr = get_config('training.learning_rate')
    print(f"\n学习率: {lr}")
    
    # 测试采样方法选择
    for step in [500, 1500, 3500, 6000]:
        method = get_sampling_method(step)
        print(f"训练步数 {step}: 采样方法 = {method}")