# Fly0: 解耦语义感知与几何规划以实现零样本航空导航

我们提出了 Fly0，一个专为航空导航设计的模块化框架，旨在将高层的语义感知与底层的几何规划进行解耦。传统方法通常强制多模态大语言模型（MLLM）直接输出控制信号，易导致效率低下与运动不稳定；而 Fly0 则将 MLLM 重新定位为专用的语义观察器，其任务是从视觉-语言输入中识别出二维目标坐标，再通过基于深度信息的反投影准确映射到三维空间。随后，一个专用的基于梯度的规划器接管处理，生成平滑、动态合理且无碰撞的飞行轨迹。这种显式的感知-行动分离机制，为实现更鲁棒、高效和可扩展的自主飞行提供了新思路。

## 系统概览

![图 1-2](docs/fig1-2.png)

## 配置

### 配置文件结构

系统使用 JSON 配置文件（`config.json`）来管理所有设置。以下是一个完整的示例：

```json
{
    "API_TYPE": "openai",
    "OPENAI_API_KEY": "your-api-key-here",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_MODEL": "gpt-4-vision-preview",
    "OLLAMA_MODEL": "llama3.2-vision",
    "VLLM_BASE_URL": "http://localhost:8000/v1",
    "VLLM_MODEL": "your-vlm-model",
    "vision": {
        "enabled": true,
        "api_key": "your-vision-api-key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4-vision-preview"
    },
    "planner": {
        "lidar_sensors": ["LidarSensor1"]
    }
}
```

### 配置选项

#### VLM 提供商设置

**API_TYPE**：从 `"ollama"`、`"openai"` 或 `"vllm"` 中选择

**OpenAI 配置：**
- `OPENAI_API_KEY`：您的 OpenAI API 密钥
- `OPENAI_BASE_URL`：API 端点（默认：https://api.openai.com/v1）
- `OPENAI_MODEL`：模型名称（例如：gpt-4-vision-preview, qwen2.5-vl-72b-instruct）

**Ollama 配置：**
- `OLLAMA_MODEL`：模型名称（例如：llama3.2-vision, qwen2.5vl:3b-fp16）
- 默认 URL：http://localhost:11434

**VLLM 配置：**
- `VLLM_BASE_URL`：VLLM 服务器 URL
- `VLLM_MODEL`：模型名称
- `OPENAI_API_KEY`：API 密钥（对于本地服务器可以是 "EMPTY"）

#### 视觉检测设置

- `vision.enabled`：启用/禁用视觉目标检测（true/false）
- `vision.api_key`：视觉模型的 API 密钥
- `vision.base_url`：视觉模型的 API 端点
- `vision.model`：视觉模型名称

#### 路径规划器设置

- `planner.lidar_sensors`：LiDAR 传感器名称列表（默认：["LidarSensor1"]）

### 系统提示词

系统提示词（`sysprompt/sysprompt.txt`）定义了 VLM 的行为和能力。您可以自定义它来：

- 定义可用的无人机控制函数
- 设置响应格式要求
- 指定安全指南
- 添加领域特定知识

示例系统提示词：
```
你是一个无人机控制助手。你可以使用 Python 代码控制无人机。

可用函数：
- drone.takeoff(): 从地面起飞
- drone.land(): 降落无人机
- drone.get_position(): 获取当前位置
- drone.fly_to([x, y, z]): 飞到三维位置
- drone.set_yaw(yaw): 设置偏航角（度）
- drone.get_yaw(): 获取当前偏航角
- drone.fly_path(points): 沿航路点路径飞行
- drone.get_velocity(): 获取当前速度

始终使用 markdown 代码块中的 Python 代码进行响应。
```

## 依赖安装
### 步骤 1：克隆仓库

```bash
git clone https://gitee.com/brikit/fly0.git
cd fly0
```

### 步骤 2：创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n fly0 python=3.9
conda activate fly0
```

### 步骤 3：安装依赖

```bash
pip install -r env/requirements.txt
```

**注意**：由于 AirSim 的特殊依赖关系，请最后单独安装 AirSim：

```bash
pip install airsim==1.8.1 --no-build-isolation
```

### 步骤 4：安装 AirSim

按照官方 [AirSim 安装指南](https://microsoft.github.io/AirSim/build/windows/) 在您的系统上安装 AirSim。


## 快速开始

### 1. 启动 AirSim

启动您的 AirSim 环境，**请使用本项目提供的 `src/airsim/settings.json` 替换 AirSim 安装目录中的 settings.json**，确保配置了 LiDAR 和相机传感器。

### 2. 配置系统

编辑 `config.json`，填入您的 API 密钥和首选的 VLM 提供商。

### 3. 运行系统

```bash
cd src/airsim
python ./main.py
```

### 4. 发出命令

执行 `python ./main.py` 后，会打开一个终端窗口。
```
╔══════════════════════════════════════════════════════════════╗
║          🚁 Drone Natural Language Control System  🚁        ║
╚══════════════════════════════════════════════════════════════╝
Available Commands:
  !quit   - Exit program
  !clear  - Clear chat history
  !help   - Show help

Current mode: flight
Enter command or !help for help

>
```

首次使用请在终端中输入如下命令（很重要！）：

```
> 起飞，并转到0度
```

## 使用方法

### 命令行参数

```bash
cd src/airsim
python ./main.py [OPTIONS]

选项：
  --config PATH    配置文件路径（默认：config.json）
  --prompt PATH    系统提示词文件路径（默认：sysprompt/sysprompt.txt）
  --help           显示帮助信息并退出
```

### 可用命令

- `!quit` - 退出程序
- `!clear` - 清除聊天历史
- `!help` - 显示帮助信息

## 许可证

请查看 [LICENSE](LICENSE) 文件了解详细信息。
