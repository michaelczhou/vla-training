# VLA Web Demo

基于 Gradio 的交互式 VLA 模型演示界面。

## 功能

- 📷 上传场景图像
- 💬 输入自然语言指令
- 🤖 预测机器人动作序列
- 📊 可视化动作曲线

## 安装

```bash
cd web_demo
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

然后访问: http://localhost:7860

## 使用示例

1. 上传机器人工作场景图像
2. 输入指令如: "pick up the red block"
3. 点击"预测"按钮
4. 查看预测的动作序列和可视化

## 支持的指令

- `pick up the [object]` - 抓取物体
- `place the [object] on [location]` - 放置物体
- `move [object] to [direction]` - 移动物体
- `open/close/push/pull [object]` - 操作物体

## 注意事项

当前使用模拟模型进行演示。实际使用时，请在 `app.py` 中替换为训练好的 VLA 模型。
