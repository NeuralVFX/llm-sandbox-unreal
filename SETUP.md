# LLM Sandbox Unreal - Setup

1. Copy the `LLMSandbox` folder to your project's `Plugins/` directory
2. Restart Unreal Engine
3. Go to `Edit → Plugins`, and enable the plugins:
   - `Python Editor Script Plugin`
   - `Python Foundation Packages`
   - `LLM Sandbox`
4. Restart Unreal Engine
5. From the `LLM Sandbox Tools` menu, click `Install Dependencies`
   - Adds required python packages to the project (`flask`, `ipython`, `lisette`)
7. Restart Unreal Engine

# Requirements

- Unreal Engine 5.6
- A conda env for the web-server

