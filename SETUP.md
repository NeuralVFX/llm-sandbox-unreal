# LLM Sandbox Unreal — Setup

Follow these steps to install and initialize the LLM Sandbox Unreal plugin.

## Installation & Initialization

1. Copy the `LLMSandbox` folder into your project’s `Plugins/` directory

2. Restart Unreal Engine

3. Open **Edit → Plugins** and enable:
   - **Python Editor Script Plugin**
   - **Python Foundation Packages**
   - **LLM Sandbox**

4. Restart Unreal Engine

5. From the **LLM Sandbox Tools** menu, click **Install Dependencies**
   - Installs required Python packages into the project:
     - `flask`
     - `ipython`
     - `lisette`

6. Restart Unreal Engine

## Requirements

- Unreal Engine 5.6
- **[LLM Sandbox Web Interface](https://github.com/NeuralVFX/llm-sandbox-ui)** ( installation instruction provided )
- The Unreal plugin requires the Web Interface to be installed and running in order to use notebooks and LLM features.
