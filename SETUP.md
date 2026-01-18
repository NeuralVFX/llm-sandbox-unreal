# LLM Sandbox Unreal — Setup

Follow these steps to install and initialize the LLM Sandbox Unreal plugin.

## Installation

1. Copy the `LLMSandbox` folder into your project’s `Plugins/` directory

2. Restart Unreal Engine

3. Open **Edit → Plugins** and enable:
   - **Python Editor Script Plugin**
   - **Python Foundation Packages**
   - **LLM Sandbox**
     
![unreal plugin](assets/unreal_plugin.jpg)
![other plugins](assets/other_plugins.jpg)


4. Restart Unreal Engine

5. Find the new menu `LLM Sandbox Tools`, click: `LLM Sandbox Tools -> Install Dependencies`
   - Installs required Python packages into the project:
     - `flask`
     - `ipython`
     - `lisette`

6. Restart Unreal Engine

## Initialization
 1. In Unreal click `LLM Sandbox Tools -> Start Server`
 2. Follow instructions at [LLM Sandbox Web Interface](https://github.com/NeuralVFX/llm-sandbox-ui) to start Web Interface
 3. Check [USAGE](docs/USAGE.md) for notebook usage directions

## Requirements

- Unreal Engine 5.6
- The Unreal plugin requires the Web Interface to be installed and running in order to use notebooks and LLM features.
   - **[LLM Sandbox Web Interface](https://github.com/NeuralVFX/llm-sandbox-ui)** ( installation instruction provided )
- `OpenAI` Api Key

