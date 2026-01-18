# LLM Sandbox Unreal

<p align="center">
  <img src="assets/demo.gif" alt="Demo">
</p>
LLM Sandbox (Unreal) is an Unreal Engine plugin that runs a local Python execution server inside the editor, allowing external tools and LLMs to inspect and manipulate Unreal scenes in real time.

## The Whole Package
- Web-Interface: [llm-sandbox-ui](https://github.com/NeuralVFX/llm-sandbox-ui)
- Unreal Plugin: [llm-sandbox-unreal](https://github.com/NeuralVFX/llm-sandbox-unreal)   <--- You are here
  
Together, the components enable:
- Executing Python code inside Unreal Engine from a web interface
- LLM-assisted interaction with full visibility into code, output, and errors
- Creation and execution of agentic tools that operate directly on Unreal scenes

## Documentation

- **[Setup](SETUP.md)** - Install / Initialize (Web UI Specific Instructions)
- **[Notebook Usage](https://github.com/NeuralVFX/llm-sandbox-ui/blob/main/docs/USAGE.md)** - Work with notebooks and cell types
- **[Agent Customization](https://github.com/NeuralVFX/llm-sandbox-ui/blob/main/docs/TOOLS.md)** - Register and extend agentic tools

## Requirements
- Unreal Engine 5.6
- The Unreal plugin requires the Web Interface to be installed and running in order to use notebooks and LLM features.
   - **[LLM Sandbox Web Interface](https://github.com/NeuralVFX/llm-sandbox-ui)** ( installation instruction provided )
- `OpenAI` **API Key**

