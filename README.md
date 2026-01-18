# LLM Sandbox Unreal

<p align="center">
  <img src="assets/demo.gif" alt="Demo">
</p>


LLM Sandbox is a two-part system that links a web-based notebook interface with Unreal Engine,
allowing Python code and LLMs to interact directly with the editor and scene.

#### This project has two components:
- Web-Interface: [llm-sandbox-ui](https://github.com/NeuralVFX/llm-sandbox-ui)  
- Unreal Plugin: [llm-sandbox-unreal](https://github.com/NeuralVFX/llm-sandbox-unreal)  <--- You are here
  
Together, the components enable:
- Executing Python code inside Unreal Engine from a web interface
- LLM-assisted interaction with full visibility into code, output, and errors
- Creation and execution of agentic tools that operate directly on Unreal scenes
  
## Documentation

- **[Setup](SETUP.md)** — Install ( Unreal Specific Instructions )
- **[Notebook Usage](docs/USAGE.md)** — Work with notebooks and cell types
- **[Agent Customization](docs/TOOLS.md)** — Register and extend agentic tools

## Requirements

- Unreal Engine 5.6
- Python enabled in Unreal
- LLM Sandbox Web UI
