# LLM Sandbox Unreal

<p align="center">
  <img src="assets/demo.gif" alt="Demo">
</p>

#### This project has two components:
- Web-Interface: [llm-sandbox-ui](https://github.com/NeuralVFX/llm-sandbox-ui)  
- Unreal Plugin: [llm-sandbox-unreal](https://github.com/NeuralVFX/llm-sandbox-unreal)  <--- You are here
  
# What is this?

An Unreal Engine plugin that exposes a Python execution server, enabling:
- A web-based coding interface connected directly to Unreal
- LLM-assisted code review with full context of your code and errors
- Creation and deployment of agentic tools that operate inside the engine

The Unreal plugin is responsible for **execution, scene access, and tool hosting**.  
The web interface handles **notebooks, LLM interaction, and UI**.

---

## Features

- **Code Execution** – All Python code is executed directly inside Unreal Engine
- **LLM Assistance** – Ask an LLM for help with full code and error context
- **Agentic Tool Use** – LLMs can invoke tools that manipulate the Unreal scene
- **Custom Tools** – Register your own tools for LLM-driven workflows

---

## Getting Started

This plugin is one half of a two-part system. To get everything running:

- **Setup & Startup**  
  ➡️ See [SETUP.md](SETUP.md) for installation, configuration, and starting the Unreal server

- **Using the System**  
  ➡️ See [docs/USAGE.md](docs/USAGE.md) for how to work with notebooks, cell types, and workflows

- **Extending with Agentic Tools**  
  ➡️ See [docs/TOOLS.md](docs/TOOLS.md) for registering custom tools and schema overrides

---

## Related Repository

- **Web Interface**: https://github.com/NeuralVFX/llm-sandbox-ui  
  Jupyter-style notebook UI, LLM prompt cells, and client-side execution logic

---

## Requirements

- Unreal Engine 5.6
- Python enabled in Unreal
- A working LLM Sandbox Web Interface
