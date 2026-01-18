# LLM Sandbox - Usage

## Running Both Servers
Both  the Unreal Server and the Web UI must be running to use this:
### Unreal Side
From the menu bar:
`LLM Sandbox → Start Server`
- The server runs at `http://127.0.0.1:5002`
  
### Web Interface Side
- Start `llm-sandbox` from command line ( outside of Unreal )
- Open `http://localhost:5001/notebook/notebook.ipynb` ( or any `ipynb` name )

## Overall Notebook Functionality

The web app provides:
- **Notebook Interface** - Jupyter-style interface
- `Code Cells` - Write and executing Python code in Unreal
- `Markdown Cells` - Write notes in Markdown
- `Prompt Cells` - Chat with LLMs that have full context of your notebook + agentic control of Unreal

### Saving/Loading
To navigate to a notebook, put `http://localhost:5001/notebook/NotebookName.ipynb` in the browser
- If the notebook exists, it will be opened
- If not it will be added
- The notebook is autosaved, every 5 seconds!

### Renaming 
This functionality is simpler at the momment: If you click on the title on the left, you can change the notebook name.
- This will cause all future automatic saves to be directed to the new name
- This does not delete the original notebook 

### Add Cell
Click one of the + buttons in the upper right of the notebook:
- A notebook of that type will be added at the bottom of the page
- TODO: Checkbox to add under highltighted cell

### Re-organize Notebook
Each cell has same buttons in its upper right:
- Minimize/Maximize
- Move Up/Move Down
- Delete

## Cell Types

### Code Cell
This cell stores and runs code, directly in the Unreal Engine Kernel
- Variables are stored between cells
- You can register new tools using the `@register_tool` here
- Any print statements or error output are displayed in the bottom of the cell
Buttons:
- Play: Runs code
- Stop: Interrupts running code
- Sweep: Clears output

### Prompt Cell
This cell sends a prompt to an LLM
- All of the notebook above the cell, will be visible to the LLM
    - This includes code output and errors ( wonderfull for debugging )
Buttons:
- Play: Runs code
- Stop: Interrupts LLM output
- Sweep: Clears output
- Paper/Pencil: Toggles input between edit mode, and markdown mode
- Hammer/Wrench: Toggles Unreal tool usage
  - If Hammer/Wrench is toggled, the Unreal tools are a available, and its encouraged to use them
  - Otherwise its encouraged to help generate code examples, and can only use web searching tools

### Markdown Cell
This cell is for notes, which are displayed in Markdown when not editing
Buttons:
- Paper/Pencil: Toggles input between edit mode, and markdown mode