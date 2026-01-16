import time
import flask, threading, sys, io
from IPython.core.interactiveshell import InteractiveShell
from io import StringIO
import traceback
from IPython.core.ultratb import VerboseTB
from IPython.core.ultratb import FormattedTB
from lisette import lite_mk_func
from queue import Queue
import uuid
import json
import glob
from flask import Flask, request, Response, jsonify
from werkzeug.serving import make_server
import litellm
import importlib
import os
import importlib.util
import types

from tool_schema import *


try:
    import unreal
except:
    unreal =None


app = flask.Flask(__name__) 


shell = InteractiveShell.instance()
shell.user_ns['unreal'] = unreal

ftb = FormattedTB(mode='Plain')
vtb = VerboseTB(theme_name='Linux')


def custom_showsyntaxerror(filename=None, running_compiled_code=False):
    pass  # Do nothing - we're handling syntax errors ourselves

def custom_showtraceback(exc_tuple=None, filename=None, tb_offset=None, exception_only=False, running_compiled_code=False):
    pass  # Do nothing - we're handling tracebacks ourselves


####################################
# TOOL REGISTRY (dynamic)
####################################

TOOLS = {}
TOOL_SCHEMAS = []
execution_queue = Queue()

def register_tool(patches=None,debug=True):
    """
    Register a function (or ToolSchema wrapper) as a tool.
    Supports both auto-generated schemas and custom pre-defined schemas.
    """
    def decorator(func):

        name = func.__name__
        schema = lite_mk_func(func)

        # Replace Schema pieces if needed
        schema_good = True
        if patches:
            patch_schema(schema,patches)
            if debug:
                schema_good = schema_unit_test(schema)

        if not schema_good:
            print(f"Unable To Register, Bad Schema: {name}")
            
        else:
            print(f"Registering Tool: {name}")

            # Expose to shell namespace
            TOOLS[name] = func
            shell.user_ns[name] = func

            # 3. Register the Schema
            # Remove old schema if exists
            TOOL_SCHEMAS[:] = [s for s in TOOL_SCHEMAS if s['function']['name'] != name]
            TOOL_SCHEMAS.append(schema)
            shell.user_ns['TOOL_SCHEMAS'] = TOOL_SCHEMAS
        
        return func

    # Allows us to use the decorator without ()    
    if callable(patches):
        func = patches
        patches = None # So this doenst get used in above func
        return decorator(func)

    return decorator


####################################
# Register Default Tools / Find User Added Tools
####################################


def load_tools(tools_dir: str):
    os.makedirs(tools_dir, exist_ok=True)

    # 1. Ensure the registry dict exists in the notebook too, 
    #    so the decorator can write to it visibly.
    if 'TOOLS' not in shell.user_ns:
        shell.user_ns['TOOLS'] = TOOLS

    for path in glob.glob(os.path.join(tools_dir, "*.py")):
        fname = os.path.basename(path)
        if fname.startswith(("_", "__")):
            continue

        print(f"Loading Script: {fname}")

        with open(path, 'r') as f:
            code_content = f.read()

        try:
            # Compile first to attach the filename path
            # 'exec' mode tells Python this is a suite of statements
            code_obj = compile(code_content, path, 'exec')
            
            # Execute the compiled object (which now has metadata)
            exec(code_obj, shell.user_ns) 
            
            unreal.log(f"Injected tool script: {fname}")
            
        except Exception as e:
            # This will likely show you the "inspect" error if it still fails
            unreal.log_error(f"Failed to inject {fname}: {e}")
            import traceback
            traceback.print_exc() # Prin

shell.user_ns['register_tool'] = register_tool

####################################
# MAIN THREAD EXECUTION
####################################

PERSISTENT_NAMESPACE = {
    '__builtins__': __builtins__,
    'unreal': unreal,
    'register_tool': register_tool, 

}


class StreamingWriter:
    def __init__(self, queue, name='stdout'):
        self.queue, self.name = queue, name
    def write(self, text):
        if text:
            self.queue.put({'msg_type': 'stream', 'content': {'name': self.name, 'text': text}})
    def flush(self): pass


def execute_code_streaming(task,queue):
    """ Execute code and capture STDERR STDOUT EXEC and TRACEBACK """
    
    tb = None
    exc = None
    r = shell.run_cell(task, store_history=True, silent=False)

    if r.error_before_exec:
        tb = ftb.structured_traceback(type(r.error_before_exec), r.error_before_exec, None)
        exc = r.error_before_exec

    elif r.error_in_exec:
        tb = vtb.structured_traceback((type(r.error_in_exec)), r.error_in_exec, r.error_in_exec.__traceback__, tb_offset=1)
        exc = r.error_in_exec

    if r.result is not None:
        queue.put({
                    'msg_type': 'execute_result',
                    'content': {
                        'data': {'text/plain': repr(r.result)},
                        'execution_count': 1
                    }
                    })
    if exc:
        queue.put({
                    'msg_type': 'error',
                    'content': {
                        'ename': type(exc).__name__,
                        'evalue': str(exc),
                        'traceback': tb
                    }
                })
    queue.put(None)

def tick_executor(delta_time):
    """  Execute Code During Tick in Unreal Main Thread """
    global task

    # Tool Execution
    try:
        while True:
            tool_task = execution_queue.get_nowait()
            tool_task['result_queue'].put(tool_task['func']())
    except:
        pass

    # Code Execution
    
    if task is not None:
        request_id, code = task
        queue =  code_outputs.get(request_id)
        # Lets Swap Outpus Here
        stdout_capture,stderr_capture =  StreamingWriter(queue,'stdout'), StreamingWriter(queue,'stderr')
        old_stdout,old_stderr = sys.stdout,sys.stderr
        sys.stdout,sys.stderr = stdout_capture,stderr_capture

        # Lets Swap the TraceBack ( Or We Get Duplicate in STDOUT )
        old_showtraceback = shell.showtraceback
        shell.showtraceback = custom_showtraceback  
        old_showsyntaxerror = shell.showsyntaxerror
        shell.showsyntaxerror = custom_showsyntaxerror

        execute_code_streaming(code,queue)

        # Return Normal Outputs
        shell.showtraceback = old_showtraceback  
        shell.showsyntaxerror = old_showsyntaxerror
        sys.stdout,sys.stderr = old_stdout,old_stderr

        task = None

####################################
# FLASK ROUTES
####################################

@app.route('/tools', methods=['GET'])
def get_tools():
    """Return available tool schemas."""
    return jsonify(TOOL_SCHEMAS)


@app.route('/execute_tool', methods=['POST'])
def execute_tool():
    """Execute a single tool on main thread."""
    data = request.json
    func_name = data.get('function')
    args = data.get('arguments', {})
    
    func = TOOLS.get(func_name)
    if not func:
        return jsonify({'error': f"Unknown tool: {func_name}"}), 400
    
    result_queue = Queue()
    execution_queue.put({
        'func': lambda: func(**args),
        'result_queue': result_queue
    })
    
    try:
        result = result_queue.get(timeout=30)
        return jsonify({'result': str(result)})
    except Exception as e:
        unreal.log(f"Tool error [{func_name}]: {e}")
        return jsonify({'error': str(e)}), 500


task = None
code_outputs = {}


@app.route('/execute', methods=['POST'])
def execute():
    """Execute Python code on main thread with Jupyter-style streaming."""
    global task
    global queue
    

    code = request.json.get('code', '')
    request_id = str(uuid.uuid4())
    queue = Queue()
    code_outputs[request_id] = queue
    task = (request_id, code)
    
    unreal.log(f"🎬 Starting generator for {request_id}") 
    
    def generate():
        try:
            while True:
                msg = queue.get(timeout=30)
                unreal.log(f"Generator got: {msg}")  
                if msg is None:
                    break
                yield f'data: {json.dumps(msg)}\n\n'
        except Exception as e:
            unreal.log(f"Generator error: {e}") 
        finally:
            code_outputs.pop(request_id, None)
            unreal.log(f"Generator finished")  
    
    return Response(
        generate(), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


####################################
# Register/Unregister Tick Callback 
# For Unreal MAIN Thread
####################################

def cleanup():
    """ Cleanup any existing Sandbox theads and ticks"""
    if hasattr(unreal, '_ipy_hnd'): 
        unreal.unregister_slate_pre_tick_callback(unreal._ipy_hnd)
        del unreal._ipy_hnd
        unreal.log('Stopped: LLM Sandbox Tick Callback')

    if hasattr(unreal, '_flask_server'):
        unreal._flask_server.shutdown()
        del unreal._flask_server
        unreal.log('Stopped: LLM Sandbox Flask Server')


def register_callback():
    """ Setup the Sandbox server """
    try:
        cleanup()

        # 1. Reset the Registry so we don't have stale tools
        TOOLS.clear()
        TOOL_SCHEMAS.clear()
        
        # 2. Re-Load the Tools from disk (This picks up updates!)
        unreal.log('--- Reloading Tools ---')
        load_tools(os.path.join(os.path.dirname(__file__), "default_tools"))
        load_tools(os.path.join(unreal.Paths.project_content_dir(), "Python", "tools"))

        # Register Tick
        unreal._ipy_hnd = unreal.register_slate_pre_tick_callback(tick_executor)

        # Create and Store Server
        server = make_server('127.0.0.1',5002,app)
        unreal._flask_server = server

        # Start Serve in Thread
        threading.Thread(target=server.serve_forever, daemon=True).start()
        unreal.log('Started: LLM Sandbox')
    except Exception as e:
        unreal.log(f'Failed to Start: LLM Sandbox {e}')
    
    try:
        unreal.log('Loaded: LLM Sandbox User Agent Tools')

    except Exception as e:
        unreal.log(f'Failed Loading: LLM Sandbox User Agent Tools {e}')    


def unregister_callback():
    """ Shutdown the Sandbox server """
    try:
        cleanup()

    except Exception as e:
        unreal.log(f'Failed to Stop: LLM Sandbox {e}')

