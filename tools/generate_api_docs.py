
import inspect
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

import tgraph.transform
import tgraph.visualization

def get_doc(obj):
    return inspect.getdoc(obj) or "No documentation available."

def format_signature(obj):
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except ValueError:
        return "()"

def generate_markdown_for_module(module, module_name):
    md = f"## Module: `{module_name}`\n\n"
    doc = get_doc(module)
    if doc != "No documentation available.":
         md += f"{doc}\n\n"
    
    # Get all classes and functions
    all_members = inspect.getmembers(module)
    
    # Filter for locally defined or exposed members
    classes = []
    functions = []
    
    for name, obj in all_members:
        if name.startswith("_"): continue
        
        # Check if defined in this module or explicitly exposed packages
        if inspect.isclass(obj):
             if obj.__module__ == module.__name__ or module.__name__ in obj.__module__:
                 classes.append((name, obj))
        elif inspect.isfunction(obj):
             if obj.__module__ == module.__name__ or module.__name__ in obj.__module__:
                 functions.append((name, obj))

    if classes:
        md += "### Classes\n\n"
        for name, cls in classes:
            md += f"#### `{name}`\n\n"
            md += f"```python\nclass {name}{format_signature(cls)}\n```\n\n"
            md += f"{get_doc(cls)}\n\n"
            
            # Methods
            methods = inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
            
            # Filter methods
            public_methods = []
            for m_name, m_obj in methods:
                if m_name.startswith("_") and m_name != "__init__": continue
                # simple filter to avoid inherited noise if desired, but user wants full api. 
                # Let's keep all public + init.
                public_methods.append((m_name, m_obj))
            
            if public_methods:
                md += "**Methods:**\n\n"
                for m_name, m_obj in public_methods:
                    md += f"*   `{m_name}{format_signature(m_obj)}`\n"
                    doc = get_doc(m_obj)
                    if doc and doc != "No documentation available.":
                        # Indent docstring
                        doc_lines = doc.split('\n')
                        for line in doc_lines:
                            md += f"    > {line}\n"
                    md += "\n"

    if functions:
        md += "### Functions\n\n"
        for name, func in functions:
            md += f"#### `{name}`\n\n"
            md += f"```python\ndef {name}{format_signature(func)}\n```\n\n"
            md += f"{get_doc(func)}\n\n"

    return md

if __name__ == "__main__":
    # 1. Read version from pyproject.toml
    version = "0.0.0"
    try:
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
            version = data["project"]["version"]
    except ImportError:
        # Fallback for older python or if tomllib missing, simple parse
        with open("pyproject.toml", "r") as f:
            for line in f:
                if line.strip().startswith("version ="):
                    version = line.split("=")[1].strip().strip('"')
                    break
    except Exception as e:
        print(f"Warning: Could not read version: {e}")

    output_file = f"api_doc_v{version}.md"
    
    print(f"Generating API documentation for version {version} -> {output_file}")

    content = f"# tgraph API Reference (v{version})\n\n"
    content += "This document was automatically generated from the source code docstrings.\n\n"
    content += generate_markdown_for_module(tgraph.transform, "tgraph.transform")
    content += "\n---\n\n"
    content += generate_markdown_for_module(tgraph.visualization, "tgraph.visualization")
    
    with open(output_file, "w") as f:
        f.write(content)
    
    print("Done.")
