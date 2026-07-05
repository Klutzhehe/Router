import json
import os
import re

notebook_path = "notebooks/Train_PCB_Router.ipynb"

if os.path.exists(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source_str = "".join(cell["source"])
            if "def on_update" in source_str:
                # Find all occurrences of the Gradio block (both with variations in line endings or comments)
                # We can replace them all with empty space first, then insert exactly one clean copy.
                
                # Regex pattern matching any variation of the Gradio optional launch block
                pattern = r"(?:# ──+ Optional live Gradio server during training ──+\n)?if CONFIG\.get\(\"LAUNCH_GRADIO_DURING_TRAINING\", False\):\n\s+try:\n\s+import gradio\n\s+except ImportError:\n\s+import subprocess\n\s+subprocess\.run\(\[sys\.executable, \"-m\", \"pip\", \"install\", \"-q\", \"gradio\"\], check=True\)\n\s+try:\n\s+from scripts\.visualize_routing_gradio import build_gradio_app\n\s+is_dreamer = hasattr\(trainer, 'jepa'\)\n\s+cur_cfg = {\"stages\": trainer\.curriculum\.stages}\n\s+print\(\"\\n\[Gradio\] Launching live step-by-step routing visualizer\.\.\.\"\)\n\s+demo = build_gradio_app\(trainer, is_dreamer, cur_cfg\)\n\s+# share=True creates a public URL\.(?:\s+prevent_thread_warnings=True)?\n\s+demo\.launch\(share=True(?:, prevent_thread_warnings=True)?\)\n\s+print\(\"\[Gradio\] Live server is running! Open the public URL in a new tab to route boards interactively with active weights\.\\n\"\)\n\s+except Exception as e:\n\s+print\(f\"\\n\[Gradio\] Warning: Failed to launch live visualizer: \{e\}\\n\"\)\n*"
                
                # Replace all occurrences with empty string
                source_str_cleaned = re.sub(pattern, "", source_str)
                
                # Add exactly one clean copy right before the starting training print statement
                gradio_block = """# ── Optional live Gradio server during training ───────────────────
if CONFIG.get("LAUNCH_GRADIO_DURING_TRAINING", False):
    try:
        import gradio
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
    try:
        from scripts.visualize_routing_gradio import build_gradio_app
        is_dreamer = hasattr(trainer, 'jepa')
        cur_cfg = {"stages": trainer.curriculum.stages}
        print("\\n[Gradio] Launching live step-by-step routing visualizer...")
        demo = build_gradio_app(trainer, is_dreamer, cur_cfg)
        # share=True creates a public URL.
        demo.launch(share=True)
        print("[Gradio] Live server is running! Open the public URL in a new tab to route boards interactively with active weights.\\n")
    except Exception as e:
        print(f"\\n[Gradio] Warning: Failed to launch live visualizer: {e}\\n")

"""
                target_str = 'print(f"Starting training for {CONFIG[\'TOTAL_TIMESTEPS\']:,} timesteps...")'
                if target_str in source_str_cleaned:
                    source_str_cleaned = source_str_cleaned.replace(target_str, gradio_block + target_str)
                else:
                    # Try double quotes version
                    target_str_alt = 'print(f"Starting training for {CONFIG["TOTAL_TIMESTEPS"]:,} timesteps...")'
                    if target_str_alt in source_str_cleaned:
                        source_str_cleaned = source_str_cleaned.replace(target_str_alt, gradio_block + target_str_alt)
                
                # Convert back to list of lines for JSON notebook format
                cell["source"] = [line + "\n" for line in source_str_cleaned.split("\n")]
                # Strip last empty element if split created one
                if cell["source"][-1] == "\n":
                    cell["source"][-1] = ""
                    
                print("Cleaned up training cell in notebook!")
                
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook saved successfully.")
else:
    print("Notebook path not found.")
