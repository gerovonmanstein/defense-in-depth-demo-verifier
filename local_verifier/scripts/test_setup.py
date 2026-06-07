#!/usr/bin/env python3
"""
Test script to verify setup without requiring Ollama to be running.
"""

import sys
from pathlib import Path

print("="*80)
print("STACK Reproduction - Setup Test")
print("="*80)

# Test 1: Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Test 2: Import dependencies
try:
    import yaml
    print("✓ PyYAML installed")
except ImportError:
    print("✗ PyYAML not installed - run: pip install pyyaml")
    sys.exit(1)

try:
    import requests
    print("✓ Requests installed")
except ImportError:
    print("✗ Requests not installed - run: pip install requests")
    sys.exit(1)

# Test 3: Check file structure
files_to_check = [
    "demo_ollama.py",
    "robust_llm/models/ollama_adapter.py",
    "attacks/baseline_bon.py",
    "attacks/stack_confirm.py",
    "attacks/utils.py",
    "prompt_templates/few_shot_input_filter.yaml",
    "prompt_templates/few_shot_output_filter.yaml",
]

all_exist = True
for file in files_to_check:
    if Path(file).exists():
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - MISSING")
        all_exist = False

if not all_exist:
    print("\n✗ Some files are missing!")
    sys.exit(1)

# Test 4: Check Ollama connectivity
print("\n" + "="*80)
print("Checking Ollama...")
print("="*80)

try:
    response = requests.get("http://localhost:11434/api/tags", timeout=2)
    if response.status_code == 200:
        models = response.json().get("models", [])
        print(f"✓ Ollama is running")
        print(f"  Available models: {len(models)}")
        
        required_models = ["qwen2.5:3b", "llama3.2:3b"]
        for model_name in required_models:
            if any(m.get("name") == model_name for m in models):
                print(f"  ✓ {model_name}")
            else:
                print(f"  ✗ {model_name} - run: ollama pull {model_name}")
    else:
        print(f"✗ Ollama returned status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("✗ Ollama is not running")
    print("  Start with: ollama serve")
    print("  Or install: curl -fsSL https://ollama.com/install.sh | sh")
except Exception as e:
    print(f"✗ Error checking Ollama: {e}")

print("\n" + "="*80)
print("Setup test complete!")
print("="*80)
print("\nNext steps:")
print("1. Ensure Ollama is running: ollama serve")
print("2. Pull models: ollama pull qwen2.5:3b && ollama pull llama3.2:3b")
print("3. Test pipeline: python demo_ollama.py --test")
print("4. Run attacks: python attacks/baseline_bon.py --queries 2 --variations 5")
