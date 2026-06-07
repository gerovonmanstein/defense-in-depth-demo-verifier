# Defense in Depth paper accompanying code

## Simple installation

If you just want to run the code and nothing else, you can do the following:

1. Clone the repository.
2. `cd` into it.
3. Create a new Python 3.10 virtual environment called `venv`.
4. Activate the virtual environment.
5. Install the `defense-in-depth-demo` project.

```
git clone https://github.com/AlignmentResearch/defense-in-depth-demo.git &&
cd defense-in-depth-demo &&
python -m venv venv &&
source venv/bin/activate &&
pip install .
```

Note that this project has not been tested with different versions of Python.

## Demo
Run `python demo.py` to query the defense pipeline interactively.

## Local Verifier

The `local_verifier/` folder contains small-scale reproductions of the paper suitable for local execution using Ollama. See `local_verifier/README.md` for details on running experiments with 7B/9B models and STACK attacks with reduced iterations.