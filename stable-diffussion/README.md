### Create a Python virtual environment:
```bash
python -m venv .ven
```

### Activate the virtual environment:
```bash
source .ven/bin/activate
```

### Install packages
```bash
pip install --upgrade torch diffusers transformers scipy accelerate
```

### Change the prompt and the image name, run the stable diffusion model
```bash
python run_sd.py
```