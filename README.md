### uv
```bash
uv venv
uv pip install -r pyproject.toml
# OR
uv pip install -r requirements.txt  # Install from a requirements.txt file. OR
uv pip sync requirements.txt  # Install dependencies from a requirements.txt file.

source .venv/bin/activate
deactivate

uv add boto3  # Add to pyproject.toml

uv export --no-hashes --format requirements-txt > requirements.txt
```

### Poetry
```bash
# Poetry
poetry install --no-root
source .venv/bin/activate
python utility-chain.py # Download pdf for examples
# With poetry there is pip
```

// NLTK_DATA=PATH_
// langchain=0.0.216 Plat