# Dead Code Detection and PR Generation Agent

An AI-powered agent built with Langraph that analyzes codebases to identify dead code, generates comprehensive reports, and automatically creates pull requests with suggested changes.

## Features

- **Codebase Analysis**: Identifies unused functions, classes, variables, and imports across multiple programming languages (Python, JavaScript, TypeScript, Java, C++).
- **Detailed Reporting**: Provides contextual analysis explaining why code is considered dead.
- **Automated PR Creation**: Generates GitHub Pull Requests with necessary changes to remove dead code.
- **Configurable Settings**: Customize thresholds and language-specific rules.
- **Scalable Architecture**: Optimized for handling large codebases efficiently.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd dead_code_agent

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root with the following variables:
```
GITHUB_TOKEN=your_github_personal_access_token
OPENAI_API_KEY=your_openai_api_key
```

2. Configure the agent settings in `config.yaml` (see example configuration).

## Usage

```bash
# Run the agent on a specific repository
python src/main.py --repo username/repository --branch main
```

## Advanced Configuration

See the `config.yaml.example` file for detailed configuration options including:
- Language-specific settings
- Dead code detection thresholds
- PR creation preferences
- Analysis depth options

## License

MIT
