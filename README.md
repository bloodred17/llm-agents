# LLM Agents 🦜🔗🕸️

## Requirements
- [Poetry - Package manager for Python](https://python-poetry.org/)
- `.env`

```.dotenv
# .env contents
OPENAI_API_KEY=
PINECONE_API_KEY=
TAVILY_API_KEY=
LANGSMITH_TRACING=
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

> Except for OpenAI all other API keys are available on a Free tier.

> You can choose to replace OpenAI with Llama if you have it locally installed or any other llm of your choice.

## Setup
```sh
poetry install
```

## Run
```sh
poetry run python <agent>
```

### Agents
Contains four agents
- app (LangChain - ZeroShot)
- rag (LangChain - ReAct)
- reflection-agent (LangGraph - Reflection)
- reflexion-agent (LangGraph - Reflexion)
