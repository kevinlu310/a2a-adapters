# Quick Start Guide

**Get your first A2A agent running in 5 minutes!** ⚡

This guide will help you expose your existing agent (n8n workflow, CrewAI crew, or LangChain chain) as an A2A-compatible agent.

## Prerequisites

- Python 3.11+
- An agent to expose (n8n workflow, CrewAI crew, LangChain chain, or custom function)

## Step 1: Install

```bash
pip install a2a-adapter
```

For specific frameworks:

```bash
pip install a2a-adapter[n8n]
pip install a2a-adapter[crewai]    # For CrewAI
pip install a2a-adapter[langchain] # For LangChain
pip install a2a-adapter[all]        # Install all
```

## Step 2: Create Your Agent

Choose your framework:

### Option A: n8n Workflow

```python
# my_agent.py
import asyncio
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

async def main():
    adapter = await load_a2a_agent({
        "adapter": "n8n",
        "webhook_url": "https://your-n8n.com/webhook/workflow-id"
    })

    serve_agent(
        agent_card=AgentCard(
            name="My N8n Agent",
            description="My n8n workflow as A2A agent"
        ),
        adapter=adapter,
        port=9000
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Option B: CrewAI Crew

```python
# my_agent.py
import asyncio
from crewai import Crew, Agent, Task
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

# Create your crew
crew = Crew(agents=[...], tasks=[...])

async def main():
    adapter = await load_a2a_agent({
        "adapter": "crewai",
        "crew": crew
    })

    serve_agent(
        agent_card=AgentCard(name="Research Crew", description="..."),
        adapter=adapter,
        port=9000
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Option C: LangChain Chain

```python
# my_agent.py
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

# Create chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("user", "{input}")
])
chain = prompt | ChatOpenAI(model="gpt-4o-mini", streaming=True)

async def main():
    adapter = await load_a2a_agent({
        "adapter": "langchain",
        "runnable": chain,
        "input_key": "input"
    })

    serve_agent(
        agent_card=AgentCard(name="Chat Agent", description="..."),
        adapter=adapter,
        port=9000
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Option D: Custom Function

```python
# my_agent.py
import asyncio
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

async def my_agent(inputs: dict) -> str:
    return f"Echo: {inputs['message']}"

async def main():
    adapter = await load_a2a_agent({
        "adapter": "callable",
        "callable": my_agent
    })

    serve_agent(
        agent_card=AgentCard(name="Echo Agent", description="..."),
        adapter=adapter,
        port=9000
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 3: Run Your Agent

```bash
python my_agent.py
```

Your agent is now running at `http://localhost:9000`! 🎉

## Step 4: Test Your Agent

### Using Python Client

```python
# test_agent.py
import asyncio
from a2a.client import A2AClient
from a2a.types import Message, MessageSendParams, TextPart

async def main():
    client = A2AClient(base_url="http://localhost:9000")

    response = await client.send_message(MessageSendParams(
        messages=[Message(
            role="user",
            content=[TextPart(type="text", text="Hello!")]
        )]
    ))

    print(f"Agent says: {response.content[0].text}")
    await client.close()

asyncio.run(main())
```

### Using curl

```bash
curl -X POST http://localhost:9000/messages \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "messages.send",
    "params": {
      "messages": [{
        "role": "user",
        "content": [{"type": "text", "text": "Hello!"}]
      }]
    },
    "id": 1
  }'
```

## 🎯 What's Next?

### ✅ Supported Frameworks

- **n8n** - Expose workflows as A2A agents
- **CrewAI** - Multi-agent crews as A2A agents
- **LangChain** - Chains and agents with streaming support
- **Custom** - Any async function as an A2A agent

### 📚 Next Steps

1. **Explore Examples** - Check out [examples/](examples/) for complete working code
2. **Read Documentation** - See [README.md](README.md) for full API reference
3. **Build Multi-Agent Systems** - Connect multiple A2A agents together
4. **Create Custom Adapters** - Integrate your own frameworks

### 🔧 Advanced Usage

#### Multi-Agent Communication

```python
from a2a.client import A2AClient

# Connect to multiple agents
math_agent = A2AClient(base_url="http://localhost:9000")
research_agent = A2AClient(base_url="http://localhost:8001")

# Call agents
result1 = await math_agent.send_message(...)
result2 = await research_agent.send_message(...)
```

#### Streaming (LangChain)

```python
async for chunk in client.send_message_stream(params):
    print(chunk, end="", flush=True)
```

#### Custom Adapter Class

For full control, subclass `BaseAgentAdapter`:

```python
from a2a_adapter import BaseAgentAdapter
from a2a.types import Message, MessageSendParams, TextPart

class MyAdapter(BaseAgentAdapter):
    async def to_framework(self, params):
        return {"input": params.messages[-1].content[0].text}

    async def call_framework(self, input, params):
        return {"output": process(input["input"])}

    async def from_framework(self, output, params):
        return Message(
            role="assistant",
            content=[TextPart(type="text", text=output["output"])]
        )
```

## 🐛 Troubleshooting

### Common Issues

**1. "Webhook URL not accessible" (n8n)**

- Ensure n8n workflow is active and published
- Verify webhook URL is correct
- Check n8n instance is reachable

**2. Import errors**

```bash
pip install a2a-adapter[framework-name]  # e.g., [crewai], [langchain]
```

**3. Port already in use**

```bash
serve_agent(..., port=8001)  # Use different port
```

**4. Connection refused**

```bash
curl http://localhost:9000/health  # Check if server is running
```

**5. Missing API keys**

```bash
export OPENAI_API_KEY="your-key"  # For CrewAI/LangChain
```

📖 **Need more help?** Check [GETTING_STARTED_DEBUG.md](GETTING_STARTED_DEBUG.md) for detailed debugging guide.

## 📚 Additional Resources

- 📖 [Full Documentation](README.md) - Complete API reference
- 🏗️ [Architecture Guide](ARCHITECTURE.md) - Design and implementation details
- 💻 [Examples](examples/) - Complete working examples
- 🐛 [Debug Guide](GETTING_STARTED_DEBUG.md) - Troubleshooting and debugging
- 🤝 [Contributing](CONTRIBUTING.md) - How to contribute to the project

---

**🎉 Congratulations!** You've successfully created your first A2A agent!

**Next:** Explore the [examples/](examples/) directory to see more advanced use cases.
