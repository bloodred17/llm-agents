import ast
import re
from typing import List, Union

from dotenv import load_dotenv

# LangChain / custom imports
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description, Tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.callbacks import AgentCallbackHandler

load_dotenv()


###############################################################################
# Tools
###############################################################################
@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text"""
    return len(text)


@tool
def count_letter_in_word(_word: str, _letter: str) -> int:
    """Return the number of times `_letter` appears in `_word`."""
    return _word.count(_letter)


tools = [count_letter_in_word]


###############################################################################
# Utility: find a tool by name
###############################################################################
def find_tool_by_name(_tools: List[Tool], _tool_name: str) -> Tool:
    for _tool in _tools:
        if _tool.name == _tool_name:
            return _tool
    raise ValueError(f"No tool with name '{_tool_name}' found")


###############################################################################
# Prompt Template
###############################################################################
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action. The format should be a tuple. Do not deviate from the format.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = (
    PromptTemplate.from_template(template)
    # Insert a rendered list of tools and tool names
    .partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
)

###############################################################################
# LLM Setup
###############################################################################
# You can swap in ChatOpenAI if desired:
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", stop=["\nObservation"])
llm = ChatOllama(
    temperature=0,
    model="llama3.2",
    stop=["\nObservation", "Observation"],
    callbacks=[AgentCallbackHandler()],
)

###############################################################################
# Construct the chain that uses ReActSingleInputOutputParser
###############################################################################
agent_chain = (
    {
        "input": lambda x: x["input"],
        # format_log_to_str expects a list of (AgentAction|AgentFinish, str),
        # so we pass that list as `x["agent_scratchpad"]`.
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)


###############################################################################
# Run the chain in a loop until AgentFinish
###############################################################################
def main():
    intermediate_steps = []

    # 1) First invocation
    agent_step: Union[AgentAction, AgentFinish] = agent_chain.invoke(
        {
            "input": "How many 'r' in Strawberries?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    # 2) Keep looping until we get a final answer
    while not isinstance(agent_step, AgentFinish):
        print("Action:", agent_step.tool)
        print("Action Input:", agent_step.tool_input)

        # Find the right tool
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)

        # Parse the action input (expected format: ("Strawberries", "r"))
        tool_input_str = agent_step.tool_input
        word, letter = ast.literal_eval(tool_input_str)

        # Run the tool
        observation = tool_to_use.func(word, letter)
        print("Observation:", observation)

        # 3) Append the (AgentAction, "observation") to intermediate_steps
        #    format_log_to_str knows how to turn these pairs into:
        #    Thought: <agent_step.log>
        #    Observation: <observation>
        intermediate_steps.append((agent_step, str(observation)))

        # Re-invoke the chain, passing the updated scratchpad
        agent_step = agent_chain.invoke(
            {
                "input": "How many 'r' in Strawberries?",
                "agent_scratchpad": intermediate_steps,
            }
        )

    # 4) Once we have AgentFinish, print the final answer
    print("Final Answer:", agent_step.return_values)


if __name__ == "__main__":
    main()
