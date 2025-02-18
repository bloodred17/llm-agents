from typing import Sequence, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import MessageGraph, END

from chains import generate_chain, reflect_chain

load_dotenv()


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


if __name__ == "__main__":
    builder = MessageGraph()
    builder.add_node(GENERATE, generation_node)
    builder.add_node(REFLECT, reflection_node)
    builder.set_entry_point(GENERATE)
    builder.add_conditional_edges(GENERATE, should_continue)
    builder.add_edge(REFLECT, GENERATE)
    graph = builder.compile()

    # print(graph.get_graph().draw_mermaid())

    inputs = HumanMessage(
        content="""Make this tweet better:"
            @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post
        """
    )
    response = graph.invoke(inputs)
    print(response)
