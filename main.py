from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from tools.shell import run_batch_script_tool
from tools.file_tree import get_file_tree_tool

load_dotenv()

chat = ChatOpenAI()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content = (
            "You are an AI that has access to the integrated terminal of Visual Studio Code IDE on windows. Follow the following rules while responding to user prompts - \n"
            "1. Always use the 'get_file_tree_tool' before responding to a user prompt. the name of the root folder for this project is C:\\Users\\ketan\\Desktop\\Alfred\\poc2. All the changes are to be made under the poc2 folder." 
            "2. If you need to make any changes to the file structure (adding or modifying or deleting files), run a batch script to do so."
            "3. If you need to run any command in the terminal, use the 'run_batch_script_tool'."
            )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [run_batch_script_tool, get_file_tree_tool]

agent = OpenAIFunctionsAgent(
    llm = chat,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(
    agent = agent,
    verbose = True,
    tools = tools,
    memory=memory
)

# agent_executor("How many of the users have provided a shipping address?")
# new prompt :
while True:
    user_input = input(">> ")
    agent_executor.invoke(user_input)

# agent_executor("Repeat the same process for users.")


