from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from llama_cpp import Llama

MODEL_PATH = "/home/asrada/바탕화면/Samsung_OCR-chatbot/models/ggml-model-Q5_K_M.gguf"

template = """<질문>: {question}

<대답> : 단계별로 생각해서 대답해줘."""

prompt = PromptTemplate.from_template(template)
#callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_threads=2, # CPU cores
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=43, # Change this value based on your model and your GPU VRAM pool.
    n_ctx=4096, # Context window
)

llm_chain = prompt | llm
question = "대한민국의 수도는 어디입니까?"
#llm_chain.invoke({"question": "대한민국의 수도는 어디입니까?"})
print(llm_chain.invoke({"question": "대한민국의 수도는 어디입니까?"}))