import torch
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Dict, Union, Optional
from pydantic import BaseModel




# 定义输入数据模型
class ChatRequest(BaseModel):
    context: List[Dict[str, Union[str, Dict[str, Union[int, str, float]], List[Dict[str, Union[int, str, float]]]]]]


# 功能：全局变量的初始化
# 初始化一个 ArgumentParser 对象，用于定义和解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, required=True, help="main model weights")
# 功能：调用 parse_args() 方法，解析命令行输入的参数，并将结果存储在 args 变量中。args 是一个命名空间对象，包含所有定义的参数及其值
args = parser.parse_args()


# 功能：使用 AutoTokenizer 加载分词器
# AutoTokenizer.from_pretrained 是 transformers 库中的一个方法，能够根据指定路径自动加载适配的分词器
# model_path 提供了分词器的路径，通常与模型路径一致，以确保分词器和模型兼容
# 作用：为模型准备分词器，能够将输入文本处理成模型所需的张量格式，方便后续推理使用
tokenizer = AutoTokenizer.from_pretrained(args.model)
# 功能：使用 AutoModelForCausalLM 加载预训练模型
# AutoModelForCausalLM.from_pretrained 根据路径加载预训练模型，适用于生成任务（如因果语言建模，Causal Language Modeling）
# torch_dtype=torch.bfloat16 设置模型权重的精度为 bfloat16，这是一种节省内存的浮点数格式，通常在 GPU 上用于降低显存占用而不损失太多精度
# 作用：将模型加载到内存中，准备好用于后续加载微调权重，且通过 bfloat16 的数据类型节省内存
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")


# 用于构建提示字符串，按特定格式拼接输入输出
def build_prompt(context):
    # 检查上下文是否为字符串类型，如果是，则将其解析为 JSON 对象
    if isinstance(context,str):
        context = json.loads(context)
    # 初始化一个空字符串，用于存储构建的提示文本
    prompt = ''
    # 遍历上下文中的每个对话轮次
    for turn in context:
        # 检查角色是否为用户或助手
        if turn["role"] in ["user","assistant"]:
            # 将当前对话的角色和内容以指定格式添加到提示字符串中
            prompt += f'<|im_start|>{turn["role"]}\n{turn["content"]}<|im_end|>\n'
        # 处理角色不为用户或助手的情况
        else:
            # 检查角色是否为 search
            if turn["role"] == "search":
                # 提取搜索参数对象
                obj = turn["arguments"]
                # 过滤掉值为 None 的参数，创建新的字典
                filtered_obj = {k: v for k, v in obj.items() if v is not None}
                # 添加搜索的开始标记
                prompt += '<|im_start|>search\n'
                # 将过滤后的搜索参数转换为格式化的 JSON 字符串，并添加到提示中
                prompt += json.dumps(filtered_obj,indent=4,ensure_ascii=False)
            # 处理角色为返回（return）的情况
            else:
                # 提取记录数据
                obj = turn["records"]
                # 添加返回数据的结束标记，完成这一轮对话的提示构建
                prompt += '<|im_start|>return\n'
                prompt += json.dumps(obj,indent=4,ensure_ascii=False)
            prompt += '<|im_end|>\n'
    # 返回构建好的完整提示字符串
    return prompt


# 功能：定义一个名为 get_completion 的函数，该函数接受一个提示文本并生成响应
# 参数 prompt 是输入的提示文本，通常是要模型生成响应的上下文
async def get_completion(context):
    prompt = build_prompt(context)
    # 功能：将输入的提示文本转换为模型可接受的格式，并将其转移到GPU
    # tokenizer([prompt], return_tensors="pt")：使用 tokenizer 对输入的 prompt 进行编码，返回PyTorch张量格式（return_tensors="pt"），这样可以直接输入到模型中
    # to("cuda")：将张量移动到GPU上，以加速计算
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    # 功能：在不计算梯度的情况下执行后续代码
    # 使用torch.no_grad()上下文管理器，这意味着在这个上下文中不会跟踪梯度，从而节省内存并提高推理速度，适用于模型评估或推理时
    with torch.no_grad():
        # 功能：生成模型的输出
        # 调用模型的 generate 方法生成输出，**inputs 是之前准备的输入张量，max_new_tokens=1024 限制生成的最大 token 数量为1024。模型将根据提供的提示生成相应的输出
        outputs = model.generate(**inputs, max_new_tokens=1024)
        # 功能：解码生成的输出，得到人类可读的响应
        # outputs[:,inputs['input_ids'].shape[1]:][0]：从生成的输出中提取模型的生成部分，inputs['input_ids'].shape[1] 获取输入的长度，[:, ...] 用于切片，获取生成的文本部分
        # tokenizer.decode(..., skip_special_tokens=True)：使用分词器将生成的 token 转换为字符串，skip_special_tokens=True 表示在解码时跳过特殊的 token（如结束符等）
        response = tokenizer.decode(outputs[:,inputs['input_ids'].shape[1]:][0], skip_special_tokens=True)
    # 功能：返回生成的响应
    # return response：将生成的文本响应返回给调用者，供后续使用或展示
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print("服务初始化完成")
    # yield 关键字将控制权交还给FastAPI框架，使应用开始运行
    # 分隔了启动和关闭的逻辑。在yield 之前的代码在应用启动时运行，yield 之后的代码在应用关闭时运行
    yield
    # 关闭时执行
    print("正在关闭...")


# 实例化一个FastAPI实例
# lifespan 参数用于在应用程序生命周期的开始和结束时执行一些初始化或清理工作
app = FastAPI(lifespan=lifespan)

# 启用CORS，允许任何来源访问以 /api/ 开头的接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# POST接口 /api/chat，开启一次模型推理
@app.post("/api/chat")
async def run_chat(request: ChatRequest):
    print(f'接收到的数据:{request.context}')
    try:
        # 调用模型推理函数
        response = await get_completion(request.context)
        print(f'推理完成，已返回数据:{response}')
        return {"response": response}

    except Exception as e:
        print(f"模型推理时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == '__main__':
    # 服务访问的端口
    PORT = 8012
    print(f"在端口 {PORT} 上启动服务器")
    # uvicorn是一个用于运行ASGI应用的轻量级、超快速的ASGI服务器实现
    # 用于部署基于FastAPI框架的异步PythonWeb应用程序
    uvicorn.run(app, host="0.0.0.0", port=PORT)
