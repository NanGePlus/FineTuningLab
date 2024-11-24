import torch
import argparse
# 用于处理LoRA（Low-Rank Adaptation）模型
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model
# 功能：从自定义模块中导入 load_model 函数
# from evaluate import load_model：从名为 evaluate 的模块中导入 load_model 函数，用于加载和初始化模型
from evaluate import load_model




# 功能：全局变量的初始化
# 初始化一个 ArgumentParser 对象，用于定义和解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, required=True, help="main model weights")
parser.add_argument("--ckpt", type=str, default=None, required=True, help="The checkpoint path")
parser.add_argument("--output_dir", type=str, default=None, required=True, help="output path")
# 功能：调用 parse_args() 方法，解析命令行输入的参数，并将结果存储在 args 变量中。args 是一个命名空间对象，包含所有定义的参数及其值
args = parser.parse_args()


# 功能：加载模型和分词器
# load_model(args.model, args.ckpt)：调用之前定义的 load_model 函数，传入用户从命令行提供的模型路径 args.model 和检查点路径 args.ckpt，返回一个分词器和模型
# tokenizer, model：将返回的分词器和模型分别赋值给 tokenizer 和 model 变量，以便后续使用
tokenizer, model = load_model(args.model, args.ckpt)


# 合并LoRA模型与基础模型，并卸载LoRA模型的参数，返回合并后的模型
# 这是 model 的方法，执行合并和卸载操作
# merge：将 LoRA 微调的权重与原始模型的权重合并。这一步将 LoRA 适配器中的权重应用到模型的原始权重上，以获得一个新的权重集合，该集合可以直接用于推理和使用
# unload：从内存中卸载 LoRA 适配器。这一步是为了释放资源，因为在合并权重后，适配器不再需要保存在内存中
model = model.merge_and_unload()
# 将合并后的模型保存到指定的输出路径
model.save_pretrained(args.output_dir)
# 将模型上传到HuggingFace Hub
# model.push_to_hub("zhuzhu01/Qwen2.5-7B-Instruct-Merge", token = "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")
# 将分词器保存到相同的输出路径
tokenizer.save_pretrained(args.output_dir)
# 将分词器上传到HuggingFace Hub
# base_tokenizer.push_to_hub("zhuzhu01/Qwen2.5-7B-Instruct-Merge", token = "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")





