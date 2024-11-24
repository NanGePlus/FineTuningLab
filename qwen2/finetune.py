import json
# 导入 torch 模块，这是 PyTorch 深度学习框架的核心库
# 提供了构建和训练神经网络所需的基本功能，如张量操作、模型定义、优化器等
import torch
# AutoTokenizer：用于自动加载与特定模型对应的分词器。分词器将输入文本转换为模型可接受的格式
# AutoModelForCausalLM：用于自动加载用于因果语言建模的预训练模型。因果语言模型用于生成任务，比如文本生成
# BitsAndBytesConfig：用于设置模型的低精度计算配置（如 8-bit 或 4-bit 量化），以减少内存使用和加速推理过程
# DataCollatorForSeq2Seq：用于处理序列到序列模型的数据整理，它可以处理批次数据并填充到相同的长度
# HfArgumentParser：用于从命令行解析参数的工具，方便用户在运行时提供各种参数设置
# TrainingArguments：用于定义训练过程的参数，如学习率、批次大小、评估策略等
# Trainer：用于封装模型训练过程的高阶API，简化了训练和评估过程的实现
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq, 
    HfArgumentParser,
    TrainingArguments, 
    Trainer
)
# 从 peft 库中导入 LoRA 相关的组件
# LoraConfig：用于配置 LoRA 微调过程的参数，包括低秩矩阵的秩、缩放系数等
# TaskType：定义任务类型的枚举，用于指定微调任务的类别（例如文本生成、文本分类等）
# get_peft_model：用于获取带有 LoRA 配置的模型的方法，用于将 LoRA 参数应用到基础模型上
from peft import LoraConfig, TaskType, get_peft_model
# 从自定义模块 arguments 中导入自定义参数类
# ModelArguments：用于存储与模型相关的参数
# DataTrainingArguments：用于存储与数据训练相关的参数
# PeftArguments：用于存储与 PEFT（参数高效微调）相关的参数
from arguments import ModelArguments, DataTrainingArguments, PeftArguments
# # 从自定义模块 data_preprocess 中导入自定义方法，是一个自定义的数据集类，用于将原始数据转换为模型的输入格式
from data_preprocess import InputOutputDataset



def main():
    # 创建一个 HfArgumentParser 实例，用于解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, TrainingArguments))
    # 将命令行参数解析为相应的数据类实例
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()
    # 加载一个预训练模型
    # 指定模型的计算精度为 bfloat16。这是一种优化内存使用的数值格式，特别适合在支持 bfloat16 的硬件（如某些 GPU）上运行，可以降低内存需求和提高计算效率
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)
    # 加载与模型对应的分词器（tokenizer）
    # 分词器的作用是将文本转化为模型能够理解的 token 序列（即数值化表示），这是将自然语言输入传递到模型中的必要步骤
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # 设置LoRA的配置，用于控制 LoRA 在特定模型和任务中的应用
    lora_config = LoraConfig(
        # 设置该 LoRA 配置用于训练模式，而非推理模式
        inference_mode=False,
        # 设置任务类型为 TaskType.CAUSAL_LM，表示这是一个因果语言建模任务
        task_type=TaskType.CAUSAL_LM,
        # 设置模型中应用 LoRA 的模块为 q_proj、k_proj 和 v_proj
        # 在常见的 Transformer 结构中，q_proj（查询投影）、k_proj（键投影）和 v_proj（值投影）是多头自注意力机制的重要组成部分
        # 使用 LoRA 只在这些投影矩阵中引入低秩更新，能够减少模型参数数量的增长，同时对注意力机制进行微调。这种方式能高效地调整模型参数，而不会显著增加训练时的计算负担
        target_modules=["q_proj", "k_proj", "v_proj"],
        # 设置LoRA 中低秩矩阵的秩（rank）
        # 低秩矩阵的秩是 LoRA 的核心超参数之一，它定义了新增参数的数量。秩越高，LoRA 可以学习到的特征表达就越多，但也会增加额外参数量和计算量
        r=peft_args.lora_rank,
        # 设置 LoRA 的缩放系数，控制低秩矩阵对原始模型参数更新的影响
        # 它起到“调节器”的作用，将学习到的低秩更新进行放缩，使 LoRA 在适应新任务时不会过度影响原始模型的表现
        lora_alpha=peft_args.lora_alpha,
        # 设置 LoRA 训练过程中使用的 dropout 概率
        # 用于防止模型过拟合。在训练过程中，会随机丢弃部分神经元（按指定的概率），从而减弱模型对特定路径的依赖，提高模型在未见数据上的泛化能力
        lora_dropout=peft_args.lora_dropout
    )

    # 将原始模型 model 转换为 LoRA 模型，并将其移动到 GPU 上
    model = get_peft_model(model, lora_config).to("cuda")
    # 打印模型中可训练的参数数量，以便检查
    # 由于 LoRA 的设计是通过低秩矩阵在特定层上进行微调，因此输出结果会显示只有少部分参数是“可训练的”
    # 可以确认 LoRA 只修改了选定的层，而非整个模型。这对于调试 LoRA 配置是否正确非常有用
    model.print_trainable_parameters()

    # 创建数据规整器实例，对数据进行批次处理
    # tokenizer=tokenizer，指定使用的分词器，用于处理输入文本并将其转化为模型理解的 token 格式。分词器会确保在每个批次中对输入文本进行相同的编码，以便与模型兼容
    # padding=True，启用自动填充（padding），使得每个批次的输入序列长度一致。这意味着对于较短的序列，DataCollatorForSeq2Seq 会填充到当前批次中的最大序列长度，确保数据在批处理中可以对齐
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True
    )

    # 控制是否加载和准备训练数据。这样可以根据需求灵活地决定是否进行训练，尤其适用于在同一脚本中设置不同模式（如仅训练、仅评估或同时进行）
    if training_args.do_train:
        # 该步骤将训练数据文件中的每一行解析为一个字典，并将所有行合并到一个列表中。通常，训练数据文件中的每一行包含一个样本的 JSON 格式数据，这样可以逐行处理大文件，节省内存
        # json.loads(line)：逐行读取文件，每行内容使用 json.loads() 将 JSON 字符串解析为 Python 字典
        # [json.loads(line) for line in f]：将所有行解析为一个包含多个字典的列表 train_data
        with open(data_args.train_file, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]
        # 将训练数据 train_data 转换为模型可接受的数据集对象 train_dataset
        # InputOutputDataset 是一个自定义的数据集类，用于将原始数据转换为模型的输入格式
        # train_data 是从训练文件读取并解析的数据
        # tokenizer 用于对数据进行分词和编码
        # data_args 传入的其他数据参数，用于控制数据处理的行为
        train_dataset = InputOutputDataset(train_data, tokenizer, data_args)

    # 控制是否加载和准备验证数据，逻辑与训练数据相同
    if training_args.do_eval:
        with open(data_args.validation_file, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]
        eval_dataset = InputOutputDataset(eval_data, tokenizer, data_args)

    # 创建了一个 Trainer 对象，并传入了模型、分词器、数据规整器、训练参数、训练数据集和验证数据集
    # 这个对象负责后续的训练、评估等任务
    trainer = Trainer(
        # 功能：指定 Trainer 要训练或评估的模型
        # 解释：这里传入的 model 是已经加载的模型，并且应用了 LoRA（低秩适配）配置，以便进行高效微调
        # 作用：让 Trainer 直接使用这个模型进行训练或评估
        model=model,
        # 功能：指定分词器 tokenizer，用于处理输入数据
        # 解释：tokenizer 会被 Trainer 用来对输入数据进行编码，以确保输入符合模型的要求
        # 作用：在训练和评估时，Trainer 会自动使用 tokenizer 将文本数据转换为模型理解的 token 格式，简化了输入处理步骤
        tokenizer=tokenizer,
        # 功能：指定数据规整器 data_collator，用于批次化输入数据
        # 解释：在批次处理时，对数据进行填充（padding），确保每个批次中的序列长度一致，使得模型输入结构整齐
        # 作用：可以有效地批次化数据，提高训练效率，同时确保批次中的每个样本大小相同，适配模型的输入需求
        data_collator=data_collator,
        # 功能：指定训练参数 training_args
        # 解释：training_args 是一个 TrainingArguments 实例，包含训练中的超参数配置，如学习率、批次大小、训练轮数等
        # 作用：training_args 控制 Trainer 的训练行为，设置超参数、日志输出、是否保存模型、模型存储路径等，为训练过程提供全局配置
        args=training_args,
        # 功能：指定训练数据集 train_dataset
        # train_dataset 是已经经过分词器处理的训练数据集
        # 只有当 do_train=True 时，才会传入 train_dataset，否则为 None。这意味着如果不执行训练，就不需要传入训练数据集
        # 作用：根据 training_args.do_train 参数的值灵活地加载训练数据集。这允许在不需要训练时跳过训练数据集的加载，减少内存开销
        train_dataset=train_dataset if training_args.do_train else None,
        # 功能：指定验证数据集 eval_dataset
        # eval_dataset 是已经经过分词器处理的验证数据集
        # 只有当 do_eval=True 时，才会传入 eval_dataset，否则为 None。这意味着如果不执行评估，就不需要传入验证数据集
        # 作用：根据 training_args.do_eval 参数的值灵活地加载验证数据集。这允许在不需要评估时跳过验证数据集的加载，避免不必要的资源消耗
        eval_dataset=eval_dataset if training_args.do_eval else None,
    )

    # 首先判断是否需要训练，然后在需要时执行如下操作：
    # (1)梯度检查点 以降低内存使用
    # (2)输入梯度计算 以确保模型可以在训练中被更新
    # (3)最后调用 trainer.train() 来启动训练流程，使得模型根据训练数据集进行权重调整
    if training_args.do_train:
        # 功能：启用梯度检查点（gradient checkpointing）技术
        # gradient_checkpointing_enable() 是 transformers 中的一项内存优化技术，尤其适合大模型的训练
        # 它通过在前向传播时保存部分中间激活值而不是全部激活值，减少内存占用；在反向传播时再动态计算需要的激活值
        # 作用：启用梯度检查点可以大幅降低显存（GPU memory）需求，使得更大批次或更大模型可以在有限的显存中训练，适合在资源受限情况下进行大模型训练
        model.gradient_checkpointing_enable()
        # 功能：启用输入梯度（input gradient）计算
        # enable_input_require_grads() 设置模型的输入张量（tensor）需要计算梯度，以便在反向传播中更新权重
        # 该方法通常用于精调（fine-tuning）任务中，确保模型的所有部分在训练中能够被更新
        # 作用：此设置对 LoRA 等微调任务特别重要，确保输入的梯度可以被计算，从而允许训练过程中的反向传播更新模型参数
        model.enable_input_require_grads()
        # 功能：启动模型训练
        # trainer.train() 是 Trainer 对象的核心方法，负责执行整个训练过程
        # Trainer 内部会根据配置好的参数（如学习率、批次大小、迭代次数等）自动运行训练循环
        # 该方法还会在训练过程中自动进行日志记录、模型保存、评估等任务（如果在 training_args 中配置了相应选项）
        # 作用：实际执行模型的训练过程。在 train() 方法运行期间，Trainer 会读取 train_dataset，将数据输入模型，并更新模型权重，从而进行模型的优化
        trainer.train()

    # 首先通过 training_args.do_eval 判断是否需要进行模型评估
    # 若需要，调用 trainer.evaluate() 来执行模型的评估过程
    # 这一操作会自动计算模型在验证集上的表现指标，帮助确定模型在验证集上的效果，从而判断模型是否具有良好的泛化能力
    if training_args.do_eval:
        # 功能：执行模型的评估过程
        # trainer.evaluate() 是 Trainer 对象中的一个方法，用于评估模型的性能
        # evaluate() 方法会自动加载之前传入的验证数据集（eval_dataset），并对模型进行评估
        # Trainer 会计算常见的评估指标，如损失（loss）、精度（accuracy）、困惑度（perplexity）等，具体取决于任务和配置
        # 作用：evaluate() 方法可以在验证数据集上评估模型的表现，帮助了解模型在训练集之外的数据上的效果。这在评估模型的泛化能力和调整模型超参数时非常有用
        trainer.evaluate()

if __name__ == "__main__":
    main()
