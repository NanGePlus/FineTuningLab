import json
import torch
import argparse
# 功能：导入 tqdm 模块，用于显示进度条
# tqdm 可以将代码循环包装在进度条中，实时显示处理进度
# 作用：在处理大量数据（如模型推理或数据加载）时提供进度反馈，尤其在处理大型数据集或长时间任务时非常有用
from tqdm import tqdm
# 功能：从 peft 库中导入 PeftModel，用于加载经过参数高效微调（PEFT）的模型
# PeftModel 是一个用于高效模型微调的类，特别适合大模型，允许通过调整少量参数来进行微调
# 作用：在微调完成后，使用 PeftModel 可以加载模型并进行推理或评估，从而节省内存和计算成本
from peft import PeftModel
# 功能：从 transformers 库中导入 AutoTokenizer 和 AutoModelForCausalLM
# AutoTokenizer 是一个自动化选择合适分词器的类，能够根据模型名称自动加载对应的分词器
# AutoModelForCausalLM 是生成模型的一个通用类，适用于因果语言建模（Causal Language Modeling）任务
# 作用：在脚本中用于加载模型和分词器，便于后续进行文本生成、推理或评估
from transformers import AutoTokenizer, AutoModelForCausalLM
# 功能：从 nltk 库中导入 BLEU 评估分数的计算方法
# sentence_bleu 用于计算一个生成句子与参考句子之间的 BLEU 分数
# SmoothingFunction 提供了平滑技术来处理较短句子的 BLEU 分数计算，使得分数更加稳定
# 作用：计算生成文本与参考文本的相似度，衡量模型的生成质量，特别在机器翻译或生成任务中用作评估指标
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# 功能：从 data_preprocess 模块中导入 build_prompt 和 parse_json 函数
# build_prompt 通常是用于构建输入提示的函数，将原始数据转换成模型可接受的格式
# parse_json 用于解析 JSON 格式的数据，从 JSON 文件中提取出模型需要的字段
# 作用：提供数据预处理功能，为后续模型的输入准备数据，使得输入数据格式符合模型要求
from data_preprocess import build_prompt, parse_json



# 加载了一个分词器和一个包含微调权重的预训练模型，并将模型转移到 GPU 上，以加速推理
# 模型以 bfloat16 精度加载，节省显存，同时切换到评估模式确保推理效果稳定
# 最终返回分词器和模型对象，用于后续的文本处理或生成任务
def load_model(model_path, checkpoint_path):
    # 功能：使用 AutoTokenizer 加载分词器
    # AutoTokenizer.from_pretrained 是 transformers 库中的一个方法，能够根据指定路径自动加载适配的分词器
    # model_path 提供了分词器的路径，通常与模型路径一致，以确保分词器和模型兼容
    # 作用：为模型准备分词器，能够将输入文本处理成模型所需的张量格式，方便后续推理使用
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 功能：使用 AutoModelForCausalLM 加载预训练模型
    # AutoModelForCausalLM.from_pretrained 根据路径加载预训练模型，适用于生成任务（如因果语言建模，Causal Language Modeling）
    # torch_dtype=torch.bfloat16 设置模型权重的精度为 bfloat16，这是一种节省内存的浮点数格式，通常在 GPU 上用于降低显存占用而不损失太多精度
    # 作用：将模型加载到内存中，准备好用于后续加载微调权重，且通过 bfloat16 的数据类型节省内存
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    # 功能：加载微调后的权重，并将模型转移到 GPU 上
    # PeftModel.from_pretrained 用于加载经过 PEFT 微调的模型权重，将微调后的权重直接加载到预训练模型上(这里并不是合并，只是配合使用)
    # model：基础的预训练模型，作为 PeftModel 的基础结构
    # model_id=checkpoint_path：指定微调权重的路径，PeftModel 会加载该路径下的权重
    # .to("cuda")：将模型移到 GPU 上运行，以利用 GPU 加速推理
    # .eval()：将模型切换到评估模式，冻结模型参数，关闭 dropout 等训练专用的特性
    # 作用：加载并应用微调权重，使得模型可以在 GPU 上高效推理，同时关闭训练特性确保稳定性
    model = PeftModel.from_pretrained(model, model_id=checkpoint_path).to("cuda").eval()
    # 功能：返回分词器和加载完成的模型对象
    # tokenizer：已经加载完成的分词器，用于将输入文本处理成模型输入格式
    # model：包含预训练权重和微调权重的模型对象，已准备好进行推理
    # 作用：将分词器和模型返回给调用方，方便在其他代码中调用模型和分词器进行文本处理和生成任务
    return tokenizer, model


# 功能：定义了一个名为 Evaluator 的类
# 解释：这个类的目的是提供一系列评估功能，以帮助测量模型在特定任务（如意图识别或槽位填充）上的表现
class Evaluator:
    # 功能：初始化 Evaluator 类的实例
    # tokenizer：传入的分词器对象，通常用于处理输入文本，确保输入格式与模型要求一致
    # model：传入的模型对象，通常是用于预测或推理的已加载模型
    # data_path：传入数据路径加载测试集，用于后续的评估
    # 作用：初始化实例的属性，以便在其他方法中使用
    def __init__(self,tokenizer,model,data_path):
        self.tokenizer = tokenizer
        self.model = model
        self.data_path = data_path

    # 功能：定义一个名为 slot_accuracy 的方法，用于计算预测结果与真实标签之间的准确性
    # 槽位填充的目标是从用户的输入文本中提取出特定的槽位值，例如在"预订一个不低于1000的餐桌"中提取“价格=不低于1000”
    # pred：预测的槽位结果，通常是模型输出的字典格式
    # label：真实的槽位标签，通常也是一个字典格式
    # 作用：评估模型在槽位填充任务中的准确性
    def slot_accuracy(self, pred, label):
        # 功能：初始化一个计数器，用于统计正确预测的数量
        # 作用：在后续的循环中，累积正确的槽位预测数
        correct = 0
        # 功能：检查 pred 是否为 None，以防止空预测导致的错误
        # 解释：如果 pred 不为 None，则继续评估过程
        if pred is not None:
            # 功能：遍历预测结果的每一个键值对
            # k：表示槽位的名称（键）
            # v：表示槽位的预测值（值）
            for k, v in pred.items():
                # 功能：检查预测值是否为 None
                # 作用：如果预测值为空，跳过当前的槽位处理，继续下一个槽位
                if v is None:
                    continue
                # 功能：检查真实标签是否存在，并且当前槽位名称 k 是否在真实标签中
                # 解释：确保只有在标签存在的情况下才进行比较
                if label and k in label:
                    # 功能：检查预测值 v 是否为列表
                    # 作用：用于判断槽位预测的类型，以决定后续的比较方式
                    if not isinstance(v,list):
                        # 功能：如果预测值 v 不是列表，直接与真实标签进行比较
                        # 解释：如果预测值等于真实标签，则计数器 correct 加 1
                        correct += int(v==label[k])
                    # 功能：处理当预测值 v 是列表的情况
                    # 作用：遍历预测列表中的每个预测值
                    else:
                        for t in v:
                            # 功能：检查每个预测值 t 是否在真实标签 label[k] 中
                            # 解释：如果存在，计数器 correct 加 1，表示该预测是正确的
                            correct += int(t in label[k])
        # 功能：计算预测槽位的总数
        # 对于每个槽位，如果预测值是列表，则统计列表长度；如果不是，则视为一个槽位（计为 1）
        # 如果 pred 为 None，则总数为 0
        pred_slots = sum(len(v) if isinstance(v, list) else 1 for v in pred.values()) if pred else 0
        # 功能：计算真实槽位的总数
        # 与上面的逻辑相似，遍历真实标签，计算每个槽位的数量
        # 如果 label 为 None，则总数为 0
        true_slots = sum(len(v) if isinstance(v, list) else 1 for v in label.values()) if label else 0
        # 功能：返回正确预测的数量、预测槽位的总数和真实槽位的总数
        # 作用：提供用于评估模型的槽位填充性能的数据
        return correct, pred_slots, true_slots


    # 功能：定义一个名为 bleu4 的方法
    # BLEU-4 分数，这是自然语言处理中的一种常用评估指标，用于衡量生成文本与参考文本之间的相似度
    # pred：模型生成的预测文本（字符串）
    # label：真实的参考文本（字符串）
    # 作用：计算预测文本和参考文本之间的 BLEU-4 分数
    def bleu4(self, pred, label):
        # 功能：去除预测文本和参考文本两端的空白字符
        # 作用：确保在计算 BLEU 分数时，文本的开头和结尾没有多余的空格，避免影响计算结果
        pred = pred.strip()
        label = label.strip()
        # 功能：将预测文本 pred 转换为字符列表
        # 解释：这里 hypothesis 变量存储的是预测文本的字符序列，以便在后续 BLEU 评分计算中使用
        hypothesis = list(pred)
        # 功能：将参考文本 label 转换为字符列表
        # 解释：类似于上面，将参考文本也转换为字符序列，存储在 reference 变量中
        reference = list(label)
        # 功能：检查 hypothesis 和 reference 的长度
        # 如果预测文本或参考文本为空（长度为0），则无法计算 BLEU 分数
        # 在这种情况下，直接返回 0，表示无法评估
        if len(hypothesis) == 0 or len(reference) == 0:
            return 0
        # 功能：计算 BLEU 分数
        # sentence_bleu 函数来自 nltk.translate.bleu_score 模块，用于计算 BLEU 分数
        # [reference] 作为一个列表传递，因为 BLEU 准确度计算需要多个参考句子，这里使用单个参考文本作为列表
        # hypothesis 是预测文本
        # smoothing_function=SmoothingFunction().method3：使用平滑函数以防止在计算时出现零概率的问题。这种方法对于短文本或没有重叠词汇的情况尤其重要
        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
        # 功能：返回计算得到的 BLEU 分数
        # 作用：将 BLEU 分数作为该方法的输出，供调用者使用
        return bleu_score


    # 该方法用于计算模型在给定测试数据集上的评估指标，包括插槽的精确度、召回率、F1 分数和 BLEU-4 分数
    # 功能：定义一个名为 compute_metrics 的方法
    # 作用：计算模型的性能指标
    def compute_metrics(self):
        # 功能：初始化各类评分和计数器
        # score_dict：字典用于存储插槽精确度 (slot_P)、召回率 (slot_R) 和 F1 分数 (slot_F1)
        # 精确率(Precision)衡量的是模型预测出的槽位中，实际为正确槽位的比例。它关注的是模型预测的“正确性”，尤其在意的是预测的正样本是否为真正的正样本。例如，如果模型在槽位填充任务中识别出了几个槽位标签，那么精确率会告诉我们这些标签中有多少是正确的
        # 召回率(Recall)衡量的是实际为正样本的槽位中，模型正确识别出来的比例。召回率关注的是模型“覆盖的全面性”，即是否漏掉了某些需要识别的槽位
        # F1(F1 Score)分数是精确率和召回率的调和平均数，用于在精确率和召回率之间取得平衡，尤其当二者存在显著差异时，F1分数可以帮助全面衡量模型的槽位填充性能
        score_dict = { "slot_P": 0.0, "slot_R": 0.0, "slot_F1": 0.0 }
        # bleu_scores：列表用于存储 BLEU-4 分数
        bleu_scores = []
        # true_slot_count、pred_slot_count 和 correct_slot_count：用于跟踪真实插槽数量、预测插槽数量和正确插槽数量
        true_slot_count = 0
        pred_slot_count = 0
        correct_slot_count = 0

        # 功能：加载测试数据集
        # 使用 UTF-8 编码打开文件 self.data_path
        # 将每一行读取为 JSON 格式，并存储在 test_dataset 列表中
        with open(self.data_path, "r", encoding="utf-8") as f:
            test_dataset = [json.loads(line) for line in f]

        # 功能：遍历测试数据集
        # 使用 tqdm 包装循环以显示进度条，提供当前处理进度
        for item in tqdm(test_dataset):
            # 功能：生成模型输入的提示
            # 调用 build_prompt 函数，使用当前项的上下文信息构建提示
            template = build_prompt(item["context"])
            # 功能：将生成的提示文本转换为张量格式
            # 使用 self.tokenizer 将模板转换为 PyTorch 张量，并将其转移到 GPU（CUDA）上
            inputs = self.tokenizer([template], return_tensors="pt").to("cuda")
            # 功能：生成模型的输出
            # 使用 torch.no_grad() 关闭梯度计算，以节省内存和提高推理速度
            # 调用 self.model.generate 方法生成文本，max_new_tokens=1024 指定最大生成的令牌数量
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=1024)
                # 功能：解码模型输出
                # 使用 self.tokenizer.decode 将生成的张量转换为文本
                # outputs[:,inputs['input_ids'].shape[1]:]：去掉输入部分，保留生成的新令牌
                # skip_special_tokens=True：跳过特殊字符（如填充标记）以获得干净的输出
                response = self.tokenizer.decode(outputs[:,inputs['input_ids'].shape[1]:][0], skip_special_tokens=True)
            # 功能：加载参考响应
            # 将当前项的响应字段转换为 JSON 对象，存储在 label 中，以便后续比较
            label = json.loads(item["response"])
            # 功能：检查响应的角色
            # 如果角色为 "search"，则进行插槽准确度的计算
            if label["role"] == "search":
                # 功能：解析模型生成的响应
                # 尝试调用 parse_json 函数将响应解析为预测插槽
                # 如果解析失败，捕获异常并将 preds 设置为一个空字典
                try:
                    preds = parse_json(response)
                except:
                    preds = {}
                # 功能：获取真实插槽
                # 从 label 中提取 "arguments" 字段，作为真实插槽
                truth = label["arguments"]
                # 功能：计算插槽的正确性
                # 调用 self.slot_accuracy 方法，传入预测插槽和真实插槽，并返回正确的插槽数量、预测的插槽数量和真实的插槽数量
                correct, pred_slots, true_slots = self.slot_accuracy(preds, truth)
                # 功能：更新插槽计数器
                # 将真实插槽数量、预测插槽数量和正确插槽数量累加到各自的总计中
                true_slot_count += true_slots
                pred_slot_count += pred_slots
                correct_slot_count += correct
            # 功能：处理非搜索角色的响应
            # 如果角色不是 "search"，则将响应中的 "assistant" 替换为空字符串，以清理输出
            # 计算 BLEU-4 分数并将其添加到 bleu_scores 列表中
            else:
                response = response.replace("assistant","")
                bleu_scores.append(self.bleu4(response, label['content']))
        # 功能：计算插槽的精确度
        # 精确度 = 正确插槽数量 / 预测插槽数量
        # 如果预测插槽数量大于0，则计算精确度，否则返回0
        score_dict["slot_P"] = float(correct_slot_count/pred_slot_count) if pred_slot_count > 0 else 0
        # 功能：计算插槽的召回率
        # 召回率 = 正确插槽数量 / 真实插槽数量
        # 如果真实插槽数量大于0，则计算召回率，否则返回0
        score_dict["slot_R"] = float(correct_slot_count/true_slot_count) if true_slot_count > 0 else 0
        # 功能：计算插槽的 F1 分数
        # F1 分数 = 2 * (精确度 * 召回率) / (精确度 + 召回率)
        # 仅在精确度和召回率之和大于0时进行计算，否则返回0
        score_dict["slot_F1"] = 2*score_dict["slot_P"]*score_dict["slot_R"]/(score_dict["slot_P"]+score_dict["slot_R"]) if (score_dict["slot_P"]+score_dict["slot_R"]) > 0 else 0
        # 功能：计算 BLEU-4 的平均分数
        # 使用 sum(bleu_scores) 计算所有 BLEU-4 分数的总和，并除以 BLEU-4 分数的数量，得出平均值
        score_dict["bleu-4"] = sum(bleu_scores) / len(bleu_scores)
        # 功能：将分数转换为百分制
        # 遍历 score_dict 中的每个指标，将其值乘以100并保留四位小数
        for k, v in score_dict.items():
            score_dict[k] = round(v * 100, 4)
        # 功能：输出最终的评分字典
        # 解释：将所有计算出的性能指标打印到控制台，以便于查看和分析
        print(f"score dict: {score_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, required=True, help="main model weights")
    parser.add_argument("--ckpt", type=str, default=None, required=True, help="The checkpoint path")
    parser.add_argument("--data", type=str, default=None, required=True, help="The dataset file path")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.ckpt)
    evaluator = Evaluator(tokenizer, model, args.data)
    evaluator.compute_metrics()
