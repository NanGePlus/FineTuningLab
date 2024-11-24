import json
from torch.utils.data import Dataset

# 定义一个名为 InputOutputDataset 的新类，它继承自 Dataset 类
# 这使得数据集能够与 PyTorch 的数据加载器兼容，支持批处理和其他数据加载功能
class InputOutputDataset(Dataset):
    # 定义类的初始化方法，接收三个参数：data（数据集）、tokenizer（分词器）、args（其他参数配置）
    def __init__(self, data, tokenizer, args):
        # 调用父类 Dataset 的初始化方法，以确保数据集正确初始化。这是继承类的标准做法
        super(InputOutputDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_column = args.prompt_column
        self.response_column = args.response_column
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length

    # 用于返回数据集的长度，满足 Python 的长度协议
    def __len__(self):
        return len(self.data)

    # 通过索引访问数据集中的样本
    def __getitem__(self, i):
        item = self.data[i]
        # add_special_tokens 不在开头加 special_tokens
        # 调用分词器，对构建的提示文本进行编码
        context = self.tokenizer(
            build_prompt(item[self.prompt_column]), 
            max_length=self.max_source_length, 
            add_special_tokens=False)
        response = self.tokenizer(
            build_response(item[self.response_column]), 
            max_length=self.max_target_length, 
            add_special_tokens=False)
        # 将上下文和响应的 input_ids 连接起来，形成一个完整的输入序列
        input_ids = context["input_ids"] + response["input_ids"]
        # 将上下文和响应的注意力掩码连接起来，确保模型在计算注意力时能够正确地关注输入序列的相关部分
        attention_mask = context["attention_mask"] + response["attention_mask"]
        # 创建标签数组，标记上下文部分为 -100（表示在计算损失时忽略），而响应部分使用真实的 input_ids
        labels = [-100] * len(context["input_ids"]) + response["input_ids"]
        # 确保输入 ID 和标签的长度一致，如果不一致，将抛出断言错误，提供长度信息以便调试
        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"
        # 返回一个字典，其中包含编码后的输入 ID、注意力掩码和标签，供模型训练使用
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


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


# 用于构建响应字符串
def build_response(response):
    # 检查响应是否为字符串类型，若是则解析为 JSON 对象
    if isinstance(response,str):
        response = json.loads(response)
    # 判断角色是否为助手
    if response["role"] == "assistant":
        # 构建助手的响应字符串，格式为
        return '<|im_start|>assistant\n' + response["content"] + '<|im_end|>'
    # 处理角色不是助手的情况
    else:
        # 提取响应中的参数对象
        obj = response["arguments"]
        # 过滤掉值为 None 的参数，创建新的字典
        filtered_obj = {k: v for k, v in obj.items() if v is not None}
        # 构建搜索的响应字符串，格式为
        return '<|im_start|>search\n' + json.dumps(filtered_obj,indent=4,ensure_ascii=False) + '<|im_end|>'

# 接受一个字符串并尝试从中解析出 JSON 对象
def parse_json(string):
    # 初始化搜索起始位置为 0
    search_pos = 0
    # 寻找字符串中第一个 { 的位置，用于标识 JSON 对象的开始
    start = string.find('{', search_pos)
    # 如果未找到 {，返回 None
    if start == -1:
        return None
    # 从找到的 { 位置开始，寻找最后一个 } 的位置，标识 JSON 对象的结束
    end = string.rfind('}', start)
    # 如果未找到 }，返回 None
    if end == -1:
        return None
    # 提取出 JSON 字符串，从找到的 { 到 } 的部分
    json_string = string[start:end + 1]
    try:
        # 将字符串解析为 Python 对象
        obj = json.loads(json_string)
        return obj
    except json.JSONDecodeError:
        return None
