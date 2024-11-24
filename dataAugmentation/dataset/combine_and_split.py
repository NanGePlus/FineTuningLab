import json
import os
import random
import copy


# 随机数生成器在不同运行时返回相同的随机数序列
random.seed(42)


# 用于处理一个对话（dialog），并将其转换为带有上下文和回复的单轮次格式，然后追加到 data 列表中
# dialog：表示整个对话的列表，每个元素是该对话的一个轮次（turn），包含角色（如用户或助手）和内容等信息
# data：用于存储处理后数据的列表，每个元素包含上下文（context）和回复（response）
def process_dialog(dialog, data):
    # 初始化空列表 buffer，用于临时存储当前对话中的轮次，构建上下文
    buffer = []
    # 遍历 dialog 中的每一轮对话 turn
    for turn in dialog:
        # 检查当前轮次的角色是否为 "assistant" 或 "search"，并确保 buffer 中已有内容
        # 此条件判断意味着，当助手或搜索角色的回复到达时，当前 buffer 的内容会作为上下文，构成上下文-回复对
        if (turn["role"] == "assistant" or turn["role"] == "search") and len(buffer)>0:
            # 当满足条件时，将当前上下文和对应的回复格式化为字典形式，并添加到 data 列表中
            # context：将 buffer 转换为 JSON 格式字符串，表示该回复的上下文
            # response：将当前轮次 turn 转换为 JSON 格式字符串，作为上下文对应的回复
            data.append({
                "context" : json.dumps(buffer,ensure_ascii=False),
                "response" : json.dumps(turn,ensure_ascii=False)
            })
        # 无论当前轮次是否符合条件，都会将其添加到 buffer 中，确保后续轮次能够包含完整的上下文
        buffer.append(turn)
    # 返回 data 列表，其中包含处理好的上下文和回复对
    return data


# 将对话数据集 data 转换为一系列单一轮次对话（turns）的列表
# data：包含多个对话（dialog）数据的列表
# shuffle：布尔值，指示是否在处理后对数据进行随机排序，默认为 False
def data_to_turns(data,shuffle=False):
    # 初始化空列表 ans，用于存储处理后的单轮次对话数据
    ans = []
    # 遍历 data 中的每个对话 dial，并调用 process_dialog 函数来处理每个对话，将结果追加到 ans 列表中
    for dial in data:
        # 根据每个对话的角色，将对话转换成适合后续使用的格式，并将处理后的结果添加到 ans 中
        process_dialog(dial,ans)
    # 检查 shuffle 参数是否为 True。如果是，则对 ans 列表中的元素顺序进行随机打乱
    # 在某些情况下，将数据顺序随机化可以帮助提升模型在训练时的泛化能力
    if shuffle:
        random.shuffle(ans)
    # 返回 ans 列表，其中包含处理后的单轮次对话数据
    return ans


# 用于判断一个对话中是否包含多次搜索请求
# dialog：表示对话内容，是一个包含多个对话轮次的列表
def is_multi_search(dialog):
    # 初始化计数器 count，用于统计该对话中 "search" 类型的轮次数量
    count = 0
    # 遍历 dialog 列表中的每个 turn，即每个对话轮次。每个 turn 是一个字典，包含了该轮对话的相关信息（如角色和内容）
    for turn in dialog:
        # 判断 turn 字典中的 role 字段是否为 "search"
        # 如果 role 的值为 "search"，则表示该轮次为搜索操作，count 增加 1
        if turn["role"] == "search":
            count += 1
    # 返回一个布尔值，表示 count 是否大于 1
    # 如果 count > 1，说明该对话中有多次搜索操作，则返回 True；否则返回 False
    return count > 1


# 主要用于读取指定目录下的 JSON 文件，并将其按多轮和单轮搜索对话分类
# dir_path：数据文件所在的目录路径
# data：传入的空数据列表，用于保存加载的对话数据
# n：限制单轮搜索对话的数量，默认为 None 表示不限制
def process_dir(dir_path,data,n=None):
    files = []
    # 遍历指定目录 dir_path 下的文件，将文件路径加入 files 列表
    # os.listdir(dir_path)：列出目录下的所有文件和子目录名
    # os.path.join(dir_path, filename)：将目录路径和文件名组合成完整路径
    # os.path.isfile(file_path)：确保只添加文件路径，忽略子目录
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            files.append(file_path)
    # 逐个读取 files 中的文件内容，假设每个文件是一个 JSON 格式的对话列表，将其解析为 dialog 对象并添加到 data 列表中
    for file_path in files:
        with open(file_path,'r',encoding="utf-8") as fp:
            # 解析 JSON 文件内容并加载为 Python 对象
            dialog = json.load(fp)
            data.append(dialog)
    # 初始化两个空列表，分别用于保存多轮（multi）和单轮（single）搜索对话
    multi = []
    single = []
    # 遍历 data 中的对话，利用 is_multi_search 函数判断每个对话是否为多轮搜索对话
    # is_multi_search(dial)：返回 True 表示该对话包含多次搜索，添加到 multi 列表，否则添加到 single 列表
    for dial in data:
        if is_multi_search(dial):
            multi.append(dial)
        else:
            single.append(dial)
    # 使用 random.shuffle 函数对 single 列表进行随机打乱，以增加数据多样性
    random.shuffle(single)
    # 如果 n 不为 None，则截取 single 列表的前 n - len(multi) 条数据
    # 目的是确保 single 列表的数量不会超过指定的最大数量 n，但同时保证 multi 列表中的所有多轮对话都被保留
    if n is not None:
        single = single[:n-len(multi)]
    # 将 multi 和 single 列表合并并返回，保证多轮对话在前，单轮对话在后
    return multi+single


# 用于从指定目录 dir_path 中读取文件内容，并将每个文件中的对话数据添加到 data 列表中
# dir_path：包含数据文件的目录路径
# data：用于存储所有对话数据的列表
def process_dir_v2(dir_path, data):
    # 遍历指定目录 dir_path 下的文件，将文件路径加入 files 列表
    # os.listdir(dir_path)：列出目录下的所有文件和子目录名
    # os.path.join(dir_path, filename)：将目录路径和文件名组合成完整路径
    # os.path.isfile(file_path)：确保只添加文件路径，忽略子目录
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            with open(file_path,'r',encoding="utf-8") as fp:
                dialogs = json.load(fp)
                for dial in dialogs:
                    data.append(dial)
                    #process_dialog(dial,data)
    # 返回更新后的 data 列表，该列表包含了 dir_path 目录中所有文件中的对话数据
    return data


# 用于将数据集 data 按比例 ratio 划分为训练集、验证集和测试集
# data：需要划分的数据集，通常为包含对话或数据样本的列表
# ratio：验证集和测试集的比例。例如，若 ratio 为 0.1，则验证集和测试集各占数据集的 10%
def split_data(data,ratio):
    # 打乱数据集 data 中的样本顺序，以确保划分后的数据集分布尽可能随机和均匀
    # 通过随机打乱数据样本顺序，减少原始顺序带来的数据偏差
    random.shuffle(data)
    # 根据比例 ratio 计算验证集的大小 dev_size
    # 将数据集总长度 len(data) 乘以 ratio，然后转换为整数，得到验证集所需的样本数量
    dev_size = int(len(data)*ratio)
    # 设置测试集的大小 test_size，与验证集大小相同
    test_size = dev_size
    # 计算训练集的大小 train_size
    # 通过总数据集大小减去验证集和测试集的大小，得到剩余的数据量作为训练集的大小
    train_size = len(data)-dev_size-test_size
    # 从打乱后的 data 中提取前 train_size 个样本作为训练集 train_data
    train_data = data[:train_size]
    # 从 data 中提取训练集之后的 dev_size 个样本作为验证集 dev_data
    dev_data = data[train_size:train_size+dev_size]
    # 从 data 中提取剩下的样本作为测试集 test_data
    test_data = data[train_size+dev_size:]
    # 返回划分后的训练集、验证集和测试集
    return train_data, dev_data, test_data


# 用于将数据列表 data 写入指定的 .jsonl 文件 filename 中，每个数据项作为独立的一行 JSON 格式字符串存储
def write_jsonl(data,filename):
    # 以写入模式 ("w") 打开指定的文件 filename，编码格式为 "utf-8"，并将文件对象引用赋予 fp 变量
    # 使用 with 语句确保文件在操作完成后自动关闭，避免文件泄露或损坏
    with open(filename,"w",encoding="utf-8") as fp:
        # 遍历 data 列表中的每个元素 example，其中每个元素是一个字典，表示一个上下文-回复对
        for example in data:
            # 将当前元素 example 转换为 JSON 格式的字符串，并确保不使用 ASCII 转义（ensure_ascii=False），这样可以保留非 ASCII 字符（如中文字符）
            # json_str 是 example 的 JSON 字符串表示形式
            json_str = json.dumps(example,ensure_ascii=False)
            # 将 json_str 写入文件 fp 中，并在每个 JSON 字符串后添加换行符 \n，以确保每条记录独占一行，符合 .jsonl（JSON Lines）文件格式
            fp.write(json_str+"\n")


# 定义 main 函数，用于数据预处理、数据集拆分和结果写入
# raw_data_path: 原始数据的路径
# more_data_path: 额外数据的路径，默认为 None
# output_dir: 输出文件夹的路径，默认为当前目录 "."
# ratio: 用于划分验证集和测试集的比例，默认为 0.1
# n: 限定返回的单轮“search”对话的最大数量，默认为 None
def main(raw_data_path, more_data_path=None, output_dir=".",ratio=0.1,n=None):
    # 检查输出目录是否存在，如果不存在则创建该目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 初始化空列表 data
    data = []
    # 会返回包含多轮和单轮“search”对话的列表
    data = process_dir(raw_data_path,data,n)
    # 检查是否提供了 more_data_path。如果提供，则调用 process_dir_v2 读取更多的对话数据并添加到 data 列表中
    # process_dir_v2 与 process_dir 不同，它逐个文件处理并追加每个文件中所有的对话记录
    if more_data_path is not None:
        data = process_dir_v2(more_data_path,data)
    # 调用 split_data 函数，按指定的 ratio 将数据拆分为训练集、开发集和测试集
    train_data, dev_data, test_data = split_data(data,ratio)
    # 将训练集 train_data 转换为对话轮次格式，然后写入 .jsonl 文件
    # 如果 n 参数为 None（即不限制单轮对话数量），则文件命名为 train.full.jsonl；否则为 train.jsonl
    write_jsonl(
        data_to_turns(train_data),
        os.path.join(output_dir,"train.jsonl" if n is not None else "train.full.jsonl")
    )
    # 将开发集 dev_data 转换为对话轮次格式，并写入文件
    write_jsonl(
        data_to_turns(dev_data),
        os.path.join(output_dir,"dev.jsonl" if n is not None else "dev.full.jsonl" )
    )
    # 将测试集 test_data 转换为对话轮次格式并写入文件
    write_jsonl(
        data_to_turns(test_data),
        os.path.join(output_dir,"test.jsonl" if n is not None else "test.full.jsonl")
    )



if __name__ == "__main__":
    # 指定原始数据目录 enhanced_data，额外数据目录 enhanced_more，和 n=None（即不限制 single 数量）
    #main("enhanced_hotel_data",more_data_path="enhanced_more",n=1500)
    main("enhanced_data",more_data_path="enhanced_more",n=None)