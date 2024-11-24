# 导入 random 模块：用于生成随机数或随机选择
# 导入 os 模块：用于访问操作系统功能，如路径和文件操作
# 导入 json 模块：用于解析和生成 JSON 数据格式
import random, os, json
# 从 rewrite 模块导入 UtteranceRewriter 类。自定义模块，提供了一个重写语句的功能
from rewrite import UtteranceRewriter
# 从 paraphrase 模块导入 Paraphraser 类，用于对输入句子进行复述或重构
from paraphrase import Paraphraser
# 从 add_qa_turn 模块中导入 QAGen 类，用于生成问答回合
from add_qa_turn import QAGen
# 从 add_qa_turn2 模块中导入 QAGen2 类，另一种生成问答回合的方法
from add_qa_turn2 import QAGen2
# 从 add_bye_turn 模块导入 ByeGen 类，可能用于生成结束对话的内容
from add_bye_turn import ByeGen




# 创建 UtteranceRewriter 类的实例，赋值给 rewriter 变量，用于重写语句
rewriter = UtteranceRewriter()
# 创建 Paraphraser 类的实例，赋值给 paraphraser 变量，用于复述输入内容
paraphraser = Paraphraser()
# 创建 QAGen 类的实例，赋值给 qa_gen 变量，准备用于生成问答回合
qa_gen = QAGen()
# 创建 QAGen2 类的实例，赋值给 qa_gen2 变量，准备用于其他类型的问答生成
qa_gen2 = QAGen2()
# 创建 ByeGen 类的实例，赋值给 bye_gen 变量，用于生成结束对话的内容
bye_gen = ByeGen()


# 定义：flip_a_coin 是一个用于模拟掷硬币的函数
# 参数：p 是成功的概率值（0 到 1 之间的浮点数）
# random.random() 生成一个 [0, 1) 范围内的随机浮点数
# 如果生成的随机数小于 p，则返回 True，表示“成功”
# 否则返回 False，表示“失败”
# 用途：此函数可用于决定随机事件是否发生，概率由参数 p 控制
def flip_a_coin(p):
    return random.random() < p


# 定义：one_hotel_only 检查对话中是否仅返回一个酒店信息，验证对话是否包含唯一的酒店信息
# 参数：dialog 是一个对话的列表，其中每个元素是字典，包含对话回合的信息
def one_hotel_only(dialog):
    # 初始化 hotel_found 为 False
    hotel_found = False
    # 遍历 dialog 中的每个 turn（对话回合）
    for turn in dialog:
        # 如果 turn["role"] 为 "return"，表示是包含酒店信息的回合
        if turn["role"] == "return":
            # 将 hotel_found 设为 True
            hotel_found = True
            # 检查 turn["records"] 的长度是否大于 1（即返回了多个酒店），如果是则返回 False
            if len(turn["records"]) > 1:
                return False
    # 循环结束后，如果 hotel_found 为 True 且没有返回多个酒店，则返回 True
    return hotel_found


# 定义：one_turn_only 检查对话中是否只有一个 assistant 回合，验证对话中是否只有一个助手发言回合
# 参数：dialog 是一个对话的列表，其中每个元素是字典，包含对话回合的信息
def one_turn_only(dialog):
    # 初始化 assistant_count 为 0
    assistant_count = 0
    # 遍历 dialog 中的每个 turn
    for turn in dialog:
        # 如果 turn["role"] 为 "assistant"，增加 assistant_count
        if turn["role"] == "assistant":
            assistant_count += 1
    # # 循环结束后，返回 assistant_count == 1，表示是否仅有一个 assistant 回合
    return assistant_count == 1


# 定义：get_last_user_turn 返回 dialog 列表中从索引 i 向前找到的最后一个用户发言回合，在对话中查找并返回给定位置之前的最近一次用户发言，用于分析上下文或处理响应
# 参数：dialog 是一个对话的列表，i 是当前索引位置，从该位置开始向前搜索用户发言
def get_last_user_turn(dialog, i):
    # 从 i - 1 开始递减，遍历 dialog，直到 0
    for j in range(i - 1, -1, -1):
        # 如果 dialog[j]["role"] 为 "user"，返回该 turn
        if dialog[j]["role"] == "user":
            return dialog[j]
    # 如果找不到用户发言，返回 None
    return None



# 定义：enhance 函数用于对 dialog（对话列表）进行增强，可能修改用户发言并添加新的问答回合或结束语
# 参数：dialog 是一个列表，每个元素是一个包含对话回合信息的字典
# 返回：修改后的 dialog 和布尔值 changed，指示是否进行了任何修改
def enhance(dialog):
    # 初始化 changed 变量为 False，用于跟踪对话是否被修改
    changed = False
    # 调整酒店设施相关语句
    # 循环遍历 dialog，i 是当前回合的索引，turn 是当前的对话回合
    for i, turn in enumerate(dialog):
        # 检查当前回合的 role 是否是 "search"，只有这种回合才会被处理
        if turn["role"] == "search":
            # 如果 turn["arguments"] 中包含 facilities 键，则将其值赋给 facilities 变量
            if "facilities" in turn["arguments"]:
                facilities = turn["arguments"]["facilities"]
                # 遍历 facilities 列表，j 是索引，facility 是当前的设施名称
                for j, facility in enumerate(facilities):
                    # 如果 facility 的长度大于 4 或者 flip_a_coin(0.2) 返回 True，则满足修改条件
                    if len(facility) > 4 or flip_a_coin(0.2):
                        # 调用 get_last_user_turn 获取 i 索引之前的最后一个用户回合。如果存在，将 user_turn 赋值为该回合
                        user_turn = get_last_user_turn(dialog, i)
                        if user_turn is not None:
                            # 获取 user_turn 的 content（用户发言），如果 facility 不在发言中，跳过该 facility
                            utterance = user_turn["content"]
                            if facility not in utterance:
                                continue
                            # 使用 paraphraser.gen 生成 facility 的同义词
                            paraphrase = paraphraser.gen(facility)
                            # 调用 rewriter.rewrite 用同义词替换用户发言中的 facility
                            new_utterance = rewriter.rewrite(utterance, facility, paraphrase)
                            # 更新 user_turn 的 content 为新的发言
                            user_turn["content"] = new_utterance
                            # 更新 facilities 列表中的对应元素为同义词
                            facilities[j] = paraphrase
                            # 设置 changed 为 True，表示对话已被修改
                            changed = True


    # 如果 one_hotel_only(dialog) 返回 True 且 flip_a_coin(0.3) 成功
    if one_hotel_only(dialog):
        if flip_a_coin(0.3):
            # 调用 qa_gen.gen 生成新的问答回合，并将其添加到 dialog，同时设置 changed 为 True
            new_turns = qa_gen.gen(dialog)
            if new_turns is not None:
                dialog.extend(new_turns)
                changed = True

    # 如果 dialog 包含多个酒店信息且 flip_a_coin(0.3) 成功
    else:
        if flip_a_coin(0.3):
            # 调用 qa_gen2.gen 生成新的问答回合，并将其添加到 dialog，同时设置 changed 为 True
            new_turns = qa_gen2.gen(dialog)
            if new_turns is not None:
                dialog.extend(new_turns)
                changed = True

    # 如果 dialog 包含唯一的酒店信息且 flip_a_coin(0.5) 成功
    if one_hotel_only(dialog):
        if flip_a_coin(0.5):
            # 调用 bye_gen.gen 生成新的结束语回合，并将其添加到 dialog，同时设置 changed 为 True
            new_turns = bye_gen.gen(dialog)
            if new_turns is not None:
                dialog.extend(new_turns)
                changed = True

    # 返回修改后的 dialog 和 changed 布尔值，指示是否进行了修改
    return dialog, changed



# 定义：main 函数用于遍历 raw_data 文件夹中的所有 JSON 文件，对其中的对话数据进行增强处理，并将结果输出到 enhanced_data 文件夹
# start（默认值为 0）：开始处理的文件编号
# end（默认值为 None）：结束处理的文件编号
def main(start=0,end=None):
    # 定义输出目录为 enhanced_data，用于存储处理后的对话文件
    output_dir = "enhanced_data"
    # 使用 os.listdir 获取 raw_data 文件夹中的所有文件名并逐一遍历
    for filename in os.listdir("raw_data"):
        # 将文件名去掉 .json 后缀，并将其转换为整数，作为 id
        # 如果 id 小于 start，跳过当前文件
        # 如果 end 非空且 id 大于等于 end，停止遍历
        id = int(filename.replace(".json", ""))
        if id < start:
            continue
        if end is not None and id >= end:
            break

        # 打印当前正在处理的文件名
        print(filename)

        # 使用 open 以只读模式打开 raw_data 文件夹中的文件，编码为 utf-8
        # 使用 json.load 将文件内容加载为 Python 数据结构（dialog 对象）
        with open(os.path.join("raw_data", filename), 'r', encoding='utf-8') as ifp:
            dialog = json.load(ifp)
            # 使用 try-except 块调用 enhance 函数对 dialog 进行处理，并捕获可能的异常
            # 如果处理过程中发生异常，设置 changed 为 False 表示未进行修改
            try:
                dialog, changed = enhance(dialog)
            except:
                changed = False

            # 如果 changed 为 True，修改文件名，在原文件名的基础上添加 _，表明文件已被增强
            # 打印“Enhanced [filename]”来记录增强的文件
            if changed:
                filename = filename.replace(".json", "_.json")
                print("Enhanced {}".format(filename))

            # 打开 output_dir 文件夹中的对应输出文件，以写模式存储增强后的 dialog，编码为 utf-8
            # 使用 json.dump 将 dialog 转储到文件中，缩进为 4 格，确保非 ASCII 字符正常保存
            # 使用 ofp.close() 关闭文件以释放资源
            ofp = open(os.path.join(output_dir, filename), 'w', encoding='utf-8')
            json.dump(dialog, ofp, indent=4, ensure_ascii=False)
            ofp.close()



if __name__ == "__main__":
    main()