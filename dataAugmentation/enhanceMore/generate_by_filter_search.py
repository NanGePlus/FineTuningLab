import json
import random
import re
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
# 导入本应用程序提供的方法
from my_llm import myLLM




PROMPT_TEMPLATE = """
你的任务是根据用户对酒店的选择条件，生成用户查询的问句
尽量覆盖口语化的说法，#注意不要说的太机械重复#
酒店包含属性分别是: 酒店类型(type)、价格(price)、评分(rating)、酒店设施(facilities)。
在JSON格式输入中的key有：type, facilities, price_range_upper, price_range_lower, rating_range_lower, rating_range_upper

以JSON格式输出，只包含字段content: string类型，即用户的查询问句

examples: 
{examples}

input:
{input_text}

output:

"""

# 写好的一些例子
examples = [
  [{"price_range_upper": 500, "type": "豪华型" }, 
   { "content": "帮我找个价格在500元之内的豪华酒店。" }],
  [{"price_range_lower": 400, "rating_range_lower": 4}, 
   { "content": "我要订一个价格在400元之上的不低于4分的宾馆" }],
  [{"price_range_upper": 500, "facilities": ["热水","wifi"]}, 
   { "content": "给我查查酒店，有热水和wifi的，价格在500以里的" }],
  [{"price_range_upper": 500, "type": "经济型"}, 
   { "content": "你好，有价格在500以内的酒店可以订吗，经济型的就行" }],
  [{"price_range_upper": 200, "rating_range_lower": 4}, 
   { "content": "请给我订个评分高于4，价格在200元以上酒店。" }],
  [{"price_range_lower": 300, "facilities": ["洗衣", "洗澡"] }, 
   { "content": "有人吗，订一个在300元之上的能洗衣洗澡的宾馆噢" }],
  [{"price_range_upper": 400, "type": "经济型"}, 
   { "content": "帮我找个价格比400低的经济酒店。" }],
  [{"price_range_upper": 900, "facilities": ["停车"] }, 
   { "content": "我要找一个比900元便宜的酒店，最好能停车的啊" }],
  [{"price_range_upper": 500, "facilities": ["棋牌室"]}, 
   { "content": "帮个忙，找一下价格在500内的酒店，要有棋牌室的噢" }],
  [{"price_range_upper": 800, "type":"舒适型"}, 
   { "content": "我要找价格在800以内的酒店，给我查查有舒适的吗" }],
]

# 可选筛选条件范围如下
keys = ['type', 'facilities', 'price_range_upper', 'price_range_lower', 'rating_range_lower', 'rating_range_upper']
types = ['豪华型','舒适型','经济型']
facilities = ['热水','wifi','停车场','按摩','棋牌室','洗衣机','泳池']



# 对大模型返回的内容进行格式化处理
# 格式化响应，对输入的文本进行段落分隔、添加适当的换行符，以及在代码块中增加标记，以便生成更具可读性的输出
def format_response(response):
    # 使用正则表达式 \n{2, }将输入的response按照两个或更多的连续换行符进行分割。这样可以将文本分割成多个段落，每个段落由连续的非空行组成
    paragraphs = re.split(r'\n{2,}', response)
    # 空列表，用于存储格式化后的段落
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照```分割成多个部分，代码块和普通文本交替出现
            parts = para.split('```')
            for i, part in enumerate(parts):
                # 检查当前部分的索引是否为奇数，奇数部分代表代码块
                if i % 2 == 1:  # 这是代码块
                    # 将代码块部分用换行符和```包围，并去除多余的空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # 将分割后的部分重新组合成一个字符串
            para = ''.join(parts)
        else:
            # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace('. ', '.\n')
        # 将格式化后的段落添加到formatted_paragraphs列表
        # strip()方法用于移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n等）
        formatted_paragraphs.append(para.strip())
    # 将所有格式化后的段落用两个换行符连接起来，以形成一个具有清晰段落分隔的文本
    return '\n\n'.join(formatted_paragraphs)

# 调用大模型生成数据
def generate_data(arguments):
    # 初始化大模型
    llm = myLLM()
    # 为了保证生成问句的多样性，prompt中的例子是从写好的一些例子中做随机挑选的
    example_str = ""
    for i in range(4): # 这里挑选了4条例子给到prompt
        example = random.choice(examples)
        example_str += json.dumps(example[0], ensure_ascii=False)+"\n"
        example_str += json.dumps(example[1], ensure_ascii=False)+"\n"
    # 构造prompt
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    # 构造chain
    chain = prompt | llm
    # 运行chain
    result = chain.invoke(
        {
            "examples": example_str,
            "input_text": arguments
        }
    )
    response = str(format_response(result.content))
    # print(f"格式化的搜索结果format_response: {response}")
    try:
        # 构造返回数据
        result = [
            {'role':'user','content':json.loads(response)},
            {'role':'search','arguments':arguments}
        ]
        return result
    except:
        return []


# 生成只有价格上限或下限的数据
def generate_price_bound(nums):
    results = []
    for i in tqdm(range(nums)):
        arguments = {
            random.choice(['price_range_upper','price_range_lower']): random.randint(2,12)*100,
        }
        results.append(generate_data(arguments))
    return results


# 生成价格范围的数据
def generate_price_range(nums):
    results = []
    for i in tqdm(range(nums)):
        price_range_lower = random.randint(2,12)*100
        price_range_upper = price_range_lower + random.randint(1,8)*100
        arguments = {
            'price_range_lower': price_range_lower,
            'price_range_upper': price_range_upper,
        }
        results.append(generate_data(arguments))
    return results


# 生成各种条件组合查询的数据
def generate_misc_filter(nums):
    results = []
    for i in tqdm(range(nums)):
        arguments = {
            'price_range_upper': random.randint(2,12)*100, # 限价格上限的多
            'rating_range_lower': random.randint(2,4),     # 限评分下限的多
            'facilities': random.sample(facilities, k=2),
            'type': random.choice(types),
        }
        # 从 arguments 字典的键中随机选择 2 个键
        keys_to_remove = random.sample(list(arguments.keys()), 2)
        # 移除选中的键，以生成多样化的筛选条件组合
        for key in keys_to_remove:
            arguments.pop(key)
        results.append(generate_data(arguments))
    return results



if __name__ == "__main__":
    results = generate_price_bound(20)
    with open('price_bound.json', 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
    print('完成生成只有价格上限或下限的数据，共写入 20 条到price_bound.json')

    results = generate_price_range(20)
    with open('price_range.json', 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
    print('完成生成价格范围的数据，共写入 20 price_range.json')

    results = generate_misc_filter(20)
    with open('misc_filter.json', 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
    print('完成生成各种条件组合查询的数据，共写入 20 misc_filter.json')
