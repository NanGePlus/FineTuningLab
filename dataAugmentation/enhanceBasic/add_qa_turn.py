import random
import re
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import json
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
# 导入本应用程序提供的方法
from my_llm import myLLM



PROMPT_TEMPLATE = """
给定一段user与assistant的对话，其中search表示搜索条件，return表示返回的结果:
```
{dialog}
```

请补充生成一轮对话，模拟user，针对assistant提供的酒店，随机选则酒店的任意一个属性进行提问，包括：价格、评分、地址、电话、是否包含某设置等。
例如: 
{examples}
etc.

请用很口语的方式提问，可以说酒店的名字也可以说”这个酒店“，或”这家酒店“。Try your best to paraphrase the question!尽可能改写问题！
注意不要重复提问之前对话中assistant已经提供的信息。

然后模拟assistant，根据酒店的实际属性值，回答该问题。答案必须与给出的酒店的真实信息一致。
对话必须先由user提问，然后assistant回答。每人只一轮。
只输出一轮对话，不需要输出多轮对话，不要重复以上的对话内容。
确保你输出与原句语言保持一致。
按以下形式输出结果：
{format_instruction}
"""


# examples 是一个包含问答示例的字符串列表，表示常见的用户问询，例如房价、评分、地址、电话和设施情况等
examples = [
    "“多少钱/什么价格/每晚房价？”",
    "“评分多少？”",
    "”在什么位置/详细地址是什么？”",
    "“电话是多少/电话发我一下？”",
    "“有没有免费wifi？”",
    "“有没有商务中心？”"
]

# 定义了一个枚举类 Role，继承自 str 和 Enum，用于描述对话中角色的身份
# USER 表示用户角色，ASSISTANT 表示助手角色
# 这种枚举方式确保角色字段只能取 user 或 assistant 这两个值，从而提高数据一致性
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

# 定义了一个 Turn 类，继承自 Pydantic 的 BaseModel，用于表示对话中的一轮发言
# role 属性使用 Role 枚举来标识是谁在发言（用户或助手），并提供了描述性注释“对话角色”
# content 属性是一个字符串，用于存储发言的具体内容，带有描述性注释“对话内容”
# Pydantic 的 Field 方法用于提供额外的元数据，例如描述，方便文档生成和可读性
class Turn(BaseModel):
    role: Role = Field(description="对话角色")
    content: str = Field(description="对话内容")

# 定义了一个 QA 类，继承自 Pydantic 的 BaseModel，用于表示一个完整的对话问答
# dialog 属性是一个 Turn 对象的列表，表示整个对话的内容
# 使用 Field 方法提供描述性注释“对话内容”，说明这是对话的主体部分
class QA(BaseModel):
    dialog: List[Turn] = Field(description="对话内容")


# QAGen 类的作用是为对话生成一轮问答
# 它使用预训练的大型语言模型（如 GPT-3.5）来生成用户提问和助手回答的对话回合
# 通过模板、示例和解析器的组合，它能够处理和修复模型输出的格式问题，并确保返回符合预期结构的问答数据
# 如果生成的问答格式无效，它会尝试修复输出并返回正确的格式
# 定义了一个 QAGen 类，其目的是为对话生成问答，例如询问酒店的价格、评分、地址、电话等信息
class QAGen:
    # __init__ 是类的构造函数，在创建 QAGen 类的实例时会被调用
    def __init__(self):
        # 设置大模型
        self.llm = myLLM()
        # 创建一个 PydanticOutputParser 对象 output_parser，用于解析从模型生成的输出，并将其转换为 QA 数据模型的实例。QA 应该是一个预定义的 Pydantic 数据模型，定义了问答数据的格式
        self.output_parser = PydanticOutputParser(pydantic_object=QA)
        # 创建一个 OutputFixingParser 对象 robust_parser，它使用 output_parser 和 llm（语言模型）来修复和解析输出
        # OutputFixingParser 是一种增强型的解析器，会修复模型输出中的格式问题
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        # 这里使用 PromptTemplate.from_template(PROMPT_TEMPLATE) 从模板 PROMPT_TEMPLATE 创建一个 PromptTemplate 对象
        # partial() 方法用于在模板中插入必要的格式化指令，这里传递了从 output_parser 获取的格式化指令
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            format_instruction=self.output_parser.get_format_instructions(),
        )

        # 创建一个chain，它将语言模型 llm 和格式化的 prompt 结合在一起，形成一个可以运行的链式处理流程
        self.chain = self.prompt | self.llm


    # gen 方法用于生成一轮问答，接收一个对话列表 dialog，并返回生成的问答对，或返回 None（如果格式无效）
    def gen(self, dialog: List) -> List|None:
        # 随机打乱 examples 列表的顺序
        # examples 是一个包含对话示例的列表，用于提供给模型，帮助模型理解上下文
        random.shuffle(examples)
        # 将 dialog 转换为格式化的 JSON 字符串，并传递给 LLMChain 的 run 方法
        # 将 examples 列表中的示例连接为一个字符串，每个示例之间用换行符分隔，传递给 run 方法，模型将使用这些示例来生成合理的回答
        response = self.chain.invoke(
            {
                "dialog":json.dumps(dialog, indent=4, ensure_ascii=False),
                "examples":"\n".join(examples),
            }
        )
        try:
            # 使用正则表达式从返回信息中结构出完成对话
            response = re.search(r'```json\n(.*?)\n```', response.content, re.S).group(1)
            # 尝试将模型生成的响应 response 解析为一个 JSON 对象。如果解析成功，将其赋值给 qa
            qa = json.loads(response).get("dialog", [])

            # 检查 qa 是否是一个列表，并且该列表的长度是否为 2（因为问答应该有两个部分：用户提问和助手回答）
            if isinstance(qa, list):
                if len(qa) != 2:
                    raise Exception("Invalid format")
                # 检查每个 turn（回合）是否包含 role（角色）和 content（内容）。如果缺少这两个字段，则认为格式无效
                for turn in qa:
                    if "role" not in turn or "content" not in turn:
                        raise Exception("Invalid format")
                # 检查 qa 列表的第一个回合是否为用户提问，第二个回合是否为助手回答。如果角色顺序不对，则抛出异常
                if qa[0]["role"] != "user" or qa[1]["role"] != "assistant":
                    raise Exception("Invalid format")
                # 如果上述检查都通过，将 qa 赋值给 ans，作为最终的问答结果
                ans = qa
            # 如果 qa 不是列表，抛出异常，表示格式无效
            else:
                raise Exception("Invalid format")
        # 如果上述 try 块中的任何一行抛出异常，则使用 robust_parser 来解析响应
        # robust_parser 是一个更强健的解析器，能够处理格式问题
        except:
            qa = self.robust_parser.parse(response)

            # 将 qa.dialog 中的每个回合（turn）转换为字典形式，包含 role 和 content 字段，并将其添加到 ans 列表中
            ans = []
            for turn in qa.dialog:
                t = {
                    "role": turn.role,
                    "content": turn.content,
                }
                ans.append(t)
            # 检查 ans 列表的长度是否为 2。如果不是，返回 None，表示生成的问答无效
            if len(ans) != 2:
                return None
        # 返回生成的问答列表 ans
        return ans



if __name__ == "__main__":
    text = '''
    [
        {
            "role": "user",
            "content": "这么多好玩的地方啊，真不错啊，溜达完我想找个地方休息，你可以帮我推荐一个价格是400-500元的酒店吗？"
        },
        {
            "role": "search",
            "arguments": {
                "price_range_lower": 400,
                "price_range_upper": 500
            }
        },
        {
            "role": "return",
            "records": [
                {
                    "name": "北京圆山大酒店",
                    "type": "舒适型",
                    "address": "北京西城区裕民路2号",
                    "subway": "安华桥地铁站A口",
                    "phone": "010-62010033",
                    "facilities": [
                        "酒店各处提供wifi",
                        "宽带上网",
                        "国际长途电话",
                        "吹风机",
                        "24小时热水",
                        "中式餐厅",
                        "残疾人设施",
                        "会议室",
                        "健身房",
                        "无烟房",
                        "酒吧",
                        "棋牌室",
                        "早餐服务",
                        "接站服务",
                        "接机服务",
                        "接待外宾",
                        "洗衣服务",
                        "行李寄存",
                        "租车",
                        "叫醒服务"
                    ],
                    "price": 416.0,
                    "rating": 4.3,
                    "hotel_id": 180
                }
            ]
        },
        {
            "role": "assistant",
            "content": "你觉得北京圆山大酒店怎么样呢，我个人还是觉得不错的。"
        }
    ]
    '''

    dialog = json.loads(text)

    new_turns = QAGen().gen(dialog)
    print('new_turns is :',new_turns)

