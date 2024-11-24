import random
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import json
import re
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

请补充生成一轮对话，模拟user，针对assistant提供的酒店，随机选则酒店的任意一个属性进行比较性提问，包括：价格、评分、地址、电话、是否包含某设置等。
例如: 
{examples}
etc.

请用很口语的方式提问。Try your best to paraphrase the question!尽可能改写问题！
注意不要重复提问之前对话中assistant已经提供的信息。

然后模拟assistant，根据酒店的实际属性值，回答该问题。答案必须与给出的酒店的真实信息一致。
对话必须先由user提问，然后assistant回答。每人只一轮。
只输出一轮对话，不需要输出多轮对话，不要重复以上的对话内容。
确保你输出与原句语言保持一致。
按以下形式输出结果：
{format_instruction}
"""

examples = [
    "“哪个便宜？”",
    "“哪个高档一些？”",
    "“哪个评分高？”",
    "”哪个有wifi”",
    "“有带商务中心的吗”",
    "“有带棋牌室的吗”",
]

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Turn(BaseModel):
    role: Role = Field(description="对话角色")
    content: str = Field(description="对话内容")


class QA(BaseModel):
    dialog: List[Turn] = Field(description="对话内容")


# 为对话生成一轮对比型问答：例如哪个酒店便宜、高档、评分高、有wifi等
class QAGen2:
    def __init__(self):
        # 设置大模型
        self.llm = myLLM()
        self.output_parser = PydanticOutputParser(pydantic_object=QA)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            format_instruction=self.output_parser.get_format_instructions(),
        )

        self.chain = self.prompt | self.llm

    def gen(self, dialog: List) -> List|None:
        random.shuffle(examples)
        response = self.chain.invoke(
            {
                "dialog": json.dumps(dialog, indent=4, ensure_ascii=False),
                "examples": "\n".join(examples),
            }
        )
        try:
            response = re.search(r'```json\n(.*?)\n```', response.content, re.S).group(1)
            qa = json.loads(response).get("dialog", [])
            if isinstance(qa, list):
                if len(qa) != 2:
                    raise Exception("Invalid format")
                for turn in qa:
                    if "role" not in turn or "content" not in turn:
                        raise Exception("Invalid format")
                if qa[0]["role"] != "user" or qa[1]["role"] != "assistant":
                    raise Exception("Invalid format")
                ans = qa
            else:
                raise Exception("Invalid format")
        except:
            qa = self.robust_parser.parse(response)

            ans = []
            for turn in qa.dialog:
                t = {
                    "role": turn.role,
                    "content": turn.content,
                }
                ans.append(t)
            if len(ans) != 2:
                return None
        return ans



if __name__ == "__main__":
    text = '''
    [
        {
            "role": "user",
            "content": "游玩结束我想找一家提供无烟房的酒店住宿，有什么推荐的吗？"
        },
        {
            "role": "search",
            "arguments": {
                "facilities": [
                    "无烟房"
                ]
            }
        },
        {
            "role": "return",
            "records": [
                {
                    "name": "北京贵都大酒店",
                    "type": "高档型",
                    "address": "北京西城区广安门内大街217号",
                    "subway": "菜市口地铁站D口",
                    "phone": "010-51979888",
                    "facilities": [
                        "公共区域和部分房间提供wifi",
                        "宽带上网",
                        "吹风机",
                        "24小时热水",
                        "西式餐厅",
                        "中式餐厅",
                        "会议室",
                        "健身房",
                        "无烟房",
                        "商务中心",
                        "接站服务",
                        "接机服务",
                        "洗衣服务",
                        "行李寄存",
                        "叫醒服务"
                    ],
                    "price": 1354.0,
                    "rating": 4.7,
                    "hotel_id": 0
                },
                {
                    "name": "瑞尔威连锁饭店(北京西客站店)",
                    "type": "舒适型",
                    "address": "北京丰台区莲花池东路116-2号",
                    "subway": "北京西站地铁站A口",
                    "phone": "010-63959988",
                    "facilities": [
                        "酒店各处提供wifi",
                        "宽带上网",
                        "国际长途电话",
                        "吹风机",
                        "24小时热水",
                        "暖气",
                        "会议室",
                        "无烟房",
                        "接待外宾",
                        "行李寄存",
                        "叫醒服务"
                    ],
                    "price": 558.0,
                    "rating": 4.3,
                    "hotel_id": 1
                }
            ]
        },
        {
            "role": "assistant",
            "content": "这边为您推荐北京贵都大酒店和瑞尔威连锁饭店(北京西客站店)。"
        }
    ]
    '''

    dialog = json.loads(text)

    new_turns = QAGen2().gen(dialog)
    print('new_turns is :',new_turns)





