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

请补充生成一轮对话，模拟user表示愿意入住这个酒店。你可以使用不同的说法。用很口语的方式表达。亲切、礼貌一些。
并模拟assistant，回答“好的，祝您入住愉快”。
对话必须先由user开始，然后assistant回复。每人只一轮。
只输出一轮对话，不需要输出多轮对话，不要重复以上的对话内容。
确保你输出与原句语言保持一致。
按以下形式输出结果：
{format_instruction}
"""


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Turn(BaseModel):
    role: Role = Field(description="对话角色")
    content: str = Field(description="对话内容")


class Bye(BaseModel):
    dialog: List[Turn] = Field(description="对话内容")


# 为对话生成一轮结束语：例如“好的，祝您入住愉快”
class ByeGen:
    def __init__(self):
        self.llm = myLLM()
        self.output_parser = PydanticOutputParser(pydantic_object=Bye)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            format_instruction=self.output_parser.get_format_instructions(),
        )

        self.chain = self.prompt | self.llm

    def gen(self, dialog: List) -> List|None:
        response = self.chain.invoke(
            {
                "dialog": json.dumps(dialog, indent=4, ensure_ascii=False),
            }
        )
        try:
            response = re.search(r'```json\n(.*?)\n```', response.content, re.S).group(1)
            qa = json.loads(response).get("dialog", [])
            if isinstance(qa,list):
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

    new_turns = ByeGen().gen(dialog)
    print('new_turns is :',new_turns)



