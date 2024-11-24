from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import re
from pydantic import BaseModel, Field
# 导入本应用程序提供的方法
from my_llm import myLLM




class Utterance(BaseModel):
    text: str = Field(description="重写后的句子文本")


PROMPT_TEMPLATE = """
给定句子:
{utterance}
请将句中的"{phrase}"替换为"{replacement}"。
并重新组织该句子，保证其意思不变的前提下，语言通顺。
确保你输出与原句语言保持一致。
按以下形式输出结果：
{format_instruction}
"""


# 替换句子中的短语，并重写句子使其通顺
class UtteranceRewriter:
    def __init__(self):
        # 设置大模型
        self.llm = myLLM()
        self.output_parser = PydanticOutputParser(pydantic_object=Utterance)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            format_instruction=self.output_parser.get_format_instructions(),
        )

        # 定义chain
        self.chain = self.prompt | self.llm

    # 重写语句
    def rewrite(self, utterance: str, phrase: str, replacement: str) -> str:
        response = self.chain.invoke(
            {
                "utterance": utterance,
                "phrase": phrase,
                "replacement": replacement,
            }
        )
        # 使用正则表达式从返回信息中结构出完成对话
        response = re.search(r'```json\n(.*?)\n```', response.content, re.S).group(1)
        utterance = self.robust_parser.parse(response)
        return utterance.text

if __name__ == "__main__":
    # 示例数据
    # [
    #     {
    #         "role": "user",
    #         "content": "好的，时间还是可以的，吃完饭我可能会在这边住下，能帮我推荐一个提供国际长途电话的高档型酒店吗？"
    #     },
    #     {
    #         "role": "search",
    #         "arguments": {
    #             "facilities": [
    #                 "暖气"
    #             ],
    #             "type": "高档型"
    #         }
    #     },
    #     {
    #         "role": "return",
    #         "records": [
    #             {
    #                 "name": "北京东方花园饭店",
    #                 "type": "高档型",
    #                 "address": "北京东城区东直门南大街6号",
    #                 "subway": "东直门地铁站D口",
    #                 "phone": "010-64168866",
    #                 "facilities": [
    #                     "公共区域和部分房间提供wifi",
    #                     "宽带上网",
    #                     "国际长途电话",
    #                     "吹风机",
    #                     "24小时热水",
    #                     "暖气",
    #                     "西式餐厅",
    #                     "中式餐厅",
    #                     "残疾人设施",
    #                     "会议室",
    #                     "健身房",
    #                     "无烟房",
    #                     "商务中心",
    #                     "酒吧",
    #                     "棋牌室",
    #                     "早餐服务",
    #                     "接机服务",
    #                     "接待外宾",
    #                     "洗衣服务",
    #                     "行李寄存",
    #                     "租车",
    #                     "叫醒服务"
    #                 ],
    #                 "price": 677.0,
    #                 "rating": 4.6,
    #                 "hotel_id": 87
    #             }
    #         ]
    #     },
    #     {
    #         "role": "assistant",
    #         "content": "那就建议你去北京东方花园饭店好了，比较经济实惠。"
    #     }
    # ]

    utterance = '好的，时间还是可以的，吃完饭我可能会在这边住下，能帮我推荐一个提供国际长途电话的高档型酒店吗？'
    phrase = '国际长途电话'
    replacement = '国际电话'

    new_text = UtteranceRewriter().rewrite(utterance, phrase, replacement)
    print('new_text is :', new_text)




