from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import re
from pydantic import BaseModel, Field
# 导入本应用程序提供的方法
from my_llm import myLLM


class Phrase(BaseModel):
    text: str = Field(description="改写后的文本")


PROMPT_TEMPLATE = """
给定短语:
{phrase}

请用口语的方式paraphrase这个短语。例如:
公共区域和部分房间提供wifi: 有wifi | 无线上网 | 无线网 
中式餐厅: 中餐厅 | 中餐
国际长途电话: 国际电话 | 国际长途
免费市内电话: 免费电话 | 免费市话
酒店各处提供wifi: wifi全覆盖 | 无线上网 | 无线网

确保你输出的短语与原短语不同。
确保你输出的是中文（wifi这个词可以保留）。
按以下形式输出结果：
{format_instruction}
"""


# 为给定短语生成一个口语化的改写
class Paraphraser:
    def __init__(self):
        # 设置大模型
        self.llm = myLLM()
        self.output_parser = PydanticOutputParser(pydantic_object=Phrase)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            format_instruction=self.output_parser.get_format_instructions(),
        )

        # 定义chain
        self.chain = self.prompt | self.llm

    def gen(self,phrase: str) -> str:
        response = self.chain.invoke(
            {
                "phrase": phrase
            }
        )

        # 使用正则表达式从返回信息中结构出完成对话
        response = re.search(r'```json\n(.*?)\n```', response.content, re.S).group(1)
        phrase = self.robust_parser.parse(response)

        if ":" in phrase.text:
            phrase.text = phrase.text.split(":")[1]
        return phrase.text



if __name__ == "__main__":
    text = '国际长途电话'

    new_text = Paraphraser().gen(text)
    print('new_text is :',new_text)



