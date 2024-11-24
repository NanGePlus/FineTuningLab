import json
import random
import re
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
# 导入本应用程序提供的方法
from db_client import HotelDB
from my_llm import myLLM




PROMPT_TEMPLATE = """
你的任务是根据酒店名称生成查询的语句
并结合查到的结果，加工生成回复文本
尽量覆盖口语化的说法，#注意不要说的太机械重复#
选择条件为酒店名称, 在JSON格式输入中的key是name

以JSON格式输出，包含字段
- content: string类型，即用户的查询问句
- reply: string类型，回复给用户的话术

examples: 
{examples}

input:
酒店名称：{name}
查询记录：{searched}

output:

"""



# 检索业务数据库
def search(db, name):
    result = db.search({'name':name}, limit=3)
    final = []
    for r in result:
        if all(char in r['name'] for char in name):
            final.append(r)
    return final


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



if __name__ == "__main__":
    # 初始化大模型
    llm = myLLM()
    # 根据名字只查到单条记录的例子
    examples1 = [
      [
       {"str": "盛厦宾馆"}, 
       {"str": "我想订一下那个盛厦宾馆，帮我查查"},
       [{"address": "北京朝阳区东三环北路16号农展馆新馆南路", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;国际长途电话;吹风机;24小时热水;无烟房;行李寄存", "hotel_id": 584, "name": "北京盛厦宾馆", "phone": "010-65916188", "price": 258, "rating": 4.4, "subway": "团结湖地铁站B口", "type": "舒适型"}],
       {"str": "欢迎您选择北京盛厦宾馆，祝您入住愉快"}
      ],
      [
       {"str": "新世贸大酒店"}, 
       {"str": "我要订个酒店，那个新世贸大酒店"},
       [{"address": "北京丰台区永外东罗园九号楼", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;宽带上网;吹风机;24小时热水;中式餐厅;会议室;无烟房;商务中心;棋牌室;早餐服务免费;洗衣服务;行李寄存;租车;叫醒服务", "hotel_id": 715, "name": "北京新世贸大酒店", "phone": "010-52268555", "price": 223, "rating": 4.2, "subway": "景泰站地铁站D口", "type": "舒适型"}],
       {"str": "为您查到北京新世贸大酒店，您要选择这家吗" }
      ],
      [
       {"str": "孔府酒店"}, 
       {"str": "帮我找个酒店，叫孔府酒店"}, 
       [{"address": "北京海淀区马甸西路月季园18号", "facilities": "酒店提供的设施:酒店各处提供wifi;宽带上网;免费市内电话;吹风机;24小时热水;中式餐厅;无烟房;早餐服务;行李寄存;叫醒服务;收费停车位", "hotel_id": 1062, "name": "北京孔府酒店", "phone": "010-68980808", "price": 402, "rating": 4, "subway": "牡丹园地铁站C口", "type": "舒适型"}],
       {"str": "为您找到北京孔府酒店。"}
      ],
    ]
    # 根据名字查到多条记录的例子
    examples2 = [
      [
       {"str": "如家快捷酒店"}, 
       {"str": "给我订个住的地方，找找如家快捷酒店"}, 
       [{"address": "北京朝阳区左家庄中街4号", "facilities": "酒店提供的设施:所有房间提供wifi;国际长途电话;24小时热水;中式餐厅;无烟房;商务中心;早餐服务;接待外宾;洗衣服务", "hotel_id": 883, "name": "如家快捷酒店(北京国展左家庄店)", "phone": "010-64686868", "price": 257, "rating": 4.4, "subway": "柳芳地铁站B口", "type": "经济型"}, {"address": "北京朝阳区劲松八区805号楼", "facilities": "酒店提供的设施:酒店各处提供wifi;宽带上网;国际长途电话;吹风机;24小时热水;无烟房;接待外宾;行李寄存;叫醒服务", "hotel_id": 937, "name": " 如家快捷酒店(北京劲松店)", "phone": "010-87776766", "price": -1, "rating": 4.6, "subway": "劲松地铁站A口", "type": "经济型"}, {" address": "北京朝阳区安立路甲52号", "facilities": "酒店提供的设施:部分房间提供wifi;宽带上网;免费市内电话;国际长途电话;吹风机;24小时热水;会议室;无烟房;接待外宾;行李寄存;叫醒服务", "hotel_id": 1015, "name": "如家快捷酒店(北京鸟巢店)", "phone": "010-84802999",  "price": 315, "rating": 4.7, "subway": "安立路地铁站B口", "type": "经济型"}],
       {"str": "为您找到如家快捷酒店(北京鸟巢店)、如家快捷酒店(北京劲松店)、如家快捷酒店(北京国展左家庄店)，请您选择" }
      ],
      [
       {"str": "汉庭酒店"}, 
       {"str": "我想住汉庭酒店，你帮我查一下都有哪些店"}, 
       [{"address": "北京朝阳区劲松九区907号楼东二环辅路路东", "facilities": "酒店提供的设施:酒店各处提供wifi;宽带上网;国际长途电话;吹风机;24小时热水;中式餐厅;无烟房;商务中心;接待外宾;行李寄存;叫醒服务", "hotel_id": 407, "name": "汉庭酒店(北京站店)", "phone": "010-67765566", "price": 267, "rating": 4.5, "subway": "广渠门外地铁站D口", "type": "经济型"}, {"address": "北京朝阳区北苑路18号院4号楼", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;宽带上网;国际长途电话;吹风机;24小时热水;中式餐厅;无烟房;行李寄存;叫醒服务", "hotel_id": 953, "name": "汉庭酒店(北京北苑店)", "phone": "010-60606099", "price": 324, "rating": 4.4, "subway": "立水桥南地铁站C口", "type": "经济型"}, {"address": "北京西城区西绦胡同15号", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;宽带上网;吹风机;24小时热水;中式餐厅;无烟房;商务中心;早餐服务;接待外宾;洗衣服务;行李寄存;叫醒服务", "hotel_id": 648, "name": "汉庭酒店(北京鼓楼店)", "phone": "010-64000123", "price": 403, "rating": 4.3, "subway": "鼓楼大街地铁站A1口", "type": "经济型"}],
       {"str": "找到了汉庭酒店(北京鼓楼店)、汉庭酒店(北京北苑店)、汉庭酒店(北京站店)这三家，请您选择" }],
      [
       {"str": "7天酒店"}, 
       {"str": "你帮我查一下7天酒店都有哪些店"}, 
       [{"address": "北京朝阳区德胜门外黄寺大街28号", "facilities": "酒店提供的设施:所有房间提供wifi;宽带上网;24小时热水;无烟房", "hotel_id": 778, "name": "7天优品酒店(北京黄寺店)(原7天连锁酒店)", "phone": "010-59260366", "price": 332, "rating": 4.4, "subway": "安华桥地铁站D2口", "type": "经济型"}, {"address": "北京朝阳区望京南湖北路107号", "facilities": "酒店提供的设施:部分房间提供wifi;宽带上网;24小时热水;接待外宾", "hotel_id": 677, "name": "7天优品酒店(北京望京南湖东园店)(原7天连锁酒店望京南湖东园店)", "phone": "010-64725777", "price": 332, "rating": 4.1, "subway": "东湖渠地铁站D口", "type": "经济型"}, {"address": "北京海淀区定慧东里18号楼", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;宽带上网;免费国内长途电话;24小时热水;无烟房;洗衣服务", "hotel_id": 748, "name": "7天连锁酒店(北京航天桥店)", "phone": "010-88111977", "price": 316, "rating": 4.5, "subway": "西钓鱼台地铁站C口", "type": "经济型"}], {"str": "为您找到7天连锁酒店(北京航天桥店)、7天优品酒店(北京望京南湖东园店)(原7天连锁酒店望京南湖东园店)和7天优品酒店(北京黄寺店)(原7天连锁酒店)，请选择" }],
    ]
    # 读取酒店数据集中所有的酒店名字
    with open('names.json', 'r') as f:
        names = json.load(f)
    db = HotelDB()
    results = []
    for name in tqdm(names):
        # 从单条和多条记录的例子中各选一个
        example_str = ""
        example = random.choice(examples1)
        example_str += "input:\n"
        example_str += "酒店名称："+example[0]['str']+"\n"
        example_str += "查询记录："+json.dumps(example[2],ensure_ascii=False)+"\n"
        result = {'content':example[1]['str'],'reply':example[3]['str']}
        example_str += "output:\n"+json.dumps(result,ensure_ascii=False)+"\n"
        example = random.choice(examples2)
        example_str += "input:\n"
        example_str += "酒店名称："+example[0]['str']+"\n"
        example_str += "查询记录："+json.dumps(example[2],ensure_ascii=False)+"\n"
        result = {'content':example[1]['str'],'reply':example[3]['str']}
        example_str += "output:\n"+json.dumps(result,ensure_ascii=False)+"\n"
        records = search(db, name)
        searched =  json.dumps(records,ensure_ascii=False)

        # 构造prompt
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        # 构造chain
        chain = prompt | llm
        # 运行chain
        result = chain.invoke(
            {
                "name": name,
                "searched": searched,
                "examples": example_str
            }
        )
        response = str(format_response(result.content))

        try:
            response = json.loads(response)
            result = [
                {'role':'user','content':response['content']},
                {'role':'search','arguments':{'name':name}},
                {'role':'return','records': records},
                {'role':'assistant','content':response['reply']}
            ]
            # print(json.dumps(result, ensure_ascii=False, indent=4))
            results.append(result)
        except:
            pass
    # print(json.dumps(results, ensure_ascii=False, indent=4))
    with open('results.json', 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
    print('生成数据成功并保存在results.json')
