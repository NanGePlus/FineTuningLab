import warnings
warnings.filterwarnings('ignore')
import json
import requests
import gradio as gr
import pandas as pd
from db_client import HotelDB




# 功能：实例化 HotelDB 类以创建一个数据库对象
db = HotelDB()


# 调用模型进行推理
def get_completion(context):
    url = "http://localhost:8012/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "context":context
    }
    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # HTTP 200 表示请求成功
    if response.status_code == 200:
        try:
            response_data = response.json()  # 将响应转换为字典
        except json.JSONDecodeError:
            raise ValueError(f"无法解析响应为 JSON: {response.text}")
    else:
        raise ValueError(f"请求失败，状态码: {response.status_code}, 响应: {response.text}")
    # 从 JSON 数据中提取内容
    return response_data["response"]


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


# 功能：定义一个名为 remove_search_history 的函数，该函数接受一个上下文列表并删除与搜索相关的条目
# 参数 context 是包含多个条目的列表，可能代表对话历史或上下文信息
def remove_search_history(context):
    # 功能：初始化循环变量并开始遍历上下文列表
    # i = 0：初始化索引变量 i 为0，用于遍历 context 列表
    # 当 i 小于 context 列表的长度时，继续循环
    i = 0
    while i < len(context):
        # 功能：检查当前上下文条目的角色是否为 search 或 return
        # context[i]['role']：访问 context 列表中第 i 个元素的 role 属性
        # in ['search','return']：检查角色是否在给定的角色列表中，表示该条目与搜索结果相关
        if context[i]['role'] in ['search','return']:
            # 功能：删除与搜索相关的上下文条目
            # 如果当前条目的角色是 search 或 return，则从 context 列表中删除该条目
            del context[i]
        # 功能：如果当前条目不是 search 或 return，则移动到下一个条目
        # i += 1：增加索引 i，以便检查下一个条目。只有在没有删除当前条目时才会增加索引，以确保所有条目都被检查
        else:
            i += 1


# 功能：定义一个 chat 函数，用于处理用户输入并生成模型响应
# user_input：用户输入的内容
# chatbot：保存当前对话记录的列表
# context：上下文列表，用于存储整个对话的角色和内容
# search_field 和 return_field：用于显示搜索条件和返回的结果
def chat(user_input, chatbot, context, search_field, return_field):
    print(f"====当前输入参数为user_input:\n{user_input},chatbot:{chatbot},context:{context},search_field:{search_field},return_field:{return_field}\n\n")
    # 功能：将用户输入添加到上下文中
    # 在上下文中添加一条记录，表明这是来自用户的消息，以便后续生成响应时使用完整上下文
    context.append({'role':'user','content':user_input})
    # 功能：构建提示并生成模型响应
    # build_prompt(context)：调用 build_prompt 函数，基于上下文生成输入提示
    # get_completion(...)：传入提示，调用模型生成响应，将结果赋给 response
    response = get_completion(context)
    print(f"====第1次模型推理结果response为:\n{response}\n\n")
    #print(response)
    # 功能：判断模型响应中是否包含 "search" 命令
    # 检查 response 是否包含 "search"，如果包含则执行搜索功能，表示模型判断用户输入需要搜索数据库
    if "search" in response:
        # 功能：解析模型响应中的搜索条件
        # parse_json(response)：从模型的响应中提取JSON格式的查询条件，方便后续在数据库中执行搜索操作
        # search_query：存储解析出来的查询条件
        search_query = parse_json(response)
        print(f"====模型推理结果中获取的search_query:\n{search_query}\n\n")
        # 功能：检查搜索条件是否成功解析
        # if search_query is not None:：确保 search_query 非空，这样可以避免由于解析错误导致的后续代码异常
        if search_query is not None:
            # 功能：格式化搜索查询条件并将其赋给 search_field
            # json.dumps(..., indent=4, ensure_ascii=False)：将 search_query 转换为格式化的JSON字符串，ensure_ascii=False确保显示中文字符
            search_field = json.dumps(search_query,indent=4,ensure_ascii=False)
            print(f"====模型推理结果中获取的search_field:\n{search_field}\n\n")
            # 功能：清理上下文中的搜索历史并记录新的搜索查询
            # remove_search_history(context)：调用 remove_search_history 函数，删除所有与搜索相关的上下文条目
            # context.append({'role': 'search', 'arguments': search_query})：在上下文中添加新的搜索请求记录
            remove_search_history(context)
            context.append({'role':'search','arguments':search_query})
            # 功能：执行数据库搜索
            # db.search(search_query, limit=3)：调用 HotelDB 类的 search 方法，以 search_query 作为条件在数据库中搜索，limit=3 限制返回的结果数量为3个
            # return_field：保存搜索结果，包含满足条件的酒店记录
            return_field = db.search(search_query, limit=3)
            print(f"====根据检索条件业务数据库中查询的return_field:\n{return_field}\n\n")
            # 功能：将搜索结果添加到上下文中
            context.append({'role':'return','records':return_field})
            # 功能：为搜索结果设置字段名称
            # if return_field:：检查 return_field 是否包含数据
            # keys = [...]：指定用于展示的酒店信息字段，如名称、地址、电话等
            keys = []
            if return_field:
                keys = ['name', 'address', 'phone', 'price', 'rating', 'subway', 'type', 'facilities']
            # 功能：构建一个用于存储搜索结果的字典
            # {key: [item[key] for item in return_field] for key in keys}：基于 keys 字段列表，从 return_field 的每个条目提取对应字段的数据，构建一个包含每个字段列表的字典
            # data = data or {"hotel": []}：如果 data 为空，设为包含空列表的字典，避免后续使用出错
            data = {key: [item[key] for item in return_field] for key in keys}
            data = data or {"hotel": []}
            # 功能：将搜索结果转换为数据框格式
            # pd.DataFrame(data)：创建 pandas 数据框，便于展示和分析搜索结果，return_field 现在存储了格式化的数据框
            return_field = pd.DataFrame(data)
            print(f"====根据检索条件业务数据库中查询的return_field转为数据框格式:\n{return_field}\n\n")
            # 功能：再次生成模型响应，将搜索结果发给模型
            # build_prompt(context)：基于更新的上下文（包括搜索结果）构建提示
            # get_completion(...)：再次生成响应，使模型能够参考最新的搜索结果
            response = get_completion(context)
            print(f"====第2次模型推理结果为response:\n{response}\n\n")
    # 功能：清理模型响应中的 "assistant" 字符串
    # response.replace("assistant", "")：移除响应中的 "assistant" 标识符，使返回的文本更简洁
    reply = response.replace("assistant", "")
    print(f"====模型推理最终给出的响应结果为reply:\n{reply}\n\n")
    # 功能：将用户输入和模型响应添加到对话历史中
    # 将 (user_input, reply) 这一对消息追加到 chatbot 中，便于记录完整的对话历史
    chatbot.append((user_input, reply))
    # 功能：将模型响应添加到上下文中
    # 在上下文中记录模型生成的回复内容
    context.append({'role':'assistant','content':reply})
    # 功能：返回更新后的对话状态
    # 返回空字符串（通常用于清空输入框）、更新后的对话历史、上下文，以及搜索字段和结果字段的状态
    print(f"====最终返回结果内容为user_input置空,chatbot:\n{chatbot},context:{context},search_field:{search_field},return_field:{return_field}\n\n")
    return "", chatbot, context, search_field, return_field


# 功能：定义一个 reset_state 函数，用于将状态重置为初始值
# 解释：这是一个无参数的函数，当被调用时，将返回一组用于初始化或重置的值，以确保系统没有残留的历史数据或状态
def reset_state():
    # 功能：返回一组初始化后的变量，以便重置对话相关的状态
    # []：第一个空列表，用于清空 chatbot 对话记录，使对话历史从零开始
    # []：第二个空列表，用于清空 context 上下文记录，确保上下文没有之前的消息记
    # ""：空字符串，作为 search_field 的初始值，这意味着当前没有查询条件
    # ""：另一个空字符串，作为 return_field 的初始值，表示当前没有搜索结果
    # None：作为 return_field 的默认值之一，表示没有数据框格式的结果内容
    return [], [], "", "", None


# 功能：定义 main 函数，该函数负责构建整个 Gradio 应用的用户界面
def main():
    # 功能：创建一个 gr.Blocks 容器（命名为 demo），这是 Gradio 提供的高级接口，用于构建和组织复杂的用户界面布局
    with gr.Blocks() as demo:
        # 功能：添加一个 HTML 标题，显示在界面的顶部，表明这是一个基于 Qwen2 LoRA 模型的酒店聊天机器人
        gr.HTML("""<h1 align="center">Hotel Chatbot (Qwen2 LoRA)</h1>""")

        # 创建一个水平行 Row，在其中放置列 Column
        with gr.Row():
            # 功能：创建一个水平行 Row，在其中放置一个列 Column。scale=2 表示该列占用的空间比例
            # 解释：在该列中创建一个 gr.Chatbot 对象，用于显示用户和机器人的对话记录
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
            # 功能：在同一行中添加另一个列，用于放置搜索结果显示和用户输入
            # gr.HTML：用于显示 "Search" 字样的 HTML
            # search_field：创建一个多行的 gr.Textbox，用于显示或输入搜索条件
            # user_input：创建另一个 gr.Textbox，用于让用户输入查询内容
            with gr.Column(scale=2):
                gr.HTML("""<h4>Search</h4>""")
                search_field = gr.Textbox(show_label=False, placeholder="search...", lines=8)
                user_input = gr.Textbox(show_label=False, placeholder="输入框...", lines=2)
                # 功能：创建按钮行，并定义两个按钮
                # submitBtn：提交按钮，用于将用户输入提交给聊天机器人
                # emptyBtn：清空按钮，用于重置对话和搜索输入字段
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                    emptyBtn = gr.Button("清空")
        # 功能：添加一个新的行和列，用于显示搜索返回的酒店信息
        # gr.HTML：显示 "Return" 标题
        # return_field：创建一个 gr.Dataframe，用于表格显示返回的酒店数据
        with gr.Row():
            with gr.Column():
                gr.HTML("""<h4>Return</h4>""")
                return_field = gr.Dataframe()

        # 功能：创建一个 gr.State 对象 context，用于保存和维护对话上下文状态
        # 解释：gr.State([]) 初始化为空列表，可以在多轮对话中保留上下文信息
        context = gr.State([])

        # 功能：绑定 submitBtn 按钮的点击事件，使其触发 chat 函数
        # 解释：每次点击 submitBtn 后，chat 函数被调用，接收 user_input、chatbot、context、search_field 和 return_field 作为输入参数，并更新这些输出字段
        submitBtn.click(chat, [user_input, chatbot, context, search_field, return_field],
                        [user_input, chatbot, context, search_field, return_field])

        # 功能：绑定 emptyBtn 按钮的点击事件，使其触发 reset_state 函数
        # 解释：点击 emptyBtn 时，调用 reset_state 函数，将 chatbot、context、user_input、search_field 和 return_field 重置为初始状态
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field, return_field])

    # 功能：设置并启动 Gradio 应用
    # queue()：启用队列模式，以便处理多个请求
    # launch()：启动应用并配置服务器参数
    # share=False：不共享外部访问
    # server_name 和 server_port：配置应用的访问地址和端口号
    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=6006, inbrowser=True)


if __name__ == "__main__":
    main()



