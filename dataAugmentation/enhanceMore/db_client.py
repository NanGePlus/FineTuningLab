# 功能：导入 dotenv 模块并调用 load_dotenv 函数，加载指定的环境变量文件 api_keys.env
# 解释：load_dotenv("api_keys.env") 会从 api_keys.env 文件中读取键值对并将其加载到环境变量中，使得程序可以使用 os.getenv 等方法来访问这些变量
from dotenv import load_dotenv
load_dotenv("api_keys.env")
# 功能：导入 Python 的 os 标准库，用于与操作系统交互
# 解释：os 模块常用于获取环境变量、处理文件路径以及执行其他系统相关操作
import os
# 功能：导入 Python 的 re 模块，用于正则表达式操作
# 解释：re 模块允许进行字符串的模式匹配、替换等操作
import re
# 功能：导入 Python 的 json 模块，用于处理 JSON 格式数据
# 解释：json 模块可以将 JSON 数据解析为 Python 字典或列表，也可以将 Python 对象转换为 JSON 格式字符串，通常用于与 API 的数据交互或保存配置数据
import json
# 功能：导入 requests 模块，用于执行 HTTP 请求
# 解释：requests 是一个第三方库，用于发送 HTTP 请求、处理响应并与 API 服务器交互
import requests
# 功能：导入 weaviate 模块，用于与 Weaviate 向量数据库进行交互
# 解释：weaviate 是一个开源的向量数据库，特别适合于大规模语义搜索和机器学习应用程序。通过导入 weaviate 模块，代码可以创建客户端并执行数据查询、插入、删除等操作
import weaviate
# 功能：从 tqdm 库中导入 tqdm，用于显示进度条
# 解释：tqdm 是一个快速、可扩展的 Python 进度条库，用于显示循环处理进度
from tqdm import tqdm



# 使用 RRF 算法来合并多个排序列表，将排名越靠前的文档赋予更高的权重，然后按累积分数对文档进行排序，最终输出合并后的文档列表
# 这种方法可用于场景中，例如推荐系统和搜索结果的融合
# 定义 rrf 函数，用于合并多个排序列表（即 rankings）中的结果
# rankings 是一个列表，其中每个元素是一个排序列表（即列表的列表）
# k 是 RRF 算法的常数，默认为 60，用于控制排名分数的影响
def rrf(rankings, k=60):
    # 检查 rankings 是否为列表类型。如果不是，则抛出 ValueError 错误，以确保函数输入的类型正确
    if not isinstance(rankings, list):
        raise ValueError("Rankings should be a list.")
    # 初始化一个空字典 scores，用于存储每个文档的累积分数和该文档的原始数据
    scores = dict()
    # 遍历 rankings 中的每个排序列表 ranking
    # 检查 ranking 是否为空，如果为空，则跳过该列表，继续处理下一个 ranking
    for ranking in rankings:
        if not ranking:  # 如果ranking为空，跳过它
            continue
        # 遍历 ranking 中的每个文档 doc，并通过 enumerate 获取其索引 i
        # 确保 doc 是一个字典（应包含 hotel_id 和其他相关信息）；如果不是，则抛出 ValueError 错误
        for i, doc in enumerate(ranking):
            if not isinstance(doc, dict):
                raise ValueError("Each item should be dict type.")
            # 尝试从 doc 中提取 hotel_id（文档的唯一标识符），以便对每个文档进行标识
            # 如果 hotel_id 不存在，则抛出 ValueError 错误，确保每个 doc 包含 hotel_id 这一关键信息
            doc_id = doc.get('hotel_id', None)
            if doc_id is None:
                raise ValueError("Each item should have 'hotel_id' key.")
            # 如果 scores 字典中还没有记录 doc_id，则将其初始化为 (0, doc)，其中 0 是累积的初始分数，doc 是原始文档
            if doc_id not in scores:
                scores[doc_id] = (0, doc)
            # 使用 RRF 算法的公式更新 doc_id 的累积分数：1 / (k + i)
            # k + i 是为了使排名越靠前的文档获得越大的加权分数
            # 将更新后的分数和原始文档信息一起存储在 scores 字典中
            scores[doc_id] = (scores[doc_id][0] + 1 / (k + i), doc)

    # 对 scores 字典中的值（包含累积分数和文档数据的元组）进行排序，按照累积分数降序排列
    sorted_scores = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    # 返回排序后的文档列表，其中每个元素为 doc（即 item[1]），不包含分数
    return [item[1] for item in sorted_scores]


# 定义一个名为 HotelDB 的类，用于管理酒店数据的数据库接口，尤其是与 Weaviate 数据库的交互
class HotelDB():
    # 初始化方法 __init__，用于设置 Weaviate 数据库连接
    # def __init__(self, url="http://localhost:8080"):
    def __init__(self, url="http://139.224.72.218:8080"):
        # 使用 weaviate.Client 连接到指定的 URL 地址，并在请求头中添加X-Openai-Baseurl、X-OpenAI-Api-Key
        # os.getenv("OPENAI_API_BASE") 从环境变量中加载 OpenAI URL地址(代理方式).env 文件中配置并通过 load_dotenv() 加载
        # os.getenv("OPENAI_API_KEY") 从环境变量中加载 OpenAI API 密钥，确保该 API 密钥已在 .env 文件中配置并通过 load_dotenv() 加载
        self.client = weaviate.Client(url=url,
          additional_headers={
              "X-OpenAI-Api-Key":os.getenv("OPENAI_API_KEY"),
              "X-Openai-Baseurl": os.getenv("OPENAI_API_BASE")
          }
        )


    # 定义 insert 方法，用于将酒店数据插入 Weaviate 数据库
    def insert(self):
        # 删除 Weaviate 数据库中已有的 Hotel 类，以确保新的架构不会与旧的数据冲突
        self.client.schema.delete_class("Hotel")
        # 这是一个数据库的 schema 定义，用于创建一个名为 "Hotel" 的类（类似于关系型数据库中的表），包含多个属性（表中的列）。每个属性定义了字段名称、数据类型和一些特定的搜索配置
        # Hotel 类包括多个 properties 属性，如 hotel_id、name、type、address 等
        # vectorizer 和 moduleConfig 指定了如何处理文本数据的向量化，其中 model 是 OpenAI 的 ada 模型
        # properties 列表中定义了每个字段的属性，包括 dataType（数据类型）、description（描述）、name（字段名称）、indexSearchable（是否可搜索）等
        # "tokenization": "whitespace" 表示将文本按空格分词，方便搜索
        # moduleConfig 的 "skip": True 表示跳过此字段的向量化处理
        schema = {
          "classes": [
            {
              "class": "Hotel",
              "description": "hotel info",
              "properties": [
                {
                  "dataType": ["number"],
                  "description": "id of hotel",
                  "name": "hotel_id"
                },
                {
                  "dataType": ["text"],
                  "description": "name of hotel",
                  "name": "_name", #分词过用于搜索的
                  "indexSearchable": True,
                  "tokenization": "whitespace",
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "type of hotel",
                  "name": "name",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "type of hotel",
                  "name": "type",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "address of hotel",
                  "name": "_address", #分词过用于搜索的
                  "indexSearchable": True,
                  "tokenization": "whitespace",
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "type of hotel",
                  "name": "address",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "nearby subway",
                  "name": "subway",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["text"],
                  "description": "phone of hotel",
                  "name": "phone",
                  "indexSearchable": False,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": True }
                  },
                },
                {
                  "dataType": ["number"],
                  "description": "price of hotel",
                  "name": "price"
                },
                {
                  "dataType": ["number"],
                  "description": "rating of hotel",
                  "name": "rating"
                },
                {
                  "dataType": ["text"],
                  "description": "facilities provided",
                  "name": "facilities",
                  "indexSearchable": True,
                  "moduleConfig": {
                    "text2vec-openai": { "skip": False }
                  },
                },
              ],
              "vectorizer": "text2vec-openai",
              "moduleConfig": {
                "text2vec-openai": {
                  "vectorizeClassName": False,
                  "model": "ada",
                  "modelVersion": "002",
                  "type": "text"
                },
              },
            }
          ]
        }
        # 创建数据库
        self.client.schema.create(schema)
        # 打开并读取 hotel.json 文件，将 JSON 数据加载为 hotels 列表，每个元素代表一个酒店信息的字典
        with open('hotel.json', 'r') as f:
            hotels = json.load(f)
        # 配置批量插入的设置，其中 batch_size=4 表示每次批量插入 4 条记录
        # dynamic=True 启用动态批处理功能，以确保插入效率
        self.client.batch.configure(batch_size=4, dynamic=True)
        # 使用 tqdm 进度条遍历每个酒店信息
        # 每个酒店数据通过 add_data_object 添加到 Weaviate 批处理
        # 使用 generate_uuid5 为每个酒店生成一个唯一的 UUID，基于酒店信息和类名 "Hotel"
        for hotel in tqdm(hotels):
            self.client.batch.add_data_object(
                data_object=hotel,
                class_name="Hotel",
                uuid=weaviate.util.generate_uuid5(hotel, "Hotel")
            )
        # 执行 flush 操作，确保所有批量添加的数据对象都提交到 Weaviate 数据库
        self.client.batch.flush()


    # 这个方法接受一个 dsl 字典（用作查询条件），name（默认是 "Hotel"，即 Weaviate 中的类名），以及一个 limit 参数，表示返回结果的数量
    def search(self, dsl, name="Hotel", limit=1):
        # 1、预处理 DSL 并初始化候选列表
        # 过滤掉 dsl 中值为 None 的键值对，避免查询时处理无效条件
        dsl = {k: v for k, v in dsl.items() if v is not None}
        # 增加搜索限制，多查找10条数据，以便从中筛选出最符合条件的 limit 条结果
        _limit = limit + 10
        # 初始化候选结果列表
        candidates = []
        # 定义了查询中希望返回的字段
        output_fields = ["hotel_id","name","type","address","phone","subway","facilities","price","rating"]

        # 2、组装过滤条件
        # filters 初始化了一个过滤条件，默认过滤掉价格为 0 的酒店
        filters = [{
            "path": ["price"],
            "operator": "GreaterThan",
            "valueNumber": 0,
        }]
        # keys 是 dsl 中的关键字段，包含 类型、价格和评分范围
        keys = [
            "type",
            "price_range_lower",
            "price_range_upper",
            "rating_range_lower",
            "rating_range_upper",
        ]
        # 根据 dsl 中的条件，将附加更多的过滤条件
        if any(key in dsl for key in keys):
            # 根据 dsl 中的具体条件，将 类型、价格范围、评分范围等过滤条件添加到 filters 列表中
            if "type" in dsl:
                filters.append(
                    {
                        "path": ["type"],
                        "operator": "Equal",
                        "valueString": dsl["type"],
                    }
                )
            if "price_range_lower" in dsl:
                filters.append(
                    {
                        "path": ["price"],
                        "operator": "GreaterThan",
                        "valueNumber": dsl["price_range_lower"],
                    }
                )
            if "price_range_upper" in dsl:
                filters.append(
                    {
                        "path": ["price"],
                        "operator": "LessThan",
                        "valueNumber": dsl["price_range_upper"],
                    }
                )
            if "rating_range_lower" in dsl:
                filters.append(
                    {
                        "path": ["rating"],
                        "operator": "GreaterThan",
                        "valueNumber": dsl["rating_range_lower"],
                    }
                )
            if "rating_range_upper" in dsl:
                filters.append(
                    {
                        "path": ["rating"],
                        "operator": "LessThan",
                        "valueNumber": dsl["rating_range_upper"],
                    }
                )

        # 3、构建最终过滤条件
        # 如果 filters 只有一个条件，则直接使用它
        if (len(filters)) == 1:
            filters = filters[0]
        # 如果有多个条件，则使用 And 操作符连接所有过滤条件
        elif len(filters) > 1:
            filters = {"operator": "And", "operands": filters}

        # 4、进行向量搜索
        # 如果 dsl 中有 facilities 条件，则构建一个向量搜索
        if "facilities" in dsl:
            # 通过 self.client.query.get() 创建一个查询对象，name 是数据类的名称 "Hotel"
            # output_fields 是一个列表，指定了查询中希望返回的字段，例如酒店名称、地址、价格、评分等
            query = self.client.query.get(name, output_fields)
            # 使用 .with_near_text 方法进行基于文本的相似度匹配
            query = query.with_near_text(
                # 这是一个字典，其中 concepts 键包含了一段字符串，描述了查询文本
                {"concepts": [f'酒店提供:{";".join(dsl["facilities"])}']}
            )
            # 这里将 filters 作为查询的条件
            # filters 是由前面代码中定义的条件集合，包含价格范围、评分范围等约束，用于筛选满足这些条件的酒店记录
            if filters:
                query = query.with_where(filters)
            # 设置查询返回的最大结果数量
            # _limit 是目标返回结果数目加上一些额外数量，这样可以在候选列表中返回更多结果，以便后续排序和过滤
            query = query.with_limit(_limit)
            # 执行构建的查询，并将结果存储到 result 中
            result = query.do()
            # print('1、向量搜索的结果是:',result)
            # 查询的结果使用 rrf 方法进行融合排名
            # result["data"]["Get"][name] 获取查询返回的酒店信息列表，name 是类名"Hotel"
            # rrf([candidates, result["data"]["Get"][name]]) 使用 rrf（融合排名算法）方法对查询结果进行融合排名
            candidates = rrf([candidates, result["data"]["Get"][name]])

        # 5、进行关键字搜索
        # 检查 dsl 字典中是否包含 name 键，即用户是否在查询条件中提供了酒店名称的相关信息
        if "name" in dsl:
            # 使用正则表达式提取 dsl["name"] 中的有效字符
            # re.findall(r"[\dA-Za-z\-]+|\w", dsl["name"]) 会找到 dsl["name"] 中的所有符合模式的子串。该正则表达式匹配数字 (\d)、大小写字母 (A-Za-z)、连字符 (-) 以及任何单独的 Unicode 字符（\w）
            # join() 将所有提取的子串用空格拼接成一个单独的字符串，最终得到 text。这是为了确保查询字符串没有特殊字符干扰
            text = " ".join(re.findall(r"[\dA-Za-z\-]+|\w", dsl["name"]))
            # 通过 self.client.query.get() 创建一个查询对象，name 是数据的类名（"Hotel"），output_fields 是一个列表，指定了查询返回的字段
            query = self.client.query.get(name, output_fields)
            # .with_bm25() 用于在 Weaviate 数据库中基于 BM25 算法进行文本匹配
            # query=text 表示将前面拼接的字符串 text 作为查询内容
            # properties=["_name"] 表示在 _name 属性（即酒店名称字段）上执行 BM25 检索操作，以找到名称与查询文本相似的记录
            # BM25 是一种常用的文本检索算法，特别适合用于短文本和模糊匹配
            query = query.with_bm25(query=text, properties=["_name"])
            # 这里将 filters 作为查询的条件
            # 如果定义了 filters，则将其应用到查询中，以进一步筛选结果。如过滤条件可能包含价格、评分等限制
            if filters:
                query = query.with_where(filters)
            # 设置查询返回的最大结果数量
            # _limit 是目标返回结果数目加上一些额外数量，这样可以在候选列表中返回更多结果，以便后续排序和过滤
            query = query.with_limit(_limit)
            # 执行构建的查询，并将结果存储到 result 中
            result = query.do()
            # 查询的结果使用 rrf 方法进行融合排名
            # result["data"]["Get"][name] 获取查询返回的酒店信息列表，name 是类名 "Hotel"
            # rrf([candidates, result["data"]["Get"][name]]) 使用 rrf（融合排名算法）方法对查询结果进行融合排名
            candidates = rrf([candidates, result["data"]["Get"][name]])
        # 检查 dsl 字典中是否包含 address 键，即用户是否在查询条件中提供了酒店地址的相关信息
        if "address" in dsl:
            # 使用正则表达式提取 dsl["address"] 中的有效字符，将其用于查询
            # re.findall(r"[\dA-Za-z\-]+|\w", dsl["address"]) 会在 dsl["address"] 中找到所有符合模式的子串。正则表达式匹配数字 (\d)、大小写字母 (A-Za-z)、连字符 (-)，以及任意 Unicode 字符（\w）
            # join() 将所有提取的字符用空格拼接成一个字符串 text，用于后续查询
            text = " ".join(re.findall(r"[\dA-Za-z\-]+|\w", dsl["address"]))
            # 通过 self.client.query.get() 创建一个查询对象，name 是数据的类名（"Hotel"），output_fields 是一个列表，指定了查询返回的字段
            query = self.client.query.get(name, output_fields)
            # .with_bm25() 用于在 Weaviate 数据库中基于 BM25 算法进行文本匹配
            # query=text 表示将前面拼接的字符串 text 作为查询内容
            # properties=["_address"] 表示在 _address 属性（即酒店名称字段）上执行 BM25 检索操作，以找到名称与查询文本相似的记录
            # BM25 是一种常用的文本检索算法，特别适合用于短文本和模糊匹配
            query = query.with_bm25(query=text, properties=["_address"])
            # 这里将 filters 作为查询的条件
            # 如果定义了 filters，则将其应用到查询中，以进一步筛选结果。例如，过滤条件可能包含价格、评分等限制
            if filters:
                query = query.with_where(filters)
            # 设置查询返回的最大结果数量
            # _limit 是目标返回结果数目加上一些额外数量，这样可以在候选列表中返回更多结果，以便后续排序和过滤
            query = query.with_limit(_limit)
            # 执行构建的查询，并将结果存储到 result 中
            result = query.do()
            # 查询的结果使用 rrf 方法进行融合排名
            # result["data"]["Get"][name] 获取查询返回的酒店信息列表，name 是类名 "Hotel"
            # rrf([candidates, result["data"]["Get"][name]]) 使用 rrf（融合排名算法）方法对查询结果进行融合排名
            candidates = rrf([candidates, result["data"]["Get"][name]])

        # 6、进行条件搜索
        # 如果 candidates 为空，则执行一次只基于过滤条件的查询
        if not candidates:
            # 通过 self.client.query.get() 创建一个查询对象，name 是数据的类名（"Hotel"），output_fields 是一个列表，指定了查询返回的字段
            query = self.client.query.get(name, output_fields)
            # 这里将 filters 作为查询的条件
            # 如果定义了 filters，则将其应用到查询中，以进一步筛选结果。例如，过滤条件可能包含价格、评分等限制
            if filters:
                query = query.with_where(filters)
            # 设置查询返回的最大结果数量
            # _limit 是目标返回结果数目加上一些额外数量，这样可以在候选列表中返回更多结果，以便后续排序和过滤
            query = query.with_limit(_limit)
            # 执行构建的查询，并将结果存储到 result 中
            result = query.do()
            # 从 result 中提取查询结果，使用 result["data"]["Get"][name] 获取符合条件的酒店候选项的列表
            # 由于 candidates 原本为空，这里直接将查询到的结果赋值给 candidates
            candidates = result["data"]["Get"][name]

        # 7、排序
        # 检查 dsl 中是否包含 sort.slot 这一键，sort.slot 指定了排序的字段
        # 如果存在，意味着用户希望按特定字段对 candidates 列表进行排序
        if "sort.slot" in dsl:
            # 检查 dsl 中的 sort.ordering 键是否为 "descend"
            # sort.ordering 是排序的方向参数，如果值为 "descend"，说明用户希望降序排列
            if dsl["sort.ordering"] == "descend":
                # 如果 sort.ordering 为 "descend"，使用 Python 的 sorted() 函数对 candidates 列表按降序排序
                # key=lambda x: x[dsl["sort.slot"]] 指定排序的字段，其中 dsl["sort.slot"] 提供了要排序的字段名称，例如 "price"、"rating" 等
                # reverse=True 表示按降序排列
                candidates = sorted(
                    candidates, key=lambda x: x[dsl["sort.slot"]], reverse=True
                )
            # 如果 sort.ordering 不为 "descend"（可能是 "ascend"），执行这个 else 分支
            # 使用 sorted() 按升序排序，key=lambda x: x[dsl["sort.slot"]] 指定排序的字段
            # 由于 reverse 参数未设置为 True，默认按升序排序
            else:
                candidates = sorted(
                    candidates, key=lambda x: x[dsl["sort.slot"]]
                )

        # 8、筛选匹配结果并返回
        # 筛选和限制返回的 candidates 数量
        # 检查 dsl 字典中是否包含键 "name"，表示用户在搜索时可能指定了酒店的名称或其关键字
        if "name" in dsl:
          # 创建一个空列表 final 来存储符合条件的候选项
          final = []
          # 遍历 candidates 中的每个元素 r，逐一检查其 name 字段
          # 使用 all(char in r['name'] for char in dsl['name']) 条件筛选出符合条件的候选项
          # dsl['name'] 表示用户输入的名称字符串
          # r['name'] 是候选酒店的名称
          # 通过 all() 确保 dsl['name'] 中的每个字符都在 r['name'] 中（字符匹配）
          # 如果 r['name'] 符合条件，将其添加到 final 列表中
          for r in candidates:
              if all(char in r['name'] for char in dsl['name']):
                  final.append(r)
          # 将 candidates 更新为 final，即保留符合名称筛选条件的候选项
          candidates = final

        # 检查 candidates 的长度是否超过 limit，即用户指定的最大返回数量
        # 如果超过，则对 candidates 进行截断，只保留前 limit 条
        if len(candidates) > limit:
            candidates = candidates[:limit]

        # 返回处理过的 candidates 列表，即最终筛选和排序后的候选项
        return candidates


if __name__ == "__main__":
    db = HotelDB()

    # # 1、写入数据
    # db.insert()
    # print('完成数据入库')

    # 2、查询数据
    result = db.search({'facilities':['吹风机']}, limit=3)
    # 排序
    # result = db.search({'facilities':['吹风机'],'sort.slot':'rating','sort.ordering':'descend'}, limit=3)
    # 指定酒店
    # result = db.search({'name':'北京京仪大酒店','facilities':['SPA'],'sort.slot':'rating','sort.ordering':'descend'}, limit=3)

    print(json.dumps(result,ensure_ascii=False))
