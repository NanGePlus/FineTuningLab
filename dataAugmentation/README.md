# 1、数据预处理简介        
轻量化微调流程图                         
<img src="./001.png" alt="" width="900" />                 
## 1.1 业务数据增强             
根据业务数据特性进行数据增强，使用的数据为开源的预定酒店场景下的酒店数据库                                        
主要内容:基于原始的数据利用大模型进行数据增强                       
(1)对酒店设施的描述进行口语化重写              
(2)补充一定比例的多轮问答和结束语对话                
(3)补充按酒店名(简称)、价格上限查询等对话             

## 1.2 制作数据集及拆分训练集、验证集、测试集
根据增强后的业务数据进行数据集整理，按照规则处理成特定的数据组织格式                                          
最后按照8:1:1拆分训练集、验证集、测试集                


# 2、项目测试
打开命令行终端，执行 `cd weaviate` 指令进入到weaviate文件夹                
执行如下命令安装依赖包                                                       
`pip install -r requirements.txt`                           
每个软件包后面都指定了本次视频测试中固定的版本号            
**注意:** 建议先使用要求的对应版本进行本项目测试，避免因版本升级造成的代码不兼容。测试通过后，可进行升级测试                    

## 2.1 数据增强处理1       
enhanceBasic目录中脚本包含以下功能:                               
(1)将原始数据中设施相关的说法，改为更口语化的表达               
(2)在原始数据中，补充针对上文已推荐的酒店的问答，如：“XXX多少钱”，“XXX地址在哪”           
(3)在原始数据中，补充针对上文已推荐的酒店的比较型问答，如：“哪个更便宜”              
(4)在原始数据中，补充结束语，如：“就住XXX吧”“祝您入住愉快”           
打开命令行终端，运行如下命令进行测试                             
`cd enhanceBasic`                                              
`python enhance.py`                             

## 2.2 数据增强处理2
enhanceMore目录中脚本包含以下功能                
(1)限制价格上/下界的查询                
(2)限制价格区间的查询                 
(3)组合价格与其他条件的查询                 
(4)按酒店名称查询（包括用户不说酒店全名的情况）                    
打开命令行终端，运行如下命令进行测试                        
`cd enhanceMore`                              
`python generate_by_filter_search.py`                                    
`python generate_by_hotel_name.py`                                                     
                    
## 2.3 制作数据集               
打开命令行终端，运行如下命令进行测试                       
`cd dataset`                                     
`python combine_and_split.py`                                        

              
              