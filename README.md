# 1、项目介绍
本项目案例为基于Qwen2.5-8B-Instruct大模型进行微调并应用到酒店推荐垂直领域的完整应用案例            
**完整的垂直应用案例闭环**                   
**项目源码剖析开源共享**                      
**详实的图文指导手册**        
**手把手全流程实操演示视频**         
### (1)什么是微调
不严谨的类比，通俗理解什么是大模型微调        
<img src="./001.png" alt="" width="900" />     
轻量化微调流程图               
<img src="./002.png" alt="" width="900" />                   
### (2)服务器硬件参数
租用的是AutoDL算力平台         
https://www.autodl.com/home               
GPU:RTX 4090D(24GB)        
CPU:16 vCPU Intel(R) Xeon(R) Platinum 8481C           
内存:80GB               
硬盘:系统盘30GB、数据盘50GB              
GPU驱动:550.107.02              
CUDA版本：<=12.4            


# 2、本系列分为如下五大部分
## 2.1 构建业务数据库
详情见weaviate文件夹                        

## 2.2 数据增强、制作数据集      
详情见dataAugmentation文件夹           

## 2.3 轻量化大模型微调
详情见qwen2文件夹                 

## 2.4 大模型应用测试    
详情见webDemo文件夹                 

## 2.5 封装大模型推理应用接口并测试           
详情见webApiDemo文件夹                  



         

              
              
