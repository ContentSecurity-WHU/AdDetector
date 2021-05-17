import requests
from bs4 import BeautifulSoup
from ad_detector.ad_detector import AdDetector
from ad_detector import config
import re

print('如是复制黏贴的网址，建议手动输入最后一个字母避免py3转义和打开网址')
URL = input("请输入您要鉴定的网页：")
#URL = r"https://www.zhihu.com/question/459215291"

print('您输入的URL为：',URL)
hd={'user-agent': "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)"}
r = requests.get(URL,headers=hd)
if r.status_code != 200 :
    print('出现',r.status_code,'错误,终止程序')
    exit()

#获取所有文本
r.encoding = r.apparent_encoding
demo = r.text
soup = BeautifulSoup(demo, 'html.parser')
get_text = re.sub(r'[\？\。\！\?\!\\\u200b]', '|||', soup.get_text())#去除句子结束标点
all_text = get_text.split('|||')

#有价值文本过滤
worth_texts = []
for text in all_text:
    if len(text) > 4:#短文本过滤
        is_chinese = 0 #非中文过滤
        for ch in text:
            if u'\u4e00' <= ch <= u'\u9fff':
                is_chinese = 1
        if is_chinese == 1:
            worth_texts.append(text)
text_num = len(worth_texts)

#判断是否为广告
Ad_text = []
Ad_Detect = AdDetector(
    config.training.model_path,
    config.path.stop_words,
    config.utils.word2idx_path,
    config.model.content_size
)
for text in worth_texts:
    if  Ad_Detect.is_ad(text) is True:
        Ad_text.append(text)

f = open("test.txt", "w")
f.write(str(Ad_text))
f.close()

print('句子总数为',text_num,'   广告句数为',len(Ad_text))
print("占比%.2f"%(len(Ad_text)/len(worth_texts)*100),'%')