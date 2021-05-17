import jieba
from wordcloud import WordCloud
from ad_detector import config
import re

def remove_digits(input_str):
    punc = u'0123456789.'
    output_str = re.sub(r'[{}]+'.format(punc), '', input_str)
    return output_str

def move_stopwords(sentence_list, stopwords_list):
    # 去停用词
    out_list = []
    for word in sentence_list:
        if word not in stopwords_list:
            #if not remove_digits(word):
            #    continue
            if word != '\t':
                out_list.append(word)
    return out_list

words = open('test.txt',encoding='gbk').read()#打开歌词文件，获取到歌词
new_words = ' '.join(jieba.cut(words))#使用jieba.cut分词，然后把分好的词变成一个字符串，每个词用空格隔开
with open(config.path.stop_words, 'r', encoding='utf8') as f:
    stop_words = f.read().split('\n')
new_words = str(move_stopwords(new_words.split(),stop_words))
new_words = re.sub(r'[\'\"\“\”\\u200b\,]', ' ', new_words)

wordcloud = WordCloud(width=1000, #图片的宽度
                      height=860,  #高度
                      margin=2, #边距
                      background_color='black',#指定背景颜色
                      font_path='C:\Windows\Fonts\Sitka Banner\msyh.ttc'#指定字体文件，要有这个字体文件，自己随便想用什么字体，就下载一个，然后指定路径就ok了；刚才的字体适用于英文字体，用在中文字体上会报错，所以换了一个中文字体
                      )
wordcloud.generate(new_words) #分词
wordcloud.to_file('test.jpg')#保存到图片
