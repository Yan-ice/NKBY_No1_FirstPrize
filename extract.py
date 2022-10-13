from itertools import islice
import jieba

def extract():
    file1 = open('data/80w.txt', 'r', encoding='utf-8')
    # 读取文件的前100行
    content = list(islice(file1, 100))
    print(content)
    for i in range(0, 100):
        part = splitLine(content[i])
        print(part)
        print("\n")

# 通过\t将读取到的每一行分为标签和分词过后的句子
def splitLine(sentence):
    s = sentence.split("\t")
    return int(s[1]), jieba.lcut(s[2])

if __name__ == '__main__':
    extract()
