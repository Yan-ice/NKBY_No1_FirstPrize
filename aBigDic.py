from itertools import islice
import jieba
def creatCounts():
    file1 = open('data/80w.txt', 'r', encoding='utf-8')
    # 读取文件的前100行
    words = dict()
    N=5000
    content = list(islice(file1, N))
    count=0
    with open('data/stopWord.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for i in range(0, N):
        part = splitLine(content[i])
        sb=len(part)
        for j in range(0, sb):
            if part[j] not in words and part[j] not in lines:
                words[part[j]] = count
                count=count+1
    del words['\n']
    del words[' ']


    counts = [0 for _ in range(count)]
    for i in range(0, N):
        part = splitLine(content[i])
        sb=len(part)
        for j in range(0, sb):
            if part[j] in words:
                counts[words[part[j]]]=counts[words[part[j]]]+1
    return counts,words,content
def loveU3000(lovetimes):
    lovewords=dict()
    (a,b,content)=creatCounts()
    count=0;
    for i in range(0,lovetimes):
        c=a.index(max(a))
        lovewords[return_key(c,b)]=count
        a[c] = 0
        count=count+1
    return lovewords

def return_key(val,dictionary):
    for key, value in dictionary.items():
        if value == val:
            return key
    return ('Key Not Found')




# 通过\t将读取到的每一行分为标签和分词过后的句子
def splitLine(sentence):
    s = sentence.split("\t")
    return  jieba.lcut(s[2])
def creatMatrix(lovetimes):

    (a, b, content)=creatCounts()
    lovewords=loveU3000(lovetimes)
    print(lovewords)

    paragraph=[0 for _ in range(len(content))]
    for i in range(0, len(content)):
        sentences = [0 for _ in range(lovetimes)]
        part = splitLine(content[i])
        sb = len(part)
        #print(part)
        for j in range(0, lovetimes):

            if return_key(j,lovewords) in part:
                sentences[j] = 1
            paragraph[i]=sentences
    #print(paragraph)
    return paragraph





if __name__ == '__main__':
    lovetimes=1000
    creatCounts()
    loveU3000(lovetimes)
    creatMatrix(lovetimes)