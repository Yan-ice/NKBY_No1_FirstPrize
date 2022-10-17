import jieba
import numpy as np
import pandas
import sklearn
import csv

##################
####   函数定义
##################

# 输入一行（label+短信），输出(label, [keywords] )
def split_line(sentence):
    list_ = sentence[2].split(' ')
    return int(sentence[0]), list_

def read_files(sample_size):
    content = []
    stop_word = []

    # 读取
    with open('data/sms_pub.csv', encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            if row[0] == 'label':
                continue
            content.append(split_line(row))
            sample_size = sample_size - 1
            if sample_size < 0:
                break
    for line in open('data/stopWord.txt', 'r', encoding='utf-8'):
        stop_word.append(line.strip())

    # 移除 stop word
    for lb in content:
        for word in lb[1]:
            if word.strip() in stop_word or word.strip() == '':
                lb[1].remove(word)
    return content


def get_freq_table(content, table_size):
    wordset = []
    for data in content:
        wordset.extend(data[1])
    ser = pandas.Series(wordset)
    return ser.value_counts().iloc[:table_size].keys()


def gen_datasets(top_freq_word, content):
    X_train = []
    y_train = []
    for data in content:
        code = []
        key_words = data[1]
        for check in top_freq_word:
            if check in key_words:
                code.append(1)
            else:
                code.append(0)
        X_train.append(code)
        y_train.append(data[0])
    return np.array(X_train), np.array(y_train)


##################
####   参数设置
##################

# 关键词个数（取词频最高的多少个词作为特征？）
KEYWORD_SIZE = 1000

# 训练集样本数
TRAIN_DATA_SIZE = 3000

# 测试集样本数
TEST_DATA_SIZE = 300


##################
####   预处理
##################

content = read_files(TRAIN_DATA_SIZE+TEST_DATA_SIZE)
word_table = get_freq_table(content, KEYWORD_SIZE)

# 获得训练集
X_train, y_train = gen_datasets(word_table, content[:TRAIN_DATA_SIZE])

# 获得测试集
X_test, y_test = gen_datasets(word_table, content[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+TEST_DATA_SIZE])


##################
####   模型训练
##################
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression#引入线性回归模型
##method1
clf = GaussianNB()
clf.fit(X_train, y_train)
prediction1 = clf.predict(X_test)
# print("prediction:", prediction)
# print("actual:", y_test)

##method2
model=LinearRegression()
model.fit(X_train,y_train)
prediction2=model.predict(X_test)


##正确性检测
##accuracy_method1
accuracy_method1 = accuracy_score(prediction1, y_test)
print("accuracy:", accuracy_method1)
##accuracy_method2
print("accuracy:",model.score(X_test,y_test))#对训练情况进行打分