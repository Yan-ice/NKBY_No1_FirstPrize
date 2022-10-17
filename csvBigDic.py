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

# 训练集样本数
TRAIN_DATA_SIZE = 50000

# 测试集样本数
TEST_DATA_SIZE = 10000


# 关键词个数（取词频最高的多少个词作为特征？）
KEYWORD_SIZE = 1000

#
# 以下是 决策树参数调整。这些参数太大会导致过拟合，太小会导致特征不足。
#

# 决策树最大深度
MAX_DEPTH = 4

# 决策树需要多少个分支才会区分
MIN_SAMPLES_SPLIT = 5

# 节点后包含样本数量小于该值就不会区分。如果是int，则是具体数目，如果是float，则是百分比。
MIN_SAMPLE_LEAF = 5

# 可以填写random或best。一般来说，特征多的情况下random效果较好。
METHOD = "random"


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
from sklearn.tree import DecisionTreeClassifier

##method1
clf = GaussianNB()
clf.fit(X_train, y_train)
prediction1 = clf.predict(X_test)
# print("prediction:", prediction)
# print("actual:", y_test)

##method2
model = DecisionTreeClassifier(splitter=METHOD, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLE_LEAF, min_samples_split=MIN_SAMPLES_SPLIT)
model.fit(X_train,y_train)
prediction2=model.predict(X_test)


##正确性检测
##accuracy_method1
accuracy_method1 = accuracy_score(prediction1, y_test)
print("Gaussian accuracy:", accuracy_method1)
##accuracy_method2
print("Decision Tree accuracy:",model.score(X_test,y_test))#对训练情况进行打分