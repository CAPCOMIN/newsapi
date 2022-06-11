# -*- coding: utf-8 -*-
'''
    Author: x
    Desc：
        代码11-2 不同语料库下的新闻关键词抽取-基于TFIDF
'''
import logging
import time
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from collections import defaultdict
from sympy import content
import pymysql
import jieba.analyse
from sklearn.externals import joblib
from gensim.models import Word2Vec
import jieba.analyse
import jieba.posseg as pseg
from Spider.settings import DB_HOST, DB_USER, DB_PASSWD, DB_NAME, DB_PORT

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)-7s - %(message)s')

# 2. 初始化handler,并配置formater
log_file_handler = TimedRotatingFileHandler(filename="Recommend/analysis/kwg.log",
                                            when="S", interval=5,
                                            backupCount=20)
log_file_handler.setFormatter(formatter)

# 3. 向logger对象中添加handler
logger.addHandler(log_file_handler)


class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, start, end, weight=1):
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self, iteration=10):
        """
        textrank算法的实现
        :param iteration: 迭代次数
        :return: dict type
        """
        # print("begin to run rank func...")
        ws = defaultdict(float)
        outSum = defaultdict(float)  # 节点出度之和
        wsdef = 1.0 / (len(self.graph) or 1.0)  # 节点权值初始定义
        for n, edge in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((i[2] for i in edge), 0.0)
        sorted_keys = sorted(self.graph.keys())
        for i in range(iteration):  # 迭代
            # print("iteration %d..." % i)
            for n in sorted_keys:
                s = 0
                # 遍历节点的每条边
                for edge in self.graph[n]:
                    s += edge[2] / outSum[edge[1]] * ws[edge[1]]
                ws[n] = (1 - self.d) + self.d * s  # 更新节点权值

        min_rank, max_rank = min(ws.values()), max(ws.values())
        # 归一化权值
        for n, w in ws.items():
            ws[n] = (w - min_rank / 10) / (max_rank - min_rank / 10)
        return ws


class TextRank:
    def __init__(self, data):
        """
        :param data: 输入的数据，字符串格式
        """
        self.data = data  # 字符串格式

    def extract_key_words(self, topK=20, window=4, iteration=200, allowPOS=('ns', 'n'), stopwords=True):
        """
        抽取关键词
        :param allowpos: 词性
        :param topK:   前K个关键词
        :param window: 窗口大小
        :param iteration: 迭代次数
        :param stopwords: 是否过滤停止词
        :return:
        """
        text = self.generate_word_list(allowPOS, stopwords)
        graph = UndirectWeightedGraph()
        # 定义共现词典
        cm = defaultdict(int)
        # 构建无向有权图
        for i in range(1, window):
            if i < len(text):
                text2 = text[i:]
                for w1, w2 in zip(text, text2):
                    cm[(w1, w2)] += 1
        for terms, w in cm.items():
            graph.add_edge(terms[0], terms[1], w)
        joblib.dump(graph, 'Recommend/data/graph')
        ws = graph.rank(iteration)
        return sorted(ws.items(), key=lambda x: x[1], reverse=True)[:topK]

    def generate_word_list(self, allowPOS, stopwords):
        """
        对输入的数据进行处理，得到分词及过滤后的词列表
        :param allowPOS: 允许留下的词性
        :param stopwords: 是否过滤停用词
        :return:
        """
        s = time.time()
        # thu_tokenizer = thulac.thulac(filt=True, rm_space=True, seg_only=False)
        # text = thu_tokenizer.cut(self.data)
        text = [(w.word, w.flag) for w in pseg.cut(self.data)]  # 词性标注
        word_list = []
        if stopwords:
            stop_words = [line.strip() for line in open(
                'Recommend/stopwords.txt', encoding='UTF-8').readlines()]
            stopwords_news = [line.strip() for line in open(
                'Recommend/stopwords.txt', encoding='UTF-8').readlines()]
            all_stopwords = set(stop_words + stopwords_news)
        # 词过滤
        if text:
            for t in text:
                if len(t[0]) < 2:
                    continue
                if len(t[0]) < 2 or t[1] not in allowPOS:
                    continue
                if stopwords:
                    # 停用词过滤
                    if t[0] in all_stopwords:
                        continue
                word_list.append(t[0])
        return word_list

    def bulid_w2c(self, ndim):
        """
        训练Wordvec模型
        :param ndim:  词向量维度
        :return:
        """
        print("train bulid_w2c...")
        data = [s[1] for s in self.sentence_list]
        model = Word2Vec(data, vector_size=ndim, window=3, epochs=10)
        model.save("model/w2v_model")
        return model

    @classmethod
    def cos_sim(cls, vec_a, vec_b):
        """
        计算两个向量的余弦相似度
        :param vec_a:
        :param vec_b:
        :return:
        """
        vector_a = np.mat(vec_a)
        vector_b = np.mat(vec_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        if cos == 'nan':
            print(cos)
        sim = 0.5 + 0.5 * cos
        return sim

# class SelectKeyWord:
#     def __init__(self, _type):
#         self._type = _type
#         self.db = self.connection()
#         self.cursor = self.db.cursor()
#         self.news_dict = self.loadData()
#         self.key_words = self.getKeyWords()
#
#     def connection(self):
#         '''
#             @Description：数据库连接
#             @:param host --> 数据库链接
#             @:param user --> 用户名
#             @:param password --> 密码
#             @:param database --> 数据库名
#             @:param port --> 端口号
#             @:param charset --> 编码
#         '''
#         db = pymysql.Connect(host=DB_HOST, user=DB_USER, password=DB_PASSWD, database=DB_NAME, port=DB_PORT,
#                              charset='utf8')
#         # db = pymysql.connections.Connection.connect(DB_HOST, DB_USER, DB_PASSWD, DB_NAME, DB_PORT, charset='utf8')
#         return db
#
#     def loadData(self):
#         '''
#             @Description：加载数据
#             @:param None
#         '''
#         news_dict = dict()
#         table = self.getDataFromDB()
#         # 遍历每一行
#         # for row in range(1, table.nrows):
#         for row in range(len(table)):
#             line = table[row]
#             news_id = int(line[0])
#             news_dict.setdefault(news_id, {})
#             news_dict[news_id]["tag"] = line[0]
#             news_dict[news_id]["title"] = line[2]
#             news_dict[news_id]["content"] = line[6]
#         return news_dict
#
#     def getDataFromDB(self):
#         '''
#             @Description：从数据库获取数据
#             @:param None
#         '''
#         logger.info("从数据库获取数据")
#         sql_s = "select * from news_api_newsdetail"
#         try:
#             self.cursor.execute(sql_s)
#             message = self.cursor.fetchall()
#         except:
#             self.db.rollback()
#         return message
#
#     # 调用结巴分词获取每篇文章的关键词
#     def getKeyWords(self):
#         '''
#             @Description：通过jieba提取关键词TF-IDF算法
#             @:param _type --> 选择提取内容（标题提取、标题+内容提取）
#         '''
#         news_key_words = list()
#         # 加载停用词表
#
#         stop_words_list = [line.strip() for line in open("Recommend/stopwords/stop_words.txt", 'r').readlines()]
#         for new_id in self.news_dict.keys():
#             if self._type == 1:
#                 # allowPOS 提取地名、名词、动名词、动词
#                 keywords = jieba.analyse.extract_tags(
#                     self.news_dict[new_id]["title"] + self.news_dict[new_id]["content"],
#                     topK=10,
#                     withWeight=False,
#                     allowPOS=('ns', 'n', 'vn', 'rn', 'nz')
#                 )
#                 news_key_words.append(str(new_id) + '\t' + ",".join(keywords))
#                 sql_i = 'update news_api_newsdetail set keywords=\"%s\" where news_id=%d' % (",".join(kws), new_id)
#                 try:
#                     self.cursor.execute(sql_i)
#                     self.db.commit()
#                 except Exception:
#                     logger.error("Error:KeyWords update Error!!")
#                     self.db.rollback()
#             elif self._type == 2:
#                 # cut_all :False 表示精确模式
#                 # keywords = jieba.cut(self.news_dict[new_id]["content"], cut_all=False)
#                 keywords = jieba.analyse.extract_tags(
#                     self.news_dict[new_id]["title"] + self.news_dict[new_id]["content"],
#                     topK=10,
#                     withWeight=False,
#                     allowPOS=('ns', 'n', 'vn', 'rn', 'nz')
#                 )
#                 kws = list()
#                 for kw in keywords:
#                     if kw not in stop_words_list and kw != " " and kw != " ":
#                         kws.append(kw)
#                         logger.info("keyword:{}".format(kw))
#                 news_key_words.append(str(new_id) + '\t' + ",".join(kws))
#                 sql_i = 'update news_api_newsdetail set keywords=\"%s\" where news_id=%d' % (",".join(kws), new_id)
#                 try:
#                     self.cursor.execute(sql_i)
#                     self.db.commit()
#                 except Exception:
#                     logger.error("Error:KeyWords update Error!!")
#                     self.db.rollback()
#             else:
#                 logger.error("请指定获取关键词的方法类型<1：TF-IDF 2：标题分词法>")
#         return news_key_words
#
#     def writeToFile(self):
#         '''
#             @Description：将关键词获取结果写入文件
#             @:param None
#         '''
#         fw = open("Recommend/data/keywords/1.txt", "w", encoding="utf-8")
#         fw.write("\n".join(self.key_words))
#         fw.close()


# def splitTxt():
#     source_dir = 'Recommend/data/keywords/1.txt'
#     target_dir = 'Recommend/data/keywords/split/'
#
#     # 计数器
#     flag = 0
#
#     # 文件名
#     name = 1
#
#     # 存放数据
#     dataList = []
#
#     with open(source_dir, 'rb') as f_source:
#         for line in f_source:
#             flag += 1
#             dataList.append(line)
#             if flag == 200:
#                 with open(target_dir + "pass_" + str(name) + ".txt", 'wb+') as f_target:
#                     for data in dataList:
#                         f_target.write(data)
#                 name += 1
#                 flag = 0
#                 dataList = []
#
#     # 处理最后一批行数少于200万行的
#     with open(target_dir + "pass_" + str(name) + ".txt", 'wb+') as f_target:
#         for data in dataList:
#             f_target.write(data)


def beginSelectKeyWord(_type):
    # skw = SelectKeyWord(_type=_type)
    # skw.writeToFile()
    # print("\n关键词获取完毕，数据写入路径 Recommend/data/keywords")
    db = pymysql.Connect(host=DB_HOST, user=DB_USER, password=DB_PASSWD, database=DB_NAME, port=DB_PORT,
                         charset='utf8')
    # 执行sql语句
    try:
        with db.cursor() as cursor:
            sql = "SELECT news_id,CONCAT(CONCAT(title,' '), mainpage) as info FROM news_api_newsdetail"
            cursor.execute(sql)
            result = cursor.fetchall()
    except pymysql.Error as e:
        print(e.args[0], e.args[1], sql)
    # finally:
    #     db.close()

    result = list(result)
    # result = [e[1] for e in result]

    for text in result:
        content = text[1]
        tr = TextRank(content)
        # key_sentences = tr.extract_key_sentences(topK=4, window=3, ndim=10)
        # print(key_sentences)
        key_words = jieba.analyse.textrank(
            content, topK=5, withWeight=True, allowPOS=('n', 'ni', 'nz'))
        # print(key_words)
        key_words = jieba.analyse.extract_tags(
            content, topK=5, withWeight=True, allowPOS=('n', 'ni', 'nz'))  # tf-idf
        # print(key_words)
        key_words = tr.extract_key_words(
            topK=10, window=5, iteration=50, stopwords=True, allowPOS=('n', 'ni', 'nz'))
        key_words = [e[0] for e in key_words]
        # print(text[0], end=' ')
        print(key_words)
        keywordString = ",".join(str(i) for i in key_words)
        sql_w = "UPDATE news_api_newsdetail SET keywords='%s' WHERE news_id=%s" % (
            keywordString, text[0])
        try:
            cur = db.cursor()
            cur.execute(sql_w)
            db.commit()
            # print('ok')
            # print(sql_w)
        except pymysql.Error as e:
            print(e.args[0], e.args[1], sql_w)
    db.close()
    logger.info("各个新闻的关键词获取完毕，数据写入数据库")
