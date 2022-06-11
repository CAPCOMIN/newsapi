'''
    Author: Zeng
    Desc：
        代码11-3 每个类型下新闻的相似度计算
'''
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

import jieba
from Recommend.tfidf import TFIDF
from Recommend.cosine import Cosine

import pymysql

from Spider.settings import DB_HOST, DB_USER, DB_PASSWD, DB_NAME, DB_PORT
sys.path.append(r'E:\大三下课程\信息内容安全\project\NewsRecommends\FinalProject\newsapi\Recommend')

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)-7s - %(message)s')

# 2. 初始化handler,并配置formater
log_file_handler = TimedRotatingFileHandler(filename="Recommend/analysis/ccg.log",
                                            when="S", interval=5,
                                            backupCount=20)
log_file_handler.setFormatter(formatter)

# 3. 向logger对象中添加handler
logger.addHandler(log_file_handler)


class Correlation:
    def __init__(self, file):
        # self.db = self.connection()
        # self.cursor = self.db.cursor()

        self.file = file
        self.titles, self.titleDict = self.loadData()
        self.title_words = self.split_word(self.titles)
        self.getCorrelation()

    # 加载数据
    def loadData(self):
        """
            @Description：加载关键词分析结果文件
            @:param None
        """
        db = pymysql.Connect(host=DB_HOST, user=DB_USER, password=DB_PASSWD, database=DB_NAME, port=DB_PORT,
                             charset='utf8')
        # 执行sql语句
        try:
            with db.cursor() as cursor:
                sql = "select title,news_id from news_api_newsdetail ORDER BY news_id asc"
                cursor.execute(sql)
                result = cursor.fetchall()
        except:
            print("读取分词数据过程中出现错误，错误行为：{}".format(sql))
            logger.error("Error：{}".format(sql))
        finally:
            db.close()
        resultDict = dict(result)
        # print(result)
        result = [e[0] for e in result]
        # print("mysql result")
        return result, resultDict

    def split_word(self, lines):
        '''
        分词
        '''
        with open("Recommend/stopwords.txt", encoding="utf-8") as f:
            stopwords = f.read().split("\n")
        words_list = []
        for line in lines:
            words = [word for word in jieba.cut(line.strip().replace("\n", "").replace(
                "\r", "").replace("\ue40c", "")) if word not in stopwords]
            words_list.append(" ".join(words))
        return words_list

    def getCorrelation(self):
        '''
            @Description：计算相关度，并写入数据库
            @:param None
        '''
        # news_cor_list = list()
        # for newid1 in self.news_tags.keys():
        #     id1_tags = set(self.news_tags[newid1].split(","))
        #     for newid2 in self.news_tags.keys():
        #         id2_tags = set(self.news_tags[newid2].split(","))
        #         if newid1 != newid2:
        #             # print(newid1 + "\t" + newid2 + "\t" + str(id1_tags & id2_tags))
        #             cor = (len(id1_tags & id2_tags)) / len(id1_tags | id2_tags)
        #             if cor > 0.0:
        #                 news_cor_list.append([newid1, newid2, format(cor, ".2f")])
        #                 logger.info("news_cor_list.append：{}".format([newid1, newid2, format(cor, ".2f")]))

        # tf-idf向量化
        tfidf = TFIDF(self.title_words, max_words=300)
        # content_model = TFIDF(content_words, max_words=1000)
        title_array = tfidf.fit_transform()
        # 余弦相似度计算
        consine = Cosine(n_recommendation=8)
        indices, similarities = consine.cal_similarity(title_array)

        db = pymysql.Connect(host=DB_HOST, user=DB_USER, password=DB_PASSWD, database=DB_NAME, port=DB_PORT,
                             charset='utf8')
        for row in range(len(self.titleDict)):
            title = self.titles[row]
            index = indices[row]
            similarity = similarities[row]
            for idx, sim in zip(index, similarity):
                if self.titleDict[title] != self.titleDict[self.titles[idx]]:
                    sql_w = "insert into news_api_newssimilar( new_id_base,new_id_sim,new_correlation ) values(%s, %s ,%s)" % (
                        self.titleDict[title], self.titleDict[self.titles[idx]], sim)
                    try:
                        cur = db.cursor()
                        cur.execute(sql_w)
                        db.commit()
                        logger.info("news_cor_list.append：{}".format(
                            [self.titleDict[title], self.titleDict[self.titles[idx]], format(sim, ".2f")]))
                        # print(sql_w)
                    except pymysql.Error as e:
                        print("getCorrelation() Error", e, sql_w, row)
        db.close()

        # return news_cor_list

    # def writeToMySQL(self):
    #     '''
    #         @Description：将相似度数据写入数据库
    #         @:param None
    #     '''
    #     db = pymysql.Connect(host=DB_HOST, user=DB_USER, password=DB_PASSWD, database=DB_NAME, port=DB_PORT,
    #                          charset='utf8')
    #     for row in self.news_cor_list:
    #         sql_w = "insert into news_api_newssimilar( new_id_base,new_id_sim,new_correlation ) values(%s, %s ,%s)" % (
    #             row[0], row[1], row[2])
    #         try:
    #             cur = db.cursor()
    #             cur.execute(sql_w)
    #             db.commit()
    #         except:
    #             print("rollback", row)
    #             logger.error("rollback：{}".format(row))
    #     print("相似度数据写入数据库：newsrec.newsim")


def beginCorrelation():
    '''
        @Description：启动相似度分析
        @:param None
    '''
    original_data_path = "Recommend/data/keywords/"
    files = os.listdir(original_data_path)
    for file in files:
        # print("开始计算文件 %s 下的新闻相关度。" % file)
        cor = Correlation(original_data_path + file)
        # cor.writeToMySQL()
    # print("\n相关度计算完毕，数据写入路径 z-othersd/data/correlation")
