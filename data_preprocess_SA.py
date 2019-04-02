import os
import json
import re
import jieba
import pyprind
from gensim.models.word2vec import Word2Vec

time_inteval = 10
partition_num = 2
#Weibo_path = '/home/hsucheng/LAB/fake_news_project/dataset/Weibo/'
Weibo_path = '/Users/hsucheng/Documents/LAB/fake_news_project/dataset/Weibo/'

#corpus_path = '/home/hsucheng/LAB/fake_news_project/dataset/Senta/data/train_data/corpus.train'
corpus_path = '/Users/hsucheng/Documents/LAB/fake_news_project/dataset/Senta/data/train_data/corpus.train'


    #for i in range(len(xxxxjson_name)):

class Data_PreProcess_SA:
    def __init__(self,partition_num,time_inteval,Weibo_path,corpus_path):
        self.partition_num = partition_num
        self.time_inteval = time_inteval
        self.Weibo_path = Weibo_path
        self.corpus_path = corpus_path
        self.xxxxjson_name = os.listdir(Weibo_path)

    def Cumulate_diff_time(self,user_data):
        time = [user_data[i]['t'] for i in range(len(user_data))]
        diff_time = []
        for i in range(len(time)):
            if i == 0 :
                diff_time.append(0)
            else:
                diff_time.append(time[i]-time[i-1])

        cumulate_diff_time = []
        for i in range(len(diff_time)):
            cumulate = sum([diff_time[j] for j in range(i+1)])
            cumulate_diff_time.append(cumulate)

        return cumulate_diff_time

    def Partition_users_text(self,xxxxjson_name_point):

        with open(self.Weibo_path+xxxxjson_name_point, encoding = "utf8") as f:
            data = json.load(f)
            author_data = data[0]
            user_data = data[1:]

        cumulate_diff_time = self.Cumulate_diff_time(user_data)

        time_step,point = 0,0
        partition_users_text = [[]]
        loop_key = True

        if cumulate_diff_time[-1] < self.time_inteval*self.partition_num:
            for j in range(len(user_data)):
                if cumulate_diff_time[j] < self.time_inteval*(time_step+1):
                    partition_users_text[time_step].append(user_data[point]['text'])
                else:
                    time_step +=1
                    partition_users_text.append([])
                    partition_users_text[time_step].append(user_data[point]['text'])
            if len(partition_users_text) < self.partition_num:
                token_num = self.partition_num-len(partition_users_text)
                for _ in range(token_num):
                    partition_users_text.append(['SHORTAGE'])
        else:
            while loop_key:
                if cumulate_diff_time[point] < self.time_inteval*(time_step+1):
                    partition_users_text[time_step].append(user_data[point]['text'])
                    point +=1
                else:
                    if cumulate_diff_time[point] < self.time_inteval*(time_step+2):
                        time_step +=1
                        partition_users_text.append([])
                        partition_users_text[time_step].append(user_data[point]['text'])
                        point +=1
                    else:
                        time_step +=1
                        partition_users_text.append([])
                        partition_users_text[time_step].append('SHORTAGE')
                if time_step+1 == self.partition_num:
                    loop_key = False
        return partition_users_text
    
    def __iter__(self):
        for i in range(len(self.xxxxjson_name)):
            xxxxjson_name_point = self.xxxxjson_name[i]
            partition_users_text = self.Partition_users_text(xxxxjson_name_point)
            for j in range(len(partition_users_text)):
                yield list(jieba.cut(re.sub('\W', '',''.join(partition_users_text[j]))))
        with open(self.corpus_path,encoding='utf-8') as f:
            sentiment_corpus = list(f)
        for i in pyprind.prog_bar(range(len(sentiment_corpus))):
            yield list(jieba.cut(re.sub('\W', '',''.join(sentiment_corpus[i].split()[1:]))))

data_preprocess_SA = Data_PreProcess_SA(partition_num,time_inteval,Weibo_path,corpus_path)
generator_SA = data_preprocess_SA
print('Start building Word2Vec!!')
Word2Vec_model = Word2Vec(generator_SA,size=100,min_count=1)
print('End building Word2Vec!!')





    
    
    
    
    
    
       












