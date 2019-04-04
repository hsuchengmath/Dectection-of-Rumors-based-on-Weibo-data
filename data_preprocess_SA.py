import os
import json
import re
import jieba
import pyprind
import numpy as np
from gensim.models.word2vec import Word2Vec
from Model import *
time_inteval = 10
partition_num = 2
#Weibo_path = '/home/hsucheng/LAB/fake_news_project/dataset/Weibo/'
Weibo_path = '/Users/hsucheng/Documents/LAB/fake_news_project/dataset/Weibo/'

#corpus_path = '/home/hsucheng/LAB/fake_news_project/dataset/Senta/data/train_data/corpus.train'
corpus_path = '/Users/hsucheng/Documents/LAB/fake_news_project/dataset/Senta/data/train_data/corpus.train'


class Data_PreProcess_SA:  

    def __init__(self,partition_num,time_inteval,Weibo_path,corpus_path):
        self.partition_num = partition_num
        self.time_inteval = time_inteval
        self.Weibo_path = Weibo_path
        self.corpus_path = corpus_path
        self.xxxxjson_name = os.listdir(Weibo_path)
        self.pretrain_batch_size = 100

    def __iter__(self):
        for i in range(len(self.xxxxjson_name)):
            xxxxjson_name_point = self.xxxxjson_name[i]
            partition_users_text = self.Partition_users_text(xxxxjson_name_point)
            for j in range(len(partition_users_text)):
                yield list(jieba.cut(re.sub('\W', '',''.join(partition_users_text[j]))))
        with open(self.corpus_path,encoding='utf-8') as f:
            sentiment_corpus = list(f)
        for i in range(len(sentiment_corpus)):
            yield list(jieba.cut(re.sub('\W', '',''.join(sentiment_corpus[i].split()[1:]))))
        yield ['SHORTAGE']

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
    
    def Batch_Function(self,datafile_path,batch_size):
        if datafile_path != self.corpus_path:
            print('The format of data should be similiar with original copus data!!\n ')
            print('Please try again and correct your file path to corpus path!!\n')
        else:
            with open(datafile_path,encoding='utf-8') as f:
                datafile = list(f) 
            data_size = len(datafile)                   
            batch_num = round(data_size/batch_size)
            batch_point = []
            remainder = data_size-(batch_num*batch_size)
            if remainder != 0:
                batch_num -=1    
                for i in range(batch_num):
                    batch_point.append(((i+1)*batch_size)-1)
                batch_point.append(data_size-1)
            else:
                for i in range(batch_num):
                    batch_point.append(((i+1)*batch_size)-1)
        return batch_point   

    def Maxlengh_words(self):
        lengh_words = []
        for i in range(len(self.xxxxjson_name)):
            xxxxjson_name_point = self.xxxxjson_name[i]
            partition_users_text = self.Partition_users_text(xxxxjson_name_point)
            for j in range(len(partition_users_text)):
                lengh_words.append(len(list(jieba.cut(re.sub('\W', '',''.join(partition_users_text[j]))))))
        with open(self.corpus_path,encoding='utf-8') as f:
            sentiment_corpus = list(f)
        for i in range(len(sentiment_corpus)):
            lengh_words.append(len(list(jieba.cut(re.sub('\W', '',''.join(sentiment_corpus[i].split()[1:]))))))
        return max(lengh_words)
    
    def Generator_Corpus(self):
        with open(self.corpus_path,encoding='utf-8') as f:
            sentiment_corpus = list(f)
        maxlengh_words = self.Maxlengh_words()
        batch_point = self.Batch_Function(self.corpus_path,self.pretrain_batch_size)
        while True:
            X_batch,Y_batch,point = [],[],0
            for i in range(len(sentiment_corpus)):
                if i <= batch_point[point]:
                    corpus_textvec = []
                    corpus_segment = list(jieba.cut(re.sub('\W', '',''.join(sentiment_corpus[i].split()[1:]))))
                    for j in range(len(corpus_segment)):
                        corpus_textvec.append(Word2Vec_model[corpus_segment[j]])
                    for j in range(maxlengh_words-len(corpus_segment)):
                        corpus_textvec.append(Word2Vec_model['SHORTAGE'])
                    X_batch.append(corpus_textvec)
                    Y_batch.append(int(sentiment_corpus[i].split()[0]))
                else:
                    X_batch = np.array(X_batch)
                    Y_batch = np.array(Y_batch)
                    yield (X_batch,Y_batch)
                    point +=1
                    X_batch,Y_batch = [],[]

    def Users_sentiment(self,Word2Vec_model,bi_LSTM_model):         
        users_sentiment = []
        for i in range(len(self.xxxxjson_name)):
            xxxxjson_name_point = self.xxxxjson_name[i]
            partition_users_text = self.Partition_users_text(xxxxjson_name_point)
            users_textvec = []
            for j in range(len(partition_users_text)):
                user_text = list(jieba.cut(re.sub('\W', '',''.join(partition_users_text[j]))))
                remainder = self.Maxlengh_words()-len(user_text)
                user_textvec = []
                for k in range(len(user_text)):
                    user_textvec.append(Word2Vec_model[user_text[k]])
                for k in range(remainder):
                    user_textvec.append(Word2Vec_model['SHORTAGE'])
                users_textvec.append(user_textvec)
            users_textvec = np.array(users_textvec)
            users_sentiment.append(bi_LSTM_model.predict(users_textvec))
        users_sentiment = np.array(users_sentiment)
        return users_sentiment

    def SA_Embedding(self,Word2Vec_model,sentiment_embedding_model):    
        print('Starting building SA_embedding!!')      
        SA_embedding = []
        for i in pyprind.prog_bar(range(len(self.xxxxjson_name))):
            xxxxjson_name_point = self.xxxxjson_name[i]
            partition_users_text = self.Partition_users_text(xxxxjson_name_point)
            embedding = []
            for j in range(len(partition_users_text)):
                user_text = list(jieba.cut(re.sub('\W', '',''.join(partition_users_text[j]))))
                remainder = self.Maxlengh_words()-len(user_text)
                user_textvec = []
                for k in range(len(user_text)):
                    user_textvec.append(Word2Vec_model[user_text[k]])
                for k in range(remainder):
                    user_textvec.append(Word2Vec_model['SHORTAGE'])
                embedding.append(user_textvec)
            embedding = np.array(embedding)
            SA_embedding.append(sentiment_embedding_model.predict(embedding))
        SA_embedding = np.array(SA_embedding)
        print('Ending building SA_embedding!!')        
        return SA_embedding



data_preprocess_SA = Data_PreProcess_SA(partition_num,time_inteval,Weibo_path,corpus_path)
generator_SA = data_preprocess_SA
print('Start building Word2Vec!!')
Word2Vec_model = Word2Vec(generator_SA,size=100,min_count=1)
print('End building Word2Vec!!')
generator_Corpus = data_preprocess_SA.Generator_Corpus()
maxlengh_words = data_preprocess_SA.Maxlengh_words()

bi_LSTM_model,sentiment_embedding_model = BI_LSTM_Model(generator_Corpus,maxlengh_words,corpus_path)
SA_Y = data_preprocess_SA.Users_sentiment(Word2Vec_model,bi_LSTM_model)  #(3,2,1)  ##

SA_embedding = data_preprocess_SA.SA_Embedding(Word2Vec_model,sentiment_embedding_model)  ##(3,2,64)

        
        
        
        
        
        
        












