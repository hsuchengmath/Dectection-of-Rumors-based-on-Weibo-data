import os
import json
import numpy as np
import jieba
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import networkx as nx
import matplotlib.pyplot as plt
import pyprind
import re
from node2vec_main import *


time_inteval = 10 #dont touch
partition_num = 2
ranktop_num = 4  #dont touch

nodepair_xx_filename = 'nodepair_ns2.edgelist'
nodevec_xx_filename = 'nodevec_ns2.txt'
matplotlib_filename = 'matplotlib_ns2.png'

#Weibo_path = '/home/hsucheng/LAB/fake_news_project/dataset/Weibo/'
Weibo_path = '/Users/hsucheng/Documents/LAB/fake_news_project/dataset/Weibo/'

#Weibotxt_path = '/home/hsucheng/LAB/fake_news_project/dataset/Weibo.txt'
Weibotxt_path = '/Users/hsucheng/Documents/LAB/fake_news_project/dataset/Weibo.txt'

#node2vec_path = '/home/hsucheng/LAB/fake_news_project/code/new_node2vec_LSTM/'+nodevec_xx_filename
node2vec_path = '/Users/hsucheng/Documents/LAB/fake_news_project/code/new_node2vec_LSTM/'+nodevec_xx_filename


nodepair_xx_filename = 'nodepair_test.edgelist'
nodevec_xx_filename = 'nodevec_test.txt'
matplotlib_filename = 'matplotlib_test.png'

class Data_PreProcess_NV_AU:

    def __init__(self,partition_num,time_inteval,Weibo_path,Weibotxt_path,nodepair_xx_filename,nodevec_xx_filename):
        self.partition_num = partition_num
        self.time_inteval = time_inteval
        self.ranktop_num = 4 
        self.Weibo_path = Weibo_path
        self.Weibotxt_path = Weibotxt_path
        self.xxxxjson_name = os.listdir(Weibo_path)
        self.nodepair_xx_filename = nodepair_xx_filename
        self.nodevec_xx_filename = nodevec_xx_filename
        #self.matplotlib_filename = matplotlib_filename

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
        return partition_users_text,author_data

    def Doc2Vec_Model(self):
        documents_tag = []
        cumulate_num = 0
        print('Start Preprocess!!')
        for i in pyprind.prog_bar(range(len(self.xxxxjson_name))):
            xxxxjson_name_point = self.xxxxjson_name[i]
            partition_users_text,author_data = self.Partition_users_text(xxxxjson_name_point)
            documents_tag.append(TaggedDocument(list(jieba.cut(author_data['text'])),[cumulate_num]))
            cumulate_num +=1
            for k in range(len(partition_users_text)):
                documents_tag.append(TaggedDocument(list(jieba.cut(','.join(partition_users_text[k]))),[cumulate_num]))
                cumulate_num +=1 
        print('Start building Doc2vec!!')
        doc2vec_model = Doc2Vec(dm=1, vector_size=100, window=5, negative=5, hs=0, min_count=2, workers=4)
        doc2vec_model.build_vocab(documents_tag)
        doc2vec_model.train(documents_tag, total_examples=doc2vec_model.corpus_count, epochs=40)
        print('End building Doc2vec!!')
        return doc2vec_model

    def Dict_users_textvec(self,doc2vec_model):
        print('Starting building Dict_Users_TextVec!!')
        dict_users_textvec = {}
        for i in pyprind.prog_bar(range(len(self.xxxxjson_name))):
            xxxxjson_name_point = self.xxxxjson_name[i]
            partition_users_text,author_data = self.Partition_users_text(xxxxjson_name_point)
            for j in range(len(partition_users_text)):
                partition_users_text[j] = doc2vec_model.infer_vector(list(jieba.cut(','.join(partition_users_text[j]))))            
            partition_users_text = np.array(partition_users_text)
            for j in range(partition_users_text.shape[0]):
                dict_users_textvec[xxxxjson_name_point+'@'+str(j)] = partition_users_text[j]
        print('Ending building Dict_Users_TextVec!!')
        return dict_users_textvec

    def Author_users_textvec(self,doc2vec_model):
        print('Starting building Author_Users_TextVec!!')
        author_users_textvec = []
        for i in pyprind.prog_bar(range(len(self.xxxxjson_name))):
            xxxxjson_name_point = self.xxxxjson_name[i]
            partition_users_text,author_data = self.Partition_users_text(xxxxjson_name_point)
            for j in range(len(partition_users_text)):
                partition_users_text[j] = doc2vec_model.infer_vector(list(jieba.cut(','.join(partition_users_text[j]))))            
            partition_users_text = np.array(partition_users_text)
            author_textvec = doc2vec_model.infer_vector(list(jieba.cut(author_data['text']))).reshape((1,100))
            one_author_users_textvec = np.concatenate((author_textvec,partition_users_text))
            author_users_textvec.append(one_author_users_textvec)
        author_users_textvec = np.array(author_users_textvec) 
        print('Ending building Author_Users_TextVec!!')
        return author_users_textvec

    def Cosine_Similarity(self,v1,v2):
        length_v1 = sum([v1[i]**2 for i in range(len(v1))])**(0.5)
        length_v2 = sum([v2[i]**2 for i in range(len(v2))])**(0.5)
        inner_product = sum([v1[i]*v2[i] for i in range(len(v1))])
        cos_sim = inner_product/(length_v1*length_v2)
        return cos_sim 
    
    def Similarity_top(self,dict_node_vec):
        print('Start building graph similarity!!')
        nodename = list(dict_node_vec.keys())
        graph_similarity = []
        for i in range(len(nodename)):
            print(i)
            node_cs = []
            for j in range(len(nodename)):            
                if i != j:
                    textvec1 = dict_node_vec[nodename[i]]
                    textvec2 = dict_node_vec[nodename[j]]
                    cosine_similarity = self.Cosine_Similarity(textvec1,textvec2)
                    node_cs.append((cosine_similarity,nodename[j]))
            ranktop_node = sorted(node_cs,reverse=True)[:self.ranktop_num]
            for k in range(len(ranktop_node)):
                graph_similarity.append((nodename[i],ranktop_node[k][1],ranktop_node[k][0]))
        print('End building graph similarity!!')
        print('Start building Graph!!')
        G = nx.Graph()
        G.add_weighted_edges_from(graph_similarity)
        print('Fininsh Graph!!')            
        nx.write_edgelist(G,self.nodepair_xx_filename)
        nx.draw(G,with_labels=False,node_size=1)
        #plt.show()
        #plt.savefig(matplotlib_filename)

    def Node2Vec(self):
        Node2vec_main(self.nodepair_xx_filename,self.nodevec_xx_filename)

    def Dict_nodevec(self):
        nodevec = list(open(self.nodevec_xx_filename,'r'))
        dict_nodevec = {}
        for i in pyprind.prog_bar(range(len(nodevec)-1)):
            node_name_vec = nodevec[i+1].split()
            node_name = node_name_vec[0].split('@')[0]
            vec = [float(node_name_vec[j+1]) for j in range(len(node_name_vec)-1)]
            if node_name not in dict_nodevec:
                dict_nodevec[node_name] = [vec]
            else:
                dict_nodevec[node_name].append(vec)
        return dict_nodevec

    def X_Function(self,dict_nodevec):
        xxxxjson_name = os.listdir(self.Weibo_path)
        node_name = list(dict_nodevec.keys())
        X = []
        for i in pyprind.prog_bar(range(len(self.xxxxjson_name))):
            X.append([])
            if xxxxjson_name[i] in node_name:
                for j in range(len(dict_nodevec[xxxxjson_name[i]])):
                    X[i].append(dict_nodevec[xxxxjson_name[i]][j])
            else:
                print('Error!!')        
        X = np.array(X)
        return X

    def Y_Funtion(self):
        xxxxjson_name = os.listdir(self.Weibo_path)
        xxxxjson_name_no = [int(self.xxxxjson_name[i].split('.')[0]) for i in range(len(self.xxxxjson_name))]
        Weibotxt = open(self.Weibotxt_path, 'r')
        Weibotxt = list(Weibotxt)
        Weibotxt_no = [(int(Weibotxt[i].split()[2]),int(Weibotxt[i].split()[1][6])) for i in range(len(Weibotxt))]
        Y = []
        for i in range(len(xxxxjson_name_no)):
            no = xxxxjson_name_no[i]
            for j in range(len(Weibotxt_no)):
                if no == Weibotxt_no[j][0]:
                    Y.append(Weibotxt_no[j][1])
        Y = np.array(Y)
        return Y



data_preprocess_NV_AU = Data_PreProcess_NV_AU(partition_num,time_inteval,Weibo_path,Weibotxt_path,nodepair_xx_filename,nodevec_xx_filename)
doc2vec_model = data_preprocess_NV_AU.Doc2Vec_Model()
dict_users_textvec = data_preprocess_NV_AU.Dict_users_textvec(doc2vec_model)
AU_X = data_preprocess_NV_AU.Author_users_textvec(doc2vec_model)  ##(3,3,100)
data_preprocess_NV_AU.Similarity_top(dict_users_textvec)
data_preprocess_NV_AU.Node2Vec()
dict_nodevec = data_preprocess_NV_AU.Dict_nodevec()
NV_X = data_preprocess_NV_AU.X_Function(dict_nodevec)   ##(3,2,128)
NV_AU_Y = data_preprocess_NV_AU.Y_Funtion()   ##(3,)













