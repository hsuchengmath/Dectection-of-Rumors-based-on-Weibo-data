from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input,Dense,LSTM,concatenate,Bidirectional,Lambda,RepeatVector




def BI_LSTM_Model(generator_CP,maxlength_words,corpus_path):
    pretrain_batch_size = 100  #% pretrain_batch_size

    with open(corpus_path,encoding='utf-8') as f:  #% corpus_path
        corpus_sentiment = list(f)
    data_size = len(corpus_sentiment)
    print('Starting building BI_LSTM!!')
    input_data = Input(shape=(maxlength_words,100)) #%100
    lstm_ht = Bidirectional(LSTM(100,input_shape=(maxlength_words,100),name='lstm_ht'))(input_data) #%100
    FCN1 = Dense(64,activation='tanh',name='latent_vec')(lstm_ht)
    Y_hat = Dense(1,activation='sigmoid')(FCN1)
        
    model = Model(inputs=input_data,outputs=Y_hat)
    adam = Adam(lr=0.01) 
    model.compile(optimizer=adam,loss="binary_crossentropy",metrics=['accuracy'] )
    model.fit_generator(generator_CP,steps_per_epoch=round(data_size/pretrain_batch_size),epochs=2)

    sentiment_embedding_model = Model(inputs=model.input,outputs=model.get_layer('latent_vec').output)

    print('Finishing building BI_LSTM!!')
    return model,sentiment_embedding_model


##Setting partition num = 3
#NV_input : (?,3,128)
#AU_input : (?,4,100) 
#SA_input : (?,3,maxlength_words,100)
##SA_embedding_input : (?,3,embedding_dim=100)

##Experiment data
SA_embedding = np.arange(1399200).reshape(4664,3,100)
NV = np.arange(1790976).reshape(4664,3,128)
AU = np.arange(1865600).reshape(4664,4,100)
Y = np.arange(4664).reshape(4664,1)

partition_num = 3

def Split_Function(name):
    def Split(x):
        if name == 'Author':
            return x[:,0,:]
        elif name == 'Users':
            return x[:,1:,:]
    return Lambda(Split)
        
def SA_NV_AU_Model(partition_num,SA_embedding,NV,AU,Y):
    ## SA_embedding
    SA_embedding_input = Input(shape=(SA_embedding.shape[1],SA_embedding.shape[2]))
    (SA_embedding_all_ht,SA_embedding_final_ht,_) = LSTM(100,input_shape=(SA_embedding.shape[1],SA_embedding.shape[2]),return_sequences=True, return_state=True)(SA_embedding_input)

    ## NV and AU
    NV_input = Input(shape=(NV.shape[1],NV.shape[2]))
    NV_all_ht = LSTM(64,input_shape=(NV.shape[1],NV.shape[2]),return_sequences=True)(NV_input)

    AU_input = Input(shape=(AU.shape[1],AU.shape[2]))
    AU_all_ht = LSTM(64,input_shape=(AU.shape[1],AU.shape[2]),return_sequences=True)(AU_input)

    A_ht = Split_Function('Author')(AU_all_ht)
    Repeat_A_ht = RepeatVector(partition_num)(A_ht)
    U_ht = Split_Function('Users')(AU_all_ht)

    NV_U_ht = concatenate([NV_all_ht,U_ht])
    NV_U_FCN = Dense(64,activation='tanh')(NV_U_ht)

    A_NV_U_FCN = concatenate([Repeat_A_ht,NV_U_FCN])

    Y_hat = concatenate([SA_embedding_all_ht,A_NV_U_FCN])

    model = Model(inputs=[SA_embedding_input,NV_input,AU_input],outputs=Y_hat)
    model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'] )
    print(model.summary())



