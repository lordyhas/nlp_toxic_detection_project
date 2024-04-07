
# Loading Dependencies
import os
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.layers import Dense, Input, Dropout, Lambda
from keras.optimizers import Adam
from keras.optimizers import AdamW
from keras.optimizers import Nadam
from keras.optimizers import RMSprop
from keras.models import Model
from keras.callbacks import ModelCheckpoint
#from kaggle_datasets import KaggleDatasets
import transformers

from tokenizers import BertWordPieceTokenizer

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline


from transformers import TFDistilBertModel, DistilBertTokenizer
import tensorflow as tf
from keras import regularizers
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy

class TrainingOptimizer():
    def __init__(self, lr:float = 1e-5, epsilon: float = 1e-7,) -> None:
        self.lr = lr
        self.epsilon = epsilon,
        pass
    def adam(self, amsgrad:bool= False,):
        return  Adam(learning_rate=self.lr, epsilon=self.epsilon, amsgrad=amsgrad)
    def adamboost(self, amsgrad:bool= False,): # adam with weight_decay
        return  AdamW(learning_rate=self.lr, epsilon=self.epsilon, amsgrad=amsgrad)
    def rmsprop(self):
        return RMSprop(learning_rate=self.lr, epsilon=self.epsilon)
    def amsgrad(self):
        self.adam(amsgrad=True)
    def amsgradboost(self): # adam with weight_decay
        return self.adamboost(amsgrad = True)
    def nadam(self): 
        return  Nadam(learning_rate=self.lr, epsilon=self.epsilon)
    
    
    
    def named(self, name:str = None):
        if name == None or name == "":
            raise ValueError("Please choose an optimizer")
        elif name.lower() == "adam":
            return self.adam()
        elif name.lower() == "adamboost":
            return self.adamboost()
        elif name.lower() == "rmsprop":
            return self.rmsprop()
        elif name.lower() == "amsgrad":
            return self.amsgrad() 
        elif name.lower() == "nadam":
            return self.nadam()
        elif name.lower() == "amsgradboost":
            return self.amsgradboost()
        else:
            raise ValueError("Optimizer [",name,"] not found")
    
class Dataset:
    def __init__(self, train_data:tuple, valid_data:tuple, test_data:tuple, batch_size):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        
    def train(self):
        return (
            tf.data.Dataset
            .from_tensor_slices(self.train_data)
            .repeat()
            .shuffle(2048)
            .batch(self.batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
    def valid(self):
        return (
            tf.data.Dataset
            .from_tensor_slices(self.valid_data)
            .batch(self.batch_size)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
    def test(self):
        return (
            tf.data.Dataset
            .from_tensor_slices(self.test_data)
            .batch(self.batch_size)
        )
        
    
    @staticmethod
    def balanced_data(data:pd.DataFrame, random_state = 42, extra_ratio = 0.0):
        data_majority = data[data['toxic'] == 0]
        data_minority = data[data['toxic'] == 1]

        # Sous-échantillonnage des exemples de la classe majoritaire pour équilibrer les classes
        data_majority_downsampled = data_majority.sample(n=len(data_minority), random_state=random_state)

        # Concaténer les données sous-échantillonnées de la classe majoritaire avec les données de la classe minoritaire
        balanced_data = pd.concat([data_majority_downsampled, data_minority])

        # Mélanger les données pour mélanger les classes
        balanced_data = balanced_data.sample(frac=1, random_state=random_state)
        
        return balanced_data



class BertModel(object):
    dir_path = "../"
    EPOCHS = 15
    BATCH_SIZE = 32 #* strategy.num_replicas_in_sync
    MAX_LEN = 192
    
    def __init__(self, max_len=MAX_LEN, optimizer = Adam(), loss = BinaryCrossentropy()) -> None:
        self.optimizer = optimizer
        self.loss = loss
        self.max_len = max_len
    
       
    def build(self, model:int = 1):
        if model == 1 :
            return self.__model_1()
        
    def __model_1(self):
        """
        That function create the BERT model for training
        """
        # Charger le modèle pré-entraîné DistilBERT et le tokenizer
        distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        

        model = tf.keras.Sequential([
            # La couche d'entrée
            Input(shape=(self.max_len,), dtype=tf.int32, name="input_word_ids"),

            # La couche DistilBERT
            distilbert_model.layers[0],

            # La couche pour obtenir le premier token [CLS]
            Lambda(lambda seq: seq[:, 0, :]),
            # La couche de sortie
            Dense(1, activation='sigmoid')
        ])

        #loss = tf.keras.losses.BinaryCrossentropy()
        #metrics = tf.metrics.BinaryAccuracy()

        # Compiler le modèle
        # Compiler le modèle avec une loss adaptée à la classification binaire
        model.compile(optimizer = self.optimizer, loss=self.loss, metrics=['accuracy',])

        # Afficher le résumé du modèle

        return model
    
    def __model_2(self):
        pass
    
    @staticmethod
    def text_encode(texts, tokenizer, chunk_size=256, maxlen=MAX_LEN):
        """
        Encoder for encoding the text into sequence of integers for BERT Input
        """
        # Set maximum length
        tokenizer.enable_truncation(max_length=maxlen)

        #tokenizer.enable_padding(max_length=maxlen)
        # Enable padding
        #tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", pad_to_multiple_of=None)
        tokenizer.enable_padding()
        all_ids = []

        for i in tqdm(range(0, len(texts), chunk_size)):
            text_chunk = texts[i:i+chunk_size].tolist()
            encs = tokenizer.encode_batch(text_chunk)
            all_ids.extend([enc.ids for enc in encs])

        return np.array(all_ids)
    
    @classmethod
    def tokenizer():
        # First load the real tokenizer
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained('outputs/tokenizers')
        except (OSError, ValueError):
            tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
            # Save the loaded tokenizer locally
            tokenizer.save_pretrained('outputs/tokenizers')
        return tokenizer
    
    @classmethod
    def text_tokenizer():
        text_tokenizer = BertWordPieceTokenizer('outputs/tokenizers/vocab.txt', lowercase=False)
        return text_tokenizer