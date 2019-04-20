import pandas as pd
import jieba.posseg as pseg

train = pd.read_csv(
    TRAIN_CSV_PATH, index_col=0)
train.head(3)

cols = ['title1_zh', 
        'title2_zh', 
        'label']
train = train.loc[:, cols]
train.head(3)

# 此函式能將輸入的文本 text 斷詞，並回傳除了標點符號以外的詞彙列表：
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([
        word for word, flag in words if flag != 'x'])

# 利用 Pandas 的 apply 函式，將 jieba_tokenizer 套用到所有新聞標題 A 以及 B 之上，做文本分詞：
train['title1_tokenized'] = \
    train.loc[:, 'title1_zh'] \
         .apply(jieba_tokenizer)
train['title2_tokenized'] = \
    train.loc[:, 'title2_zh'] \
         .apply(jieba_tokenizer)

text = '狐狸被陌生人拍照'
words = pseg.cut(text)
words = [w for w, f in words]
words

word_index = {
    word: idx  
    for idx, word in enumerate(words)
}
word_index

import keras

MAX_NUM_WORDS = 10000
tokenizer = keras \
    .preprocessing \
    .text \
    .Tokenizer(num_words=MAX_NUM_WORDS)

# 如同上述的步驟 1，我們得將新聞 A 及新聞 B 的標題全部聚集起來，為它們建立字典：
corpus_x1 = train.title1_tokenized
corpus_x2 = train.title2_tokenized
corpus = pd.concat([
    corpus_x1, corpus_x2])
corpus.shape

# 以下是我們語料庫的一小部分：
pd.DataFrame(corpus.iloc[:5],
             columns=['title'])

# 呼叫 tokenizer 為我們查看所有文本，並建立一個字典（步驟 2 & 3）：
tokenizer.fit_on_texts(corpus)

# 請 tokenizer 利用內部生成的字典分別將我們的新聞標題 A 與 新聞 B 轉換成數字序列：
x1_train = tokenizer \
    .texts_to_sequences(corpus_x1)
x2_train = tokenizer \
    .texts_to_sequences(corpus_x2)

# tokenizer.index_word 來將索引數字對應回本來的詞彙：
for seq in x1_train[:1]:
    print([tokenizer.index_word[idx] for idx in seq])


# 而針對原來長度不足的序列，我們則會在詞彙前面補零。Keras 
# 一樣有個方便函式 pad_sequences 來幫助我們完成這件工作：

MAX_SEQUENCE_LENGTH = 20
x1_train = keras \
    .preprocessing \
    .sequence \
    .pad_sequences(x1_train, 
                   maxlen=MAX_SEQUENCE_LENGTH)

x2_train = keras \
    .preprocessing \
    .sequence \
    .pad_sequences(x2_train, 
                   maxlen=MAX_SEQUENCE_LENGTH)


# One-hot Encoding: 分類欄位 label 要進行從文本到數字的轉換了：

import numpy as np 

# 定義每一個分類對應到的索引數字
label_to_index = {
    'unrelated': 0, 
    'agreed': 1, 
    'disagreed': 2
}

# 將分類標籤對應到剛定義的數字
y_train = train.label.apply(
    lambda x: label_to_index[x])

y_train = np.asarray(y_train) \
            .astype('float32')

y_train[:5]

y_train = keras \
    .utils \
    .to_categorical(y_train)

# 要切訓練資料集 / 驗證資料集，scikit-learn 中的 train_test_split 函式是一個不錯的選擇：
from sklearn.model_selection \
    import train_test_split

VALIDATION_RATIO = 0.1
# 小彩蛋
RANDOM_STATE = 9527

x1_train, x1_val, \
x2_train, x2_val, \
y_train, y_val = \
    train_test_split(
        x1_train, x2_train, y_train, 
        test_size=VALIDATION_RATIO, 
        random_state=RANDOM_STATE
)

from keras import layers
lstm = layers.LSTM()

# 當然，我們可以用 tokenizer 裡頭的字典 index_word 還原文本看看：
for i, seq in enumerate(x1_train[:5]):
    print(f"新聞標題 {i + 1}: ")
    print([tokenizer.index_word.get(idx, '') for idx in seq])
    print()

# 而在 Keras 裡頭，我們可以使用 Embedding 層來幫我們做詞嵌入（Word Embedding）
from keras import layers
embedding_layer = layers.Embedding(
    MAX_NUM_WORDS, NUM_EMBEDDING_DIM)    

# 基本參數設置，有幾個分類
NUM_CLASSES = 3

# 在語料庫裡有多少詞彙
MAX_NUM_WORDS = 10000

# 一個標題最長有幾個詞彙
MAX_SEQUENCE_LENGTH = 20

# 一個詞向量的維度
NUM_EMBEDDING_DIM = 256

# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 128


# 建立孿生 LSTM 架構（Siamese LSTM）
from keras import Input
from keras.layers import Embedding, \
    LSTM, concatenate, Dense
from keras.models import Model

# 分別定義 2 個新聞標題 A & B 為模型輸入
# 兩個標題都是一個長度為 20 的數字序列
top_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')
bm_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')

# 詞嵌入層
# 經過詞嵌入層的轉換，兩個新聞標題都變成
# 一個詞向量的序列，而每個詞向量的維度
# 為 256
embedding_layer = Embedding(
    MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
top_embedded = embedding_layer(
    top_input)
bm_embedded = embedding_layer(
    bm_input)

# LSTM 層
# 兩個新聞標題經過此層後
# 為一個 128 維度向量
shared_lstm = LSTM(NUM_LSTM_UNITS)
top_output = shared_lstm(top_embedded)
bm_output = shared_lstm(bm_embedded)

# 串接層將兩個新聞標題的結果串接單一向量
# 方便跟全連結層相連
merged = concatenate(
    [top_output, bm_output], 
    axis=-1)

# 全連接層搭配 Softmax Activation
# 可以回傳 3 個成對標題
# 屬於各類別的可能機率
dense =  Dense(
    units=NUM_CLASSES, 
    activation='softmax')
predictions = dense(merged)

# 我們的模型就是將數字序列的輸入，轉換
# 成 3 個分類的機率的所有步驟 / 層的總和
model = Model(
    inputs=[top_input, bm_input], 
    outputs=predictions)

為了確保用 Keras 定義出的模型架構跟預期相同，我們也可以將其畫出來：
from keras.utils import plot_model
plot_model(
    model, 
    to_file='model.png', 
    show_shapes=True, 
    show_layer_names=False, 
    rankdir='LR')

model.summary()

#在 Keras 裏頭，我們可以這樣定義模型的損失函數：
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# 決定一次要放多少成對標題給模型訓練
BATCH_SIZE = 512

# 決定模型要看整個訓練資料集幾遍
NUM_EPOCHS = 10

# 實際訓練模型
history = model.fit(
    # 輸入是兩個長度為 20 的數字序列
    x=[x1_train, x2_train], 
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    # 每個 epoch 完後計算驗證資料集
    # 上的 Loss 以及準確度
    validation_data=(
        [x1_val, x2_val], 
        y_val
    ),
    # 每個 epoch 隨機調整訓練資料集
    # 裡頭的數據以讓訓練過程更穩定
    shuffle=True
)

# 讓我們把測試資料集讀取進來：
import pandas as pd
test = pd.read_csv(
    TEST_CSV_PATH, index_col=0)
test.head(3)

# 以下步驟分別對新聞標題 A、B　進行
# 文本斷詞 / Word Segmentation
test['title1_tokenized'] = \
    test.loc[:, 'title1_zh'] \
        .apply(jieba_tokenizer)
test['title2_tokenized'] = \
    test.loc[:, 'title2_zh'] \
        .apply(jieba_tokenizer)

# 將詞彙序列轉為索引數字的序列
x1_test = tokenizer \
    .texts_to_sequences(
        test.title1_tokenized)
x2_test = tokenizer \
    .texts_to_sequences(
        test.title2_tokenized)

# 為數字序列加入 zero padding
x1_test = keras \
    .preprocessing \
    .sequence \
    .pad_sequences(
        x1_test, 
        maxlen=MAX_SEQUENCE_LENGTH)
x2_test = keras \
    .preprocessing \
    .sequence \
    .pad_sequences(
        x2_test, 
        maxlen=MAX_SEQUENCE_LENGTH)    

# 利用已訓練的模型做預測
predictions = model.predict(
    [x1_test, x2_test])


# 將機率值最大的類別當作答案，並將這個結果轉回對應的文本標籤即可上傳到 Kaggle：
index_to_label = {v: k for k, v in label_to_index.items()}

test['Category'] = [index_to_label[idx] for idx in np.argmax(predictions, axis=1)]

submission = test \
    .loc[:, ['Category']] \
    .reset_index()

submission.columns = ['Id', 'Category']
submission.head()

