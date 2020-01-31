from operator import itemgetter
from keras.models import load_model
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd

#create sentence pair
df = pd.read_csv('data/exp1/java_test.csv')
print(df.shape)
df = df.dropna()
print(df.shape)
sentences1 = list(df['sentences1'])
sentences2 = list(df['sentences2'])
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]

#load tokenizer
with open('checkpoints/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#load model
model = load_model(sys.argv[1])

test_sentence_pairs = [('What can make Physics easy to learn?','How can you make physics easy to learn?'),('How many times a day do a clocks hands overlap?','What does it mean that every time I look at the clock the numbers are the same?')]

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs, siamese_config['MAX_SEQUENCE_LENGTH'])

# get intermediate layer output
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('lstm').output)
intermediate_output = intermediate_layer_model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1)

#preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
#results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
#results.sort(key=itemgetter(2), reverse=True)
print intermediate_output