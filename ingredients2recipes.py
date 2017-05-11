import json
import numpy as np
import collections
import tensorflow as tf

from tensorflow.models.rnn.translate import seq2seq_model

import pdb

tf.app.flags.DEFINE_integer("ingredients_vocab_size", 1000, "Ingredients vocabulary size.")
tf.app.flags.DEFINE_integer("recipes_vocab_size", 1000, "Recipes vocabulary size.")
tf.app.flags.DEFINE_string("checkpoints_dir", "checkpoints/ingredients2recipes/", "Checkpoints dir")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")

FLAGS = tf.app.flags.FLAGS

class Parser():
    def __init__(self, file_path):
        self.file_path = file_path
        self.ingredient_size = 1000
        self.extract()
        
    def extract(self):
        with open(self.file_path, "r") as f:
            lines = f.readlines()
            self.raw_data = [json.loads(line) for line in lines]
            
        self.all_ingredients = []
        self.all_recipes = []
        for recipe in self.raw_data:
            self.all_ingredients.extend([i[0] for i in recipe["ingredients"]])
            self.all_recipes.extend(recipe["name"])
            
        self.recipes_size = len(self.all_recipes)
            
        self.ingredients_counter = collections.Counter(self.all_ingredients)
        
        ingredients_count = [['UNK', -1]]
        ingredients_count.extend(self.ingredients_counter.most_common(self.ingredient_size - 1))
        
        self.ingredient_dict = dict()
        for _ingredient, _ in ingredients_count:
            self.ingredient_dict[_ingredient] = len(self.ingredient_dict)

        self.reversed_dictionary = dict(zip(self.ingredient_dict.values(), self.ingredient_dict.keys()))
        
    def generate_batch(self, batch_size=64):
        recipes = np.random.choice(self.raw_data, batch_size, replace=False)
        input_data = []
        output_data = []
        for recipe in recipes:
            output_data.append(recipe["name"])
            _group = []
            for ingredient, _ in recipe["ingredients"]:
                if ingredient in self.ingredient_dict:
                    _group.append(self.ingredient_dict[ingredient])
                else:
                    _group.append(0)
            input_data.append(_group)
        return input_data, output_data


class Engine():
    def __init__(self):
        self.parser = Parser("data/sitemap.json")
        self.batch_size = 64
        self.size = 256
        self.num_layers = 3
        self.num_encoder_symbols = 1000
        self.num_decoder_symbols = 1000
        self.embedding_size = 200
        
    def build_model(self):
        self.encoder_inputs = tf.placeholder(tf.int32, shape=[None], name="encoder")
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None], name="decoder")
        self.target_weights = tf.placeholder(tf.float32, shape=[None], name="weight")

        single_cell = tf.nn.rnn_cell.GRUCell(self.size)
        cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            
        return tf.nn.seq2seq.embedding_rnn_seq2seq(self.encoder_inputs,
                                                   self.decoder_inputs,
                                                   cell,
                                                   self.num_encoder_symbols,
                                                   self.num_decoder_sumbols,
                                                   self.embedding_size)
        
    def train(self):
        with tf.Session() as sess:
            outputs, states = self.build_model()
            
            
            
            
def main(_):
    engine = Engine()

    if FLAGS.decode:
        engine.decode()
    else:
        engine.train()

if __name__ == "__main__":
  tf.app.run()
