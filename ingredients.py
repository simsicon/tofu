import json
import collections
import numpy as np
import tensorflow as tf
from matplotlib import pylab
from matplotlib import font_manager
from sklearn.manifold import TSNE
import pdb

flags = tf.app.flags

flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g data/sitemap.json")

FLAGS = flags.FLAGS

class Parser:
    def __init__(self, batch_size, vocabulary_size, filepath):
        self.filepath = filepath
        self.cursor = 0
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.extract()
    
    def extract(self):
        with open(self.filepath, "r") as f:
            lines = f.readlines()
            self.raw_data = [json.loads(line) for line in lines]

        self.ingredient_pairs = []
        _all_ingredients = []
        
        for recipe in self.raw_data:
            _ingredients = recipe['ingredients']
            _all_ingredients += [_igd[0] for _igd in _ingredients]
        self.ingredients_counter = collections.Counter(_all_ingredients)
        
        count = [['UNK', -1]]
        count.extend(self.ingredients_counter.most_common(self.vocabulary_size - 1))
        
        self.dictionary = dict()
        for _ingredient, _ in count:
            self.dictionary[_ingredient] = len(self.dictionary)

        self.reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        
        for recipe in self.raw_data:
            _ingredients = recipe['ingredients']
            for _i, _trgt_ingredient in enumerate(_ingredients):
                for _j, _ctx_ingredient in enumerate(_ingredients):
                    if _i != _j:
                        _ctx, _trgt = _ctx_ingredient[0], _trgt_ingredient[0]
                        if _ctx in self.dictionary:
                            _ctx_index = self.dictionary[_ctx]
                        else:
                            _ctx_index = 0
                            
                        if _trgt in self.dictionary:
                            _trgt_index = self.dictionary[_trgt]
                        else:
                            _trgt_index = 0
                            
                        self.ingredient_pairs.append( (_ctx_index, _trgt_index) )
                        
        self.pairs_length = len(self.ingredient_pairs)
                
    def generate_batch(self):
        next_cursor = (self.cursor + self.batch_size) % self.pairs_length
        if next_cursor < self.cursor:
            batch_pairs = self.ingredient_pairs[self.cursor:] + self.ingredient_pairs[:next_cursor]
        else:
            batch_pairs = self.ingredient_pairs[self.cursor:next_cursor]
            
        batch_data = [pair[0] for pair in batch_pairs]
        batch_labels = [pair[1] for pair in batch_pairs]
        try:
            batch_labels = np.reshape(batch_labels, (self.batch_size, 1))
        except:
            pdb.set_trace()
        self.cursor = next_cursor
        return batch_data, batch_labels

class Ingredient2Vec:
    def __init__(self, train_file):
        self.batch_size = 128
        self.vocabulary_size = 1000
        self.emb_dim = 256
        self.num_steps = 1000000
        self.parser = Parser(self.batch_size, self.vocabulary_size, train_file)
        self.build_graph()
    
    def build_graph(self):
        self.graph = tf.Graph()
        batch_size, voc_size, emb_dim = self.batch_size, self.vocabulary_size, self.emb_dim
        num_sampled = 128
        
        with self.graph.as_default():
            with tf.variable_scope("ingredients"):
                self.train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
                self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
                
                with tf.variable_scope("embeddings"):
                    embeddings = tf.get_variable("table", [voc_size, emb_dim],
                                                initializer=tf.random_uniform_initializer(-1.0, 1.0))
                
                with tf.variable_scope("softmax"):
                    _initializer = tf.truncated_normal_initializer(stddev=1.0/np.sqrt(emb_dim))
                    weights = tf.get_variable("weights", [voc_size, emb_dim], initializer=_initializer)
                    biases = tf.get_variable("biases", initializer=tf.zeros_initializer(voc_size))
                
                emb = tf.nn.embedding_lookup(embeddings, self.train_dataset)
                
                self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                                weights, biases, emb, self.train_labels, num_sampled, voc_size))
                self.global_step = tf.Variable(0, trainable=False)
                self.learning_rate = tf.train.exponential_decay(10.0, self.global_step, 10000, 0.95, staircase=True)
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
                
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                self.normalized_embeddings = embeddings / norm
                
    def train(self):
        ckpt_dir = "checkpoints/"
        ckpt_filename = "bigrams2vec.ckpt"
        with tf.Session(graph=self.graph) as sess:
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            saver = tf.train.Saver()
            
            def train_loop(start_at=0):
                average_loss = 0
                for step in range(start_at, self.num_steps, 1):
                    batch_data, batch_labels = self.parser.generate_batch()
                    feed_dict = {self.train_dataset : batch_data, self.train_labels : batch_labels}
                    _, _l, _lr = sess.run([self.optimizer, self.loss, self.learning_rate], feed_dict=feed_dict)
                    average_loss += _l

                    if step % 1000 == 0:
                        if step > 0:
                            average_loss = average_loss / 1000
                            print('Average loss at step %d: %f with learning rate %f' %
                                  (step, average_loss, _lr))
                            average_loss = 0

                    if step % 100000 == 0:
                        if step > 0:
                            save_path = saver.save(sess, ckpt_dir + ckpt_filename,
                                                   global_step=self.global_step)
                            print "Ingredient2Vec Model saved in file: %s" % save_path
        
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print "Ingredient2Vec Model load from file: %s" % ckpt.model_checkpoint_path
                current_step = self.global_step.eval()
                if current_step < self.num_steps:
                    train_loop(start_at=current_step)
            else:
                train_loop()
                
            save_path = saver.save(sess, ckpt_dir + ckpt_filename)
            print "Ingredient2Vec Model saved in file: %s" % save_path

            self.final_embeddings = self.normalized_embeddings.eval()
                        
    def plot(self):
        fm = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
        
        num_points = 500
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=10000)
        two_d_embeddings = tsne.fit_transform(self.final_embeddings[1:num_points, :])
        labels = [self.parser.reversed_dictionary[i] for i in range(1, num_points)]
        pylab.figure(figsize=(15,15))
        for i, label in enumerate(labels):
            x, y = two_d_embeddings[i, :]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontproperties=fm)
        pylab.savefig('ingredients2vec.png')
                        
def main(_):
    if not FLAGS.train_data:
        print "--train_data must be specified."
        exit(-1)

    i2v = Ingredient2Vec(FLAGS.train_data)
    i2v.train()
    i2v.plot()

if __name__ == "__main__":
    tf.app.run()