#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import psycopg2
from flask import Flask, jsonify, request, abort
from preprocess import MyVocabularyProcessor

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("vocab_filepath", "runs/1541748108/checkpoints/vocab",
                       "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "runs/1541748108/checkpoints/model-33000",
                       "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Build vocabulary
vocab_processor = MyVocabularyProcessor(30, min_frequency=0)
vocab_processor = vocab_processor.restore(FLAGS.vocab_filepath)


def char2vec(arr):
    return np.asarray(list(vocab_processor.transform(arr)))


def get_test_data_set(text):
    return char2vec(np.full([len(abbr_vec_arr)], text))


def batch_iter(data, batch_size):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.asarray(data)
    # print(data)
    # print(data.shape)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def prepare_abbr_arr():
    cur = conn.cursor()
    cur.execute("SELECT trim(machine_name), trim(formal_name) FROM game_alias WHERE status = 'confirmed'")
    conn.commit()
    abbr = []
    formal = []
    for t in cur:
        abbr.append(t[0])
        formal.append(t[1])
    return np.asarray(abbr), np.asarray(formal), char2vec(abbr)


def predict(x1_arr):
    batches = batch_iter(list(zip(abbr_vec_arr, x1_arr, np.zeros([len(abbr_vec_arr)]))), 2 * FLAGS.batch_size)
    all_predictions = []
    all_d = []
    for db in batches:
        x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
        batch_predictions, batch_acc, batch_sim = sess.run([predictions, accuracy, sim],
                                                           {input_x1: x1_dev_b, input_x2: x2_dev_b,
                                                            input_y: y_dev_b, dropout_keep_prob: 1.0})
        all_predictions = np.concatenate([all_predictions, batch_predictions])
        all_d = np.concatenate([all_d, batch_sim])

    keys = np.nonzero(all_d)
    print(keys)
    print(list(zip(abbr_arr[keys], formal_arr[keys], all_predictions[keys])))
    # print(all_predictions[keys])
    mink = np.argmin(all_predictions[keys])
    print(abbr_arr[keys][mink])
    print(all_predictions[keys][mink])
    # return 'ok'
    return formal_arr[keys][mink]


# Initial global variables

app = Flask(__name__)
conn = psycopg2.connect("dbname=feedback_nlp user=postgres password=postgres host=data-platform-sh-01.nvidia.com")
abbr_arr, formal_arr, abbr_vec_arr = prepare_abbr_arr()

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(FLAGS.model))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, FLAGS.model)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]


@app.route('/sim/debug', methods=['POST'])
def sim_debug():
    data = request.get_json(force=True)
    if 'text' in data:
        _text = data['text']
        return jsonify(predict(get_test_data_set(_text)))
    else:
        return abort(406)


def main():
    app.run(host="0.0.0.0", debug=True)


if __name__ == '__main__':
    main()
    conn.close()
