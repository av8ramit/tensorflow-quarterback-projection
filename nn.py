"""An Example of a DNNClassifier for the NFL Quarterback dataset."""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int,
                    help='number of training steps')

if tf.gfile.Exists("results"):
  tf.gfile.DeleteRecursively("results")
tf.gfile.MakeDirs("results")


TRAIN_FILENAME = "dataset/quarterback_training.csv"
DEV_FILENAME = "dataset/quarterback_dev.csv"
TEST_FILENAME = "dataset/quarterback_test.csv"

CSV_COLUMN_NAMES = ['DraftYear',
                    'Round',
                    'Pick',
                    'Age',
                    'GamesPlayed',
                    'Completions',
                    'Attempts',
                    'Yards',
                    'Touchdowns',
                    'Interceptions',
                    'RushAttempts',
                    'RushYards',
                    'RushTouchdowns',
                    'Player',
                    'College',
                    'Conference',
                    'Team',
                    'Heisman',
                    'Verdict'
                    ]
DEL_FEATURE = [
                #'DraftYear', 0.6
                #'Round', 0.9
                #'Pick', 0.4
                'Age', #0.8
                # 'GamesPlayed', 0.8
                #'Completions', 0.7
                #'Attempts', 0.5
                #'Yards', 0.6
                #'Touchdowns', 0.7
                #'Interceptions', 0.6
                #'RushAttempts', 0.5
                #'RushYards', 0.5
                #'RushTouchdowns', 0.8
                'Player',
                'College', #0.9
                #'Conference', 0.7
                #'Team', 0.8
                #'Heisman', 0.7
                ]
DATA_TYPES = {'DraftYear': np.int32,
              'Round': np.int32,
              'Pick': np.int32,
              'Age': np.int32,
              'GamesPlayed': np.int32,
              'Completions': np.int32,
              'Attempts': np.int32,
              'Yards': np.int32,
              'Touchdowns': np.int32,
              'Interceptions': np.int32,
              'RushAttempts': np.int32,
              'RushYards': np.int32,
              'RushTouchdowns': np.int32,
              'Player': np.object,
              'College': np.object,
              'Conference': np.object,
              'Team': np.object,
              'Heisman': np.int32,
              'Verdict': np.int32
              }
VERDICTS = [
            'Bust',
            'NFL-Ready'
            ]

def load_data(y_name='Verdict'):
    train = pd.read_csv(TRAIN_FILENAME, names=CSV_COLUMN_NAMES, header=0,
      dtype=DATA_TYPES)
    train_x, train_y = train, train.pop(y_name)

    dev = pd.read_csv(DEV_FILENAME, names=CSV_COLUMN_NAMES, header=0,
      dtype=DATA_TYPES)
    dev_x, dev_y = dev, dev.pop(y_name)
    return (train_x, train_y), (dev_x, dev_y)

def load_test_data(filename, y_name='Verdict'):
    train = pd.read_csv(filename, names=CSV_COLUMN_NAMES, header=0,
      dtype=DATA_TYPES)
    train_x, train_y = train, train.pop(y_name)
    return train_x, train_y

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    features = {k: np.array(v.values) for k, v in features.items()}  # Convert each column to a NumPy array.
    for delete_feature_key in DEL_FEATURE:
        del(features[delete_feature_key])
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is not None:
        features = {k: np.array(v.values) for k, v in features.items()}  # Convert each column to a NumPy array.
        labels = np.array(labels)

    for delete_feature_key in DEL_FEATURE:
        del(features[delete_feature_key])
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (dev_x, dev_y) = load_data()
    train_x = dict(train_x)
    dev_x = dict(dev_x)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        if key in DEL_FEATURE:
            print "Skipping feature %s" % key
            continue
        if key == "College":
          categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key='College', vocabulary_file='vocab_list/colleges.txt', vocabulary_size=70,
            num_oov_buckets=1)
          embedding_column = tf.feature_column.embedding_column(
            categorical_column=categorical_column,
            dimension=70)
          my_feature_columns.append(embedding_column)
        elif key == 'Team':
          categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key='Team', vocabulary_file='vocab_list/teams.txt', vocabulary_size=32,
            num_oov_buckets=0)
          embedding_column = tf.feature_column.embedding_column(
            categorical_column=categorical_column,
            dimension=32)
          my_feature_columns.append(embedding_column)
        elif key == 'Conference':
          categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key='Conference', vocabulary_file='vocab_list/conference.txt', vocabulary_size=12,
            num_oov_buckets=0)
          embedding_column = tf.feature_column.embedding_column(
            categorical_column=categorical_column,
            dimension=12)
          my_feature_columns.append(embedding_column)
        else:
          my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[50, 100, 50],
        optimizer=tf.train.ProximalAdagradOptimizer(
              learning_rate=0.01,
              l2_regularization_strength=0.0001
        ),
        # The model must choose between 2 classes.
        n_classes=2,
        model_dir="results")


    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    # eval_result = classifier.evaluate(
    #     input_fn=lambda:eval_input_fn(train_x, train_y, args.batch_size))
    # print('\nTraining set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(dev_x, dev_y, args.batch_size))
    print('\nDev set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # eval_result = classifier.evaluate(
    #     input_fn=lambda:eval_input_fn(dev_x, dev_y, args.batch_size))
    # print('\nDev set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predict_x, predict_y = load_test_data(DEV_FILENAME)
    predict_x = predict_x.to_dict('list')
    player_array = predict_x["Player"]
    expected = predict_y.tolist()


    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x, batch_size=args.batch_size))
    prediction_array = []

    for i, (pred_dict, expec) in enumerate(zip(predictions, expected)):
        player = player_array[i]
        template = ('\nPrediction is "{}" for {} ({:.1f}%), who was actually a "{}."')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(VERDICTS[class_id], player, 100 * probability, VERDICTS[expec]))
        prediction_array.append(class_id)

    a = tf.confusion_matrix(expected, prediction_array)
    sess = tf.Session()
    print(sess.run(a))
    print classification_report(expected, prediction_array, target_names=['Bust', 'NFL-Ready'])




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)