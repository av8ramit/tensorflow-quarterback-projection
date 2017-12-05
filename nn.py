"""An Example of a DNNClassifier for the NFL Quarterback dataset."""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int,
                    help='number of training steps')

TRAIN_FILENAME = "dataset/quarterback_training.csv"
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
DEL_FEATURE = ['Player']
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

    test = pd.read_csv(TEST_FILENAME, names=CSV_COLUMN_NAMES, header=0,
      dtype=DATA_TYPES)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)


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
    (train_x, train_y), (test_x, test_y) = load_data()
    train_x = dict(train_x)
    test_x = dict(test_x)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        if key in DEL_FEATURE:
            continue
        if key == "College":
          categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key='College', vocabulary_file='vocab_list/colleges.txt', vocabulary_size=69,
            num_oov_buckets=0)
          embedding_column = tf.feature_column.embedding_column(
            categorical_column=categorical_column,
            dimension=69)
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

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[50, 100, 50],
        # The model must choose between 2 classes.
        n_classes=2)

    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)


    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(train_x, train_y, args.batch_size))

    print('\nTrain set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    test_players = ['Marcus Mariota', 'B.J. Daniels', 'Eli Manning', 'Drew Brees', 'Sage Rosenfels', 'Michael Vick', 'Donovan McNabb', 'Akili Smith', 'Tom Brandstater', 'Curtis Painter', 'JaMarcus Russell']
    expected = [
                'NFL-Ready',
                'Bust',
                'NFL-Ready',
                'NFL-Ready',
                'Bust',
                'NFL-Ready',
                'NFL-Ready',
                'Bust',
                'Bust',
                'Bust',
                'Bust'
                ]
    predict_x = {
                  'DraftYear': [2015, 2013, 2004, 2001, 2001, 2001, 1999, 1999, 2009, 2009, 2007],
                  'Round': [1, 7, 1, 2, 4, 1, 1, 1, 6, 6, 1],
                  'Pick': [2, 237, 1, 32, 109, 1, 2, 3, 174, 201, 1],
                  'Age': [21, 23, 23, 22, 23, 21, 22, 24, 24, 24, 22],
                  'GamesPlayed': [41, 47, 43, 45, 30, 22, 45, 23, 45, 46, 36],
                  'Completions': [779, 649, 829, 1026, 306, 192, 548, 323, 584, 987, 493],
                  'Attempts': [1167, 1132, 1363, 1678, 587, 343, 938, 571, 989, 1648, 797],
                  'Yards': [10796, 8433, 10119, 11792, 4164, 3299, 8389, 5148, 6857, 11163, 6625],
                  'Touchdowns': [105, 52, 81, 90, 18, 21, 77, 45, 47, 67, 52],
                  'Interceptions': [14, 39, 35, 45, 26, 11, 26, 15, 32, 46, 21],
                  'RushAttempts': [337, 526, 128, 252, 164, 235, 465, 171, 132, 225, 139],
                  'RushYards': [2237, 2068, -135, 900, 660, 1299, 1561, 367, 152, 348, 79],
                  'RushTouchdowns': [29, 5, 5, 14, 14, 13, 19, 6, 8, 13, 4],
                  'Player': ['Marcus Mariota', 'B.J. Daniels', 'Eli Manning', 'Drew Brees', 'Sage Rosenfels', 'Michael Vick', 'Donovan McNabb', 'Akili Smith', 'Tom Brandstater', 'Curtis Painter', 'JaMarcus Russell'],
                  'College': ['Oregon', 'South Florida', 'Mississippi', 'Purdue', 'Iowa St.', 'Virginia Tech', 'Syracuse', 'Oregon', 'Fresno St.', 'Purdue', 'LSU'],
                  'Conference': ['Pac-12', 'Southeastern', 'Southeastern', 'Big Ten', 'Big 12', 'Atlantic Coast', 'Atlantic Coast', 'Pac-12', 'Mountain West', 'Big Ten', 'Southeastern'],
                  'Team': ['TEN', 'SFO', 'SDG', 'SDG', 'WAS', 'ATL', 'PHI', 'CIN', 'DEN', 'IND', 'OAK'],
                  'Heisman': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'Verdict': [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0]
                }

    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x, batch_size=args.batch_size))
    for i, (pred_dict, expec) in enumerate(zip(predictions, expected)):
        player = test_players[i]
        template = ('\nPrediction is "{}" for {} ({:.1f}%), who was actually a "{}."')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(VERDICTS[class_id], player, 100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)