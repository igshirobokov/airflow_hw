import logging
import dill
import pandas as pd
import json
import os


from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')

def get_test():
    test_examples = []
    for filename in os.listdir(f'{path}/data/test/'):
        with open(f'{path}/data/test/' + filename, 'rb') as file:
            example = json.load(file)
            test_examples.append(example)

    return test_examples


def predict():
    model_name = list(os.listdir(f'{path}/data/models'))[0]
    with open(f'{path}/data/models/{model_name}', 'rb') as file:
        best_model = dill.load(file)

    test_examples = get_test()

    df = pd.DataFrame.from_dict(test_examples)
    df['pred'] = best_model.predict(df)

    logging.info(df[['id', 'pred']])

    predict_name = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df[['id', 'pred']].to_csv(predict_name, index=False)

    logging.info(f'Predict is saved as {predict_name}')

if __name__ == '__main__':
    predict()
