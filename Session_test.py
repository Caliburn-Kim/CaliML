# All imports
from __future__ import print_function

import cklib
from cklib import ckconst
from cklib.ckstd import fprint
from cklib import ckstd
from cklib import DataFrame
import joblib
import gc

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

seed = 22
dataset_path = './bin/iscx2017session.csv'
clf_init = ['rf', 'dt', 'et', 'adt', 'arf', 'gbt']

if __name__ == "__main__":
    for clf in clf_init:
        fprint(None, '{} classifier using'.format(clf))
        dataframe = DataFrame.Session_Dataset(clf = clf, random_state = seed)
        dataframe.skip_data('Heartbleed', 'Infiltration', u'Web Attack \x96 XSS', u'Web Attack \x96 Sql Injection')
        dataframe.read_csv(path = dataset_path)
        dataframe.modelling()
        dataframe.predict()

        label_encoder = dataframe.getLabelEncoder()
        train_pred = dataframe.getTrainPredict()
        test_pred = dataframe.getTestPredict()
        train_true = label_encoder.transform(dataframe.getTrainLabel())
        test_true = label_encoder.transform(dataframe.getTestLabel())
        
        train_report = classification_report(
            y_true = train_true,
            y_pred = train_pred,
            target_names = label_encoder.classes_,
            output_dict = True
        )

        train_cfmx = confusion_matrix(
            y_true = train_true,
            y_pred = train_pred,
            labels = label_encoder.transform(label_encoder.classes_).tolist()
        ).transpose()

        test_report = classification_report(
            y_true = test_true,
            y_pred = test_pred,
            target_names = label_encoder.classes_,
            output_dict = True
        )

        test_cfmx = confusion_matrix(
            y_true = test_true,
            y_pred = test_pred,
            labels = label_encoder.transform(label_encoder.classes_).tolist()
        ).transpose()

        fprint(None, 'Train Result')
        fprint(None, 'Accuracy: {}'.format(train_report['accuracy']))
        fprint(None, 'Precision: {}'.format(train_report['macro avg']['precision']))
        fprint(None, 'Recall: {}'.format(train_report['macro avg']['recall']))
        fprint(None, 'F1-Score: {}'.format(train_report['macro avg']['f1-score']))

        print('Train confusion matrix')
        ckstd.print_cfmx(train_cfmx, 7)

        fprint(None, 'Test Result')
        fprint(None, 'Accuracy: {}'.format(test_report['accuracy']))
        fprint(None, 'Precision: {}'.format(test_report['macro avg']['precision']))
        fprint(None, 'Recall: {}'.format(test_report['macro avg']['recall']))
        fprint(None, 'F1-Score: {}'.format(test_report['macro avg']['f1-score']))

        print('Test confusion matrix')
        ckstd.print_cfmx(test_cfmx, 7)

        dataframe = None
        del dataframe
        gc.collect()
        print('\n\n')
