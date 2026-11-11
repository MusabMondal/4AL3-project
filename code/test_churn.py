"""Behavioral checks: python -m unittest discover -s code -p 'test_*.py'."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from churn_data import load_dataset, split_dataset, make_preprocessor, FEATURES
from churn_evaluation import evaluate, ChurnExperiment

class ChurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset(Path(__file__).with_name('Churn_Modelling.csv'))

    def test_split_disjoint_and_reproducible(self):
        a=split_dataset(self.data,0); b=split_dataset(self.data,0)
        self.assertEqual([len(x) for x in a.values()],[7000,1500,1500])
        self.assertEqual(len(set.union(*(set(x.index) for x in a.values()))),10000)
        for name in a: self.assertTrue(a[name].index.equals(b[name].index))

    def test_duplicate_customers_rejected(self):
        d=self.data.copy(); d.loc[1,'CustomerId']=d.loc[0,'CustomerId']
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bad.csv';d.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Repeated CustomerId'):load_dataset(p)

    def test_probability_validation_and_threshold(self):
        with self.assertRaises(ValueError):evaluate(pd.Series([0,1]),np.array([.2,np.nan]))
        m=evaluate(pd.Series([0,1]),np.array([.5,.6]))
        self.assertEqual(m['confusion_matrix'],[[1,0],[0,1]])

    def test_inference_does_not_refit_preprocessing(self):
        splits=split_dataset(self.data,0)
        e=ChurnExperiment(make_preprocessor(),DummyClassifier(strategy='prior'))
        e.fit(splits['train']); scaler=e.pipeline['preprocess']['scale']
        before=scaler.mean_.copy()
        customers=splits['validation'].copy(); customers['Balance']=1e12
        p=e.score(customers)
        np.testing.assert_array_equal(before,scaler.mean_)
        np.testing.assert_allclose(p,splits['train'].Exited.mean())
        self.assertEqual(e.pipeline['preprocess'].n_features_in_,len(FEATURES))

if __name__=='__main__':unittest.main()
