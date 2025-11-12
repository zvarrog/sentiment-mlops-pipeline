"""
Модуль обработки признаков для обучения моделей.
"""

from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin


class DenseTransformer(TransformerMixin, BaseEstimator):
    """Превращает scipy sparse в dense numpy.

    Необходим для моделей, не поддерживающих sparse:
    - sklearn.linear_model.LogisticRegression(solver='lbfgs')
    - sklearn.neural_network.MLPClassifier
    - sklearn.ensemble.HistGradientBoostingClassifier

    Примечание: материализация TF-IDF в dense может потребовать много памяти.
    """

    def transform(self, x):
        """Преобразует разреженную матрицу в плотную.

        Args:
            x: Входная матрица (sparse или dense)

        Returns:
            np.ndarray: Плотный numpy массив
        """
        return x.toarray() if sp.issparse(x) else x
