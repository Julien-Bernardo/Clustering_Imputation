import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# How to handle categorical data depending on encoding type:
# Regression: Target and catboost encoding the easiest. Ordinal encoding, would need to round to integers? One hot encoding, take the mode?
# Classification:

class Clustering_Imputers:
    def __init__(self, X):
        self.X = X

    def fast_GMM(self, n_components, reg_covar, max_iter=10):
        """Perform GMM clustering on data with missing values.

        Args:
          X: An [n_samples, n_features] array of data to cluster.
          n_components: Number of clusters to form.
          reg_covar: Non-negative regularization added to the diagonal of covariance.
          Allows to assure that the covariance matrices are all positive.
          max_iter: Maximum number of EM iterations to perform.

        Returns:
          labels: An [n_samples] vector of integer labels.
          means: The mean of each mixture component.
          X_hat: Copy of X with the missing values filled in.
        """

        # Initialize missing values to their column means
        missing = ~np.isfinite(self.X)
        mu = np.nanmean(self.X, 0, keepdims=1)
        X_hat = np.where(missing, mu, self.X)

        for i in range(max_iter):
            if i > 0:
                cls = GaussianMixture(covariance_type='full', random_state=0,
                                      n_components=n_components, reg_covar=reg_covar,
                                      weights_init=prev_weights, means_init=prev_means)
            else:
                cls = GaussianMixture(covariance_type='full', random_state=0,
                                      n_components=n_components, reg_covar=reg_covar)

            # perform clustering on the filled-in data
            labels = cls.fit_predict(X_hat)
            means = cls.means_

            # fill in the missing values based on their cluster centroids
            X_hat[missing] = means[labels][missing]

            # when the labels have stopped changing then we have converged
            if i > 0 and np.all(labels == prev_labels):
                break

            prev_labels = labels
            prev_weights = cls.weights_
            prev_means = cls.means_

        return labels, means, X_hat


    def GMM(self, n_components, reg_covar, max_iter=100):
        """Perform GMM clustering on data with missing values.

        Args:
          X: An [n_samples, n_features] array of data to cluster.
          n_components: Number of clusters to form.
          reg_covar: Non-negative regularization added to the diagonal of covariance.
          Allows to assure that the covariance matrices are all positive.
          max_iter: Maximum number of EM iterations to perform.

        Returns:
          labels: An [n_samples] vector of integer labels.
          means: The mean of each mixture component.
          X_hat: Copy of X with the missing values filled in.
        """

        # Initialize missing values to their column means
        missing = ~np.isfinite(self.X)
        mu = np.nanmean(self.X, 0, keepdims=1)
        X_hat = np.where(missing, mu, self.X)

        for i in range(max_iter):
            cls = GaussianMixture(covariance_type='full', random_state=0,
                                  n_components=n_components, reg_covar=reg_covar)

            # perform clustering on the filled-in data
            labels = cls.fit_predict(X_hat)
            means = cls.means_

            # fill in the missing values based on their cluster centroids
            X_hat[missing] = means[labels][missing]

            # when the labels have stopped changing then we have converged
            if i > 0 and np.all(labels == prev_labels):
                break

            prev_labels = labels
            prev_weights = cls.weights_
            prev_means = cls.means_

        return labels, means, X_hat

    def Kmeans(self, n_clusters, max_iter=100):
        """Perform K-Means clustering on data with missing values.

        Args:
          X: An [n_samples, n_features] array of data to cluster.
          n_clusters: Number of clusters to form.
          max_iter: Maximum number of EM iterations to perform.

        Returns:
          labels: An [n_samples] vector of integer labels.
          centroids: An [n_clusters, n_features] array of cluster centroids.
          X_hat: Copy of X with the missing values filled in.

        From: https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data
        """

        # Initialize missing values to their column means
        missing = ~np.isfinite(self.X)
        mu = np.nanmean(self.X, 0, keepdims=1)
        X_hat = np.where(missing, mu, self.X)

        for i in range(max_iter):
            cls = KMeans(n_clusters, n_init=20)

            # perform clustering on the filled-in data
            labels = cls.fit_predict(X_hat)
            centroids = cls.cluster_centers_

            # fill in the missing values based on their cluster centroids
            X_hat[missing] = centroids[labels][missing]

            # when the labels have stopped changing then we have converged
            if i > 0 and np.all(labels == prev_labels):
                break

            prev_labels = labels
            prev_centroids = cls.cluster_centers_

        return labels, centroids, X_hat

    def fast_Kmeans(self, n_clusters, max_iter=10):
        """Perform K-Means clustering on data with missing values.

        Args:
          X: An [n_samples, n_features] array of data to cluster.
          n_clusters: Number of clusters to form.
          max_iter: Maximum number of EM iterations to perform.

        Returns:
          labels: An [n_samples] vector of integer labels.
          centroids: An [n_clusters, n_features] array of cluster centroids.
          X_hat: Copy of X with the missing values filled in.

        From: https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data
        """

        # Initialize missing values to their column means
        missing = ~np.isfinite(self.X)
        mu = np.nanmean(self.X, 0, keepdims=1)
        X_hat = np.where(missing, mu, self.X)

        for i in range(max_iter):
            if i > 0:
                # initialize KMeans with the previous set of centroids. this is much
                # faster and makes it easier to check convergence (since labels
                # won't be permuted on every iteration), but might be more prone to
                # getting stuck in local minima.
                cls = KMeans(n_clusters, init=prev_centroids)
            else:
                # do multiple random initializations in parallel
                cls = KMeans(n_clusters)


            # perform clustering on the filled-in data
            labels = cls.fit_predict(X_hat)
            centroids = cls.cluster_centers_

            # fill in the missing values based on their cluster centroids
            X_hat[missing] = centroids[labels][missing]

            # when the labels have stopped changing then we have converged
            if i > 0 and np.all(labels == prev_labels):
                break

            prev_labels = labels
            prev_centroids = cls.cluster_centers_
        return labels, centroids, X_hat

    def AIC_BIC(self, minimum, maximum, reg_covar):
        n_components = np.arange(minimum, maximum+1)
        X = [self.GMM(n_components=n, reg_covar=reg_covar)[2] for n in n_components]
        models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X[n-1]) for n in n_components]
        bic = [models[n-1].bic(X[n-1]) for n in n_components]
        aic = [models[n-1].aic(X[n-1]) for n in n_components]
        plt.plot(n_components, bic, label='BIC')
        plt.plot(n_components, aic, label='AIC')
        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.show()



    def Elbow(self, minimum, maximum):
        distortions = []
        inertias = []
        # silhouette = []
        mapping1 = {}
        mapping2 = {}
        # mapping3 = {}
        K = range(minimum, maximum+1)

        for k in K:
            # Imputing missing values with Kmeans
            labels, centroids, X_hat = self.Kmeans(n_clusters=k)
            # sil_score = silhouette_score(X_hat, labels)
            # silhouette.append(sil_score)

            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k).fit(X_hat)
            kmeanModel.fit(X_hat)

            distortions.append(sum(np.min(cdist(X_hat, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / X_hat.shape[0])
            inertias.append(kmeanModel.inertia_)

            mapping1[k] = sum(np.min(cdist(X_hat, kmeanModel.cluster_centers_,
                                           'euclidean'), axis=1)) / X_hat.shape[0]
            mapping2[k] = kmeanModel.inertia_

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()

        plt.plot(K, inertias, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.show()


