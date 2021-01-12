# Clustering_Imputation
Going beyond simple imputation of missing values using mean or mode, this class implements unsupervised clustering as a means to categorize data and impute missing values based on their group label. So far, Kmeans and Gaussian Mixture model have been implemented with a fast version that is less stable. These clustering methods only support numerical values right now, please convert categorical variables, I suggest count encoding or target encoding.

List of functions:

GMM:
- fast_GMM(n_components, reg_covar)
  - return labels, means, X_hat
- GMM(n_components, reg_covar)
  - return labels, means, X_hat
  
Kmeans:
- Kmeans(n_clusters)
  - return labels, centroids, X_hat
- fast_Kmeans(n_clusters)
  - return labels, centroids, X_hat

Visualization for choosing the number of clusters:
- AIC_BIC(minimum, maximum, reg_covar):
- Elbow(minimum, maximum)


Example of use:

```python
imp = Clustering_Imputers(dataframe)

labels, centroids, imputed_kmeans = imp.Kmeans(9)[2]

labels, means, imputed_gmm = imp.GMM(3, 1e-5)[2]
```
