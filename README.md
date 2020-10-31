# Product Quantization k-Nearest Neighbors

Implementation of Product Quantization k-Nearest Neighbors ([original paper](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf)).  

> Product quantization is a vector compression algorithm that summarizes partitions of the train data efficiently by means of their k-means centroids.  
> The PQKNN algorithm performs a k-nearest neighbors search on the compressed data in a smart manner using lookups on pre-computed tables. 

## Benefit (& explaination) of PQKNN

The bottleneck of classic k-nearest neighbors is that its performance degrades on large datasets.  
Each query results in a full pass over the data to compute all distances!

**By using product quantization, the nearest neighbor search is accelerated by greatly simplifying the distance calculations.** 

### Training

*At train time* the data is compressed by means of the product quantization algorithm.  
This compression happens as follows;
1) The original train data vectors are partitioned into *n* disjoint subvectors.
2) A *c*-bit code is mapped to each partition using k-means clustering (with k = 2^c).

### Predicting
*At prediction time* the k-nearest neighbors search is performed.  
For each test sample, a table containing the distance of the sample to each of the centroids is calculated. Doing table lookups to the shared centroids (per partition) speeds up the distance computation to a great extent.

## Disadvantage of PQKNN

PQKNN is an approximate nearest neighbors algorithm, meaning that an approximate distance (based on the centroids) is used to classify training samples. This might be less exact than the classic k-nearest neighbors.

## Hyperparameters of PQKNN

When creating a ProductQuantizationKNN instance, one has to assign 2 hyperparameters;
* *n*: the amount of subvectors, i.e., the number of partitions.
* *c*: the number of bits to encode each subvector with. This determines the amount of clusters for k-means (used the compression step), i.e., *k* = 2^*c*.

When training, i.e., compressing the train data, no other hyperparameters are required.

When predicting, one has to assign 1 hyperparameter;
* *k*: the k of k-nearest neighbors.

### Example usage


```py
from product_quantization import ProductQuantizationKNN

# Fetch the train data & train labels + the test data
train_data, train_labels = ...
test_data = ...

# Create PQKNN object that partitions each train sample in 7 subvectors and encodes each subvector in 4 bits.
pqknn = ProductQuantizationKNN(n=7, c=4)
# Perform the compression
pqknn.compress(train_data, train_labels)

# Classify the test data using k-Nearest Neighbor search (with k = 10) on the compressed training
preds = pqknn.predict(test_data, nearest_neighbors=10)
```

For more elaborate example look at the Notebook `Example.ipynb`.  
You should be able to rerun this notebook as the data is downloaded in the notebook itself.

## Benchmark PQKNN implementation on MNIST

In the Notebook `Example.ipynb` this implementation of PQKNN is benchmarked on the famous MNIST dataset.
The train-test-split is identical as [the one provided by Yann LeCun](http://yann.lecun.com/exdb/mnist/);
* Train data; 60,000 28x28 images that are flattened into an array with shape (60000, 784).
* Test data; 10,000 28x28 images that are flattened into an array with shape (10000, 784).


| Algorithm                    | Train time (s) | Predict time (s) | Accuracy (%) |
|------------------------------|----------------|------------------|--------------|
| `PQKNN` *(n=7, c=4)*             | 25.09          | 7.44             | 92.35        |
| `PQKNN` *(n=7, c=7)*             | 132.72         | 15.73            | 96.6         |
| `PQKNN` *(n=4, c=8)*             | 58.24          | <b>7.14</b>            | 96.08        |
| `sklearn.neighbors.KNeighborsClassifier` | <b>12.97</b>      | 209.55    | <b>96.65</b>   |


**Discussion**: We observe that train time is significantly faster for sklearn its `KNeighborsClassifier` (2x-10x faster). This does not wheigh up against the enormous boost gained at prediction time for the PQKNN algorithm (15x-30x faster), at the cost of a small loss in accuracy. This significantly faster lookup (predictions) enable a trained PQKNN algorithm to have higher performance in production.

*Minor remark*: as the PQKNN implementation works multithreaded. The displayed results for `sklearn.neighbors.KNeighborsClassifier` are with *n_jobs*=-1. When n_jobs=None, the performance at predict time is even 60x slower than the PQKNN implementation

---

## References

Original paper: https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf  
Jegou, Herve, Matthijs Douze, and Cordelia Schmid. "Product quantization for nearest neighbor search." IEEE transactions on pattern analysis and machine intelligence 33.1 (2010): 117-128.

Blogpost explaining very well the underlying concepts; https://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/
