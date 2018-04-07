# Code for Intel AI DevCamp, Chicago, 7 April 2018

## Demonstrate Speed of Intel Distribution of Python

```
$ python numpy_exp.py
```

## Demonstrate Intel-Optimized TensorFlow

```
$ python
>>> import tensorflow
>>> tensorflow.global_variables_initializer().run(session=tensorflow.InteractiveSession())
>>> exit()
```

## Demonstrate Un-clustered and Clustered MNIST Classifier

```
$ python mnist_softmax_unclustered.py
```

```
$ python mnist_softmax_clustered.py --as_node localhost:2223 # in first window
$ python mnist_softmax_clustered.py --as_node localhost:2222 # in second window
```
