# Notebook

## Optimisations
### Builder Pattern and Batch Normalisation
Implemented CNN builder class and produced a model with batch normalisation.

##### Assumption
The assumption is an improvement in terms of accuracy when comparing base to batch normalised model.

##### Result
The result showcase an increase in terms of accuracy for both base and batch normalised models.

However, the results produced showcases a small difference (~0.3%) in accuracy when comparing base to batch normalised model.

##### Explanation
1. **Dataset Simplicity**: The CIFAR-10 is relatively small dataset. The benefits of **BN** is more apparent in deeper networks and larger datasets.

2. **Network Depth**: CNN is relatively shallow, **internal covariare shift**(distribution change in activations) is not prominent.

3. **Well-tuned Learning Rate**: BN's advantage is to allow for higher learning rates

4. **Regularisation Effect Already Present**: BN acts as a regularisation process. If other regularisation process exists (dropout or data augmentation), BN's effect might be minimal.

#### Suggestions
1. Use in deeper networks
2. When training takes too long (BN speeds up convergence)
3. Larger dataset