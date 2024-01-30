## Incidental Polysemanticity

In neural networks, neurons that simultaneously represent two completely unrelated features (e.g. a neuron in a computer vision getting activated for both dogs and airplanes), called “polysemantic neurons”, are a problem for interpretability. The usual story for why polysemanticity happens (see this Anthropic paper https://arxiv.org/abs/2209.10652) is that there are simply more useful features than there are neurons, so the network is forced to cram the features into fewer dimensions. We call this “necessary polysemanticity”.
Another possible explanation is that sometimes polysemanticity happens incidentally, because of how the random weights are initialized. In general, the reason neural networks learn anything is that at initialization, by random chance, some neuron happens to be very slightly correlated with one of the features that matter, and this correlation gets amplified by gradient descent. Therefore, if at the start, one neuron happens to be the most correlated neuron with both dogs and airplanes, then (depending on the specifics of the learning algorithm and the data) this might continue being the case throughout the learning process, causing “incidental polysemanticity” at the end.

You can find a more detailed theoretical understanding of how incidental polysemanticity happens in [this paper](https://arxiv.org/abs/2312.03096).

## Citation 

```bibtex
@article{2312.03096,
  author = {Lecomte, Victor and Thaman, Kushal and Chow, Trevor and Schaeffer, Rylan and Koyejo., Sanmi},
  title = {Incidental Polysemanticity},
  year = {2023.},
  journal = {arXiv preprint},
  eprint = {arXiv:2312.03096}
}
```
