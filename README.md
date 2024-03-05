# Scaling law by data manifold

This is the repo accompanying my [essay on scaling law by data manifold](https://yuxi-liu-wired.github.io/essays/posts/scaling-law-by-data-manifold).

I first wrote it for the [2023 Fall version of the CS282 Deep Neural Networks](https://inst.eecs.berkeley.edu/~cs182/fa23/) course at Berkeley.

## Description

`q_scaling_data_manifold_sol.ipynb` is the notebook in which I implemented the scaling law by data manifold, in the synthetic dataset case, where a student network would fit a randomly initialized teacher network. It did not replicate the original paper very well, perhaps because I could not train the models to convergence.

I trained convolutional networks over the CIFAR-10 dataset. The code describing the training runs are in `cifar10-scaling.py` and `cifar10-scaling_run.py`. To reproduce the dataset, run `python cifar10-scaling_run.py`. It outputs the training run logs into the `logs` folder, which is in the TensorBoard format.

You can view them for yourself to get a feel for the shape of the dataset, by running `tensorboard --logdir=cifar10/logs`. Suffice to say that the dataset is not very clean, and accidents sometimes happen.

* A few training runs terminated early thanks to GPU running out of memory, or the computer going to sleep. A few training runs got mashed together into the same logging file due to my choosing to start multiple training runs in parallel, and since they were all started at the exact same second, they all logged into the exact same file with the exact same name.
* Some training runs only started decreasing in loss after a few epochs. A few training runs completely failed to train, with loss curves not decreasing. These delays and failures are not consistently reproducible. I have not managed to find out why, presumably due to some trivial accident.

Despite these accidents, it is clean enough for us to proceed, with a little caution.

`q_data_manifold_cifar10_sol.ipynb` is the notebook in which I analyzed the data in `logs` folder. It replicated the original paper very well. I would say even better than the original paper. I am very proud of this notebook.
