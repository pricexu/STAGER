# STAGER
Implementation of 'Generalized Few-Shot Node Classification', submitted to ICDM'22.

## Hardwares and Softwares
We implement STAGER in pytorch and use one NVIDIA Tesla V100 SXM2-32GB. All the experiments can be finished in an hour clock time. We test our code with python 3.7, pytorch 1.10, and corresponding dependencies.

## Dataset split
For every dataset, we randomly select $10$ classes as the novel classes and $10$ classes as the validation classes (for the $N$-way setup, we will further sample $N$ classes from the 10 classes). The remaining classes are for base classes. $50$ nodes are selected per novel class for test. We select int(50*N/(# of base classes)) nodes per base class for test to ensure the numbers of test nodes from the novel classes and that from the base classes are comparable. Following the $N$-way $K$-shot setup ($K=1$ or $K=3$), $K$ nodes per novel class are labelled, and $50$ nodes per base class are labelled. There is no overlap between the labelled nodes and the test nodes. Since the test nodes come from both the novel classes and the base classes, our validation setup is composed by two parts. First, all the remaining nodes from the base classes (except the labelled nodes and the test nodes) are used for validation. Second, the setup for the validation classes is exactly the same as the novel classes. Need to mention that the \emph{base/validation/novel classes split is fixed} for all the models. Since the difference between classes can lead into large deviation on the metric, for a fair comparison, when we report the results we do not change the base/validation/novel classes split.

## Hyperparameter selection for STAGER and baseline methods.
All the MLP modules in STAGER are with "ReLU" as the activation function except $\phi_2$ whose activation function is "tanh". They share the same number of hidden units which is searched from $\{24, 48, 72, 96, 120\}$. We train the model with Adam. The learning rate is searched from $\{1\times 10^{-2}, 5\times 10^{-3}, 1\times 10^{-3}, 5\times 10^{-4}\}$ and the weight decay is searched from $\{1\times 10^{-4}, 5\times 10^{-4}, 1\times 10^{-3}, 5\times 10^{-3}, 1\times 10^{-2}, 5\times 10^{-2}, 0.1\}$. The sample re-weighting is selected as $50:N$ or $100:N$ for the novel and base classes respectively. $N$ is the number of ways for the novel classes. The integer number of steps for propagation $p$ is searched from $[2,10]$. The dropout rate between layers of the predictor $f$ and the weight assigner $g$ is searched from $\{0, 0.2, 0.4, 0.5, 0.6, 0.8\}$. For the imbalanced episodic training, the number of query nodes per class $I=30$; the number of pseudo-novel classes $N$ and the number of shots per pseudo-novel class $K$ follow the specific $N$-way $K$-shot setup; the number of pseudo-base classes $M$ is the number of all the remaining base classes (i.e., $M=|\mathcal{C}_\texttt{base}|-N$); the number of shots per pseudo-base class $L=40$. We set the parameter of dropout variational inference $T=10$ as we observed that when $T\geq10$, model performance is very stable.

For baseline methods, we adopt the public resources of GPN (https://github.com/kaize0409/GPN_Few-shot), MetaGNN (https://github.com/ChengtaiCao/Meta-GNN) and G-META (https://github.com/mims-harvard/G-Meta) from the authors and keep the original model structures. We implement APPNP with $10$ steps propagation and the teleport probability $\alpha$ is set as $0.1$ as the authors' recommendation. The steps of propagation for GPRGNN is set as $10$.  All the MLPs are $3$-layer with ReLU activation functions. For all the baseline methods, we adopt the same strategy to search the hidden dimension and learning rate. The sample re-weighting is set as $50:N$ between the novel and base classes respectively, if fine-tuning is needed.

## Running
train_STAGER.py is the entry of the code.

## Reference
If you find this repository useful, please kindly cite the following paper:


```
@inproceedings{DBLP:conf/icdm/XuDW0T22,
  author    = {Zhe Xu and
               Kaize Ding and
               Yu{-}Xiong Wang and
               Huan Liu and
               Hanghang Tong},
  editor    = {Xingquan Zhu and
               Sanjay Ranka and
               My T. Thai and
               Takashi Washio and
               Xindong Wu},
  title     = {Generalized Few-Shot Node Classification},
  booktitle = {{IEEE} International Conference on Data Mining, {ICDM} 2022, Orlando,
               FL, USA, November 28 - Dec. 1, 2022},
  pages     = {608--617},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/ICDM54844.2022.00071},
  doi       = {10.1109/ICDM54844.2022.00071},
  timestamp = {Thu, 02 Feb 2023 14:29:00 +0100},
  biburl    = {https://dblp.org/rec/conf/icdm/XuDW0T22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
