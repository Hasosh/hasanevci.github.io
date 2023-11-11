---
title: "Learning to Simulate Complex Physics with Graph Networks"
date: 2023-11-11
permalink: /posts/2023/11/seminar-blog-post/
excerpt: "This is a blog post summarizing the paper 'Learning to Simulate Complex Physics with Graph Networks' (2020) by Sanchez-Gonzalez, Alvaro, et al. published in the International conference on machine learning. <br/><img src='https://hasosh.github.io/hasanevci.github.io/images/GNS-framework.png'>"
collection: Machine Learning
tags:
  - Graph Neural Networks
  - Message Passing
  - Particle-based System
---

Lorem Ipsum

# Introduction

# Graph Networks

## Graph Network Block

## Common graph network variants

Global attribute prediction <br>
$$
\phi^{u}(\Sigma_{e'}; \Sigma_{v'}; u) := f^{u}(\Sigma_{e'})
$$

Edge update rule variants
$$
\phi^{e}(e_{k}; v_{r_{k}}; v_{s_{k}}; u) := f^{e}(v_{s_{k}}) \\
\phi^{e}(e_{k}; v_{r_{k}}; v_{s_{k}}; u) := v_{s_{k}} + f^{e}(e_{k}) \\
\phi^{e}(e_{k}; v_{r_{k}}; v_{s_{k}}; u) := f^{e}(e_{k}; v_{s_{k}})
$$

Relation Networks global output from pooled edge information
$$
\phi^{e}(e_{k}; v_{r_{k}}; v_{s_{k}}; u) := f^{e}(v_{r_{k}}; v_{s_{k}}) = NN^{e}([v_{r_{k}}; v_{s_{k}}]) \\
\phi^{u} - (\Sigma_{e'}; \Sigma_{v'}; u) := f^{u} - (\Sigma_{e'}) = NN^{u}(\Sigma_{e'})
$$

Deep Sets global output from pooled nodes information
$$
\phi^{v}(\Sigma_{e_{i}}; v_{i}; u) := f^{v}(v_{i}; u) = NN^{v}([v_{i}; u]) \\
\phi^{u} - (\Sigma_{e'}; \Sigma_{v'}; u) := f^{u} - (\Sigma_{v'}) = NN^{u}(\Sigma_{v'})
$$

PointNet update rule with max-aggregation <br>
$$
\sum_{v \rightarrow u} - (V') := \max_{i}(v'_{i})
$$

# Learning to Simulate Complex Physics with Graph Networks 

# Further Applications of Graph Networks
Mesh-Rendering with Particles

# Limitation and Outlook
Limitations: 
Outlook: 

# Conclusion

Literature
------

- A. Sanchez-Gonzalez et al., "Learning to simulate complex physics with graph networks," in Proc. Int. Conf. Machine Learning, PMLR, 2020.
- P. W. Battaglia et al., "Relational inductive biases, deep learning, and graph networks," arXiv preprint arXiv:1806.01261, 2018.
