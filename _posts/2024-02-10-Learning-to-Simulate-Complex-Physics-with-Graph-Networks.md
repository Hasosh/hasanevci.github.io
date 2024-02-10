---
title: "Learning to Simulate Complex Physics with Graph Networks"
date: 2024-02-10
permalink: /posts/2024/02/seminar-blog-post/
# excerpt: "This is a blog post summarizing the paper 'Learning to Simulate Complex Physics with Graph Networks' (2020) by Sanchez-Gonzalez, Alvaro, et al. published in the International conference on machine learning. <img src='https://hasosh.github.io/hasanevci.github.io/images/front-page-stretched.png' style='width:500px'>"
# excerpt: "This is a blog post summarizing the paper 'Learning to Simulate Complex Physics with Graph Networks' (2020) by Sanchez-Gonzalez, Alvaro, et al. published in the International conference on machine learning. <img src='/images/front-page-stretched.png' style='width:500px'>"
collection: Machine Learning
tags:
  - Graph Network
  - Message Passing
  - Particle-based System
  - Complex Physics Modeling
  - Graph-based Simulation
  - Deep Learning in Scientific Computing
toc: true
---

<!-- for the info boxes -->
<style>

  details {
      border: 1px solid #e0e0e0; 
      border-radius: 10px; 
      padding: 5px;
      margin-bottom: 20px;
  }

  details > summary {
      list-style: none;
      cursor: pointer; 
      position: relative; 
      outline: none; 
  }

  details > summary::-webkit-details-marker {
      display: none;
  }

  .info-icon {
      margin-right: 5px;
      color: white;
      background-color: #ADD8E6;
      padding: 2px;
      border-radius: 4px; 
      display: inline-block; 
      font-weight: bold; 
  }

  /* Custom arrow */
  .arrow {
      position: absolute;
      right: 10px; 
      top: 30%;
      transform: translateY(-50%);
      border: solid black;
      border-width: 0 2px 2px 0;
      display: inline-block;
      padding: 3px;
      transform: rotate(45deg);
      -webkit-transform: rotate(45deg);
  }

  /* Rotate arrow when details are open */
  details[open] .arrow {
      transform: rotate(-135deg);
      -webkit-transform: rotate(-135deg);
  }

  .content {
      margin-top: 10px;
  }
</style>

Have you ever wondered about how to predict the behavior of rigid or fluid particles and how we can use the right models for this task? If so, you're in for an interesting journey!

<!-- <div style="text-align: center;">
  <img src='/images/intro.png' style="height:240px">
  <figcaption><p style="color: grey; font-size: smaller;">Inspired by Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/intro.png' style="height:auto; width:70%;">
  <figcaption><p style="color: grey; font-size: smaller;">Inspired by Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

# Introduction
------

{% include toc %}

Welcome to this blog post, where I'll provide an overview and insights into the paper titled ***'Learning to Simulate Complex Physics with Graph Networks'*** authored by *Sanchez-Gonzalez et al.*, which was presented at the International Conference on Machine Learning (ICML) in 2020. In this paper, the authors delve into the exciting realm of simulating complex physical phenomena using graph networks. They explore how this deep learning method can be applied to enhance our understanding and prediction of complex physics, bridging the gap between deep learning and numerical methods.

As a student passionate about using machine learning to solve problems in the sciences, I find this paper particularly fascinating. It showcases the potential for mathematical concepts and deep learning to synergize, leading to significant advancements in our ability to model and simulate intricate physical systems. This post provides a comprehensive exploration of the paper's key findings and contributions in the context of enhancing our understanding of complex physics through innovative neural network architectures. It also draws inspiration from the <a href="https://www.youtube.com/watch?v=8v27_jzNynM" style="color: blue;">author's talk</a>.

Feel free to explore this post in a way that suits your interests and familiarity with the topic. If you're already acquainted with certain aspects of this study, you might choose to jump to specific sections that capture your curiosity or fill gaps in your understanding. This blog is designed to accommodate both comprehensive reading and targeted exploration for seasoned readers.

<span style="font-size: 24px;">Why do we even need to simulate complex physics?</span>

The need for simulating complex physics arises across various scientific and engineering disciplines. These realistic simulators serve as invaluable tools that enable us to gain a profound understanding of intricate physical phenomena. Whether it's designing cutting-edge aerospace technologies, optimizing fluid dynamics in engineering, or unraveling the behavior of biological systems, simulations provide a controlled and accessible environment to explore and experiment. In some situations, they are also very cost-efficient. Consider the example of a car manufacturer assessing vehicle safety. Instead of the costly and resource-intensive process of physically crashing hundreds of cars, simulations offer a more efficient alternative here.

Without such simulations, many scientific studies and engineering projects would face major difficulties. Complex physics simulations offer the means to predict and visualize the behavior of systems that may be difficult or impossible to study directly. They empower us to test hypotheses, predict outcomes, and optimize designs, advancing our knowledge and technological capabilities. However, the traditional methods of creating these simulations present significant challenges and limitations.

<span style="font-size: 24px;">What is the problem of traditional simulators?</span>

While the importance of simulating complex physics is undeniable, the traditional approach to building simulators poses several challenges. These conventional simulators can be exceptionally costly, both in terms of time and resources. Developing a simulator often demands years of painstaking engineering effort, from designing accurate models to implementing computational algorithms.

One of the critical issues is the trade-off between generality and accuracy. Traditional simulators tend to excel in specific, narrowly defined settings, sacrificing the ability to adapt to a broader range of scenarios. Moreover, constructing high-quality simulators necessitates substantial computational resources, making it very hard to scale up.

Even the best traditional simulators may fall short due to inherent limitations. Insufficient knowledge of the underlying physics or the complexities of approximating critical parameters can lead to inaccuracies. As a result, there is a compelling need for alternative approaches that can overcome these challenges and revolutionize the way we simulate complex physics. One such alternative is the utilization of machine learning, which has the potential to train simulators directly from observed data. It presents a transformative solution with **learned simulators**, which often have flexible frameworks capable of adapting to a wide range of problems. These simulators are not static; they can be continually refined for better performance and become more accurate as more data is fed into them, allowing for precise predictions and replication of physical phenomena. Among these, **graph networks (GNs)** stand out for their potential in learned simulators, pointing to a future where simulating complex physics is more efficient and accurate.

<span style="font-size: 24px;">How can you simulate complex physics with graph networks?</span>

Simulating complex physics with GNs involves representing physical systems as graphs, with nodes and edges denoting entities and their interactions. GNs excel in capturing the complex relationships and dynamics within these structured data. By training GNs on observed data, they can learn the underlying physics and predict how physical properties evolve over time. This adaptability makes GNs a powerful tool for simulating a wide range of complex physical phenomena, from fluid dynamics to structural mechanics, without relying on predefined mathematical models. In essence, GNs revolutionize complex physics simulation by leveraging the flexibility and capacity of machine learning within the framework of graph-like representations.

The paper "Learning to Simulate Complex Physics with Graph Networks" by Sanchez-Gonzalez et al. significantly advances the application of GNs in simulating complex physics. In their work, the authors propose a novel Graph Network-based Simulation (GNS) framework that utilizes the graph-like nature of physical systems. The GNS framework leverages GNs to represent physical entities and their interactions as nodes and edges within the graph, enabling the network to grasp the intricate relationships and dependencies that define complex physical phenomena. 


# Related work
------

Before delving into their approach's intricacies, it's essential to explore prior work relevant to the context of this paper. The key references are summarized in the following figure.

<!-- <div style="text-align: center;">
  <img src="/images/related-work.png" style="height: 250px;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Created by myself</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src="https://hasosh.github.io/hasanevci.github.io/images/related-work.png" style="height:auto; width:90%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Created by myself</p></figcaption>
</div>

<span style="font-size: 24px;">Simulators</span>

In the realm of simulators, several influential contributions have paved the way for data generation and performance comparison in their research. A widely recognized method for simulating fluids is "Smoothed Particle Hydrodynamics" (SPH) (Monaghan, 1992). SPH involves assessing pressure and viscosity forces around individual particles, leading to updates in their velocities and positions. Other techniques, such as "Position-Based Dynamics" (PBD) (Müller et al., 2007) and "Material Point Method" (MPM) (Sulsky et al., 1995), are more apt for handling deformable materials and interactions. PBD, for example, deals with issues like incompressibility and collision dynamics by resolving pairwise distance constraints among particles and predicting their positional changes directly.

<span style="font-size: 24px;">Graph networks</span>

The field of GNs plays a pivotal role in their work, forming the core of the GNS framework. Two noteworthy contributions in this domain include Battaglia et al.'s (2018) exploration of "Relational Inductive Biases, Deep Learning, and Graph Networks", offering valuable insights into the fundamentals and development of GNs. Furthermore, Gilmer et al. (2017) introduced "Neural Message Passing for Quantum Chemistry" which laid the groundwork for the message passing mechanism within GNS.

<span style="font-size: 24px;">Performance comparison</span>

Li et al. (2018) introduced "DPI", which they applied to fluid dynamics, and Ummenhofer et al. (2020) introduced "Continuous Convolution" (CConv) as a non-graph-based method for simulating fluids. These works serve as foundational references for evaluating their framework's performance against existing fluid simulation methods.

To summarize, previous work in simulators, particularly SPH and PBD for fluid dynamics, along with the contributions of Li et al. (2018) and Ummenhofer et al. (2020), significantly shape and inform their approach. Subsequent sections will delve into the specific methodology, highlighting their unique contributions and advancements in the field.

The following chapters assume knowledge about some terminology, a foundational understanding of graph theory, and familiarity with the Euler integrator. If you require a refresher on these topics, feel free to open the blue info boxes.

# Graph network (GN)
------

Graph Networks (GNs) are a class of neural network architectures designed to handle data with complex relational structures, such as graphs. The fundamental idea behind GNs is to enable neural networks to effectively reason about and process data points that are interconnected, where the relationships among these data points are critical for understanding the underlying patterns.

<details>
  <summary>
    <span class="info-icon">&#x2139;</span> <!-- Unicode for information sign -->
    <b>Graph Definition</b>
    <span class="arrow"></span>
  </summary>
  <div class="content">
  In the context of GNs, a graph is formally defined as a 3-tuple $G = (u, V, E)$, where:
  <ul>
    <li> $V$ is the set of nodes (of cardinality $N^v$), denoted as $V = \{v_i\}_{i=1:N^v}$, where each $v_i$ represents an individual node within the graph. For example, in a social network, $V$ could represent individuals, and in a transportation network, $V$ could represent locations or cities.</li>
    <li> $E$ represents the set of edges (of cardinality $N^e$), which defines the connections or relationships between the nodes. It can be defined as $E = \{e_k, r_k, s_k\}_{k=1:N^e}$, indicating that edges are triplets consisting of attributes $e_k$, receiver nodes $r_k$, and sender nodes $s_k$. In a social network, the edges might represent friendships between individuals, while in a computer network, they could signify network connections between devices.</li>
    <li> $u$ represents global properties or attributes associated with the entire graph, i.e. properties that may apply to all nodes and edges. These global properties can vary widely depending on the context of the graph. For instance, in a financial network, $u$ could represent the overall economic stability of a region, and in a recommendation system, $u$ might represent user preferences or trends that apply to the entire user base.</li>
  </ul>
  The following figure depicts a graph $G=(u, V, E)$ with some global property $u$ and $V,E$ defined as follows:
  $$
    \begin{align*}
    V = \, &\{v_1, v_2, v_3, v_4, v_5\} \\ \nonumber
    E = \, &\{(e_1, v_1, v_4), (e_2, v_2, v_1), (e_3, v_2, v_3), (e_4, v_2, v_5), (e_5, v_3, v_1), (e_6, v_3, v_3), \\ 
    \: &(e_7, v_4, v_1), (e_8, v_4, v_5), (e_9, v_5, v_4), (e_{10}, v_5, v_4)\} \\ \nonumber
    \end{align*}
  $$
  <!-- <div style="text-align: center;">
    <img src="/images/graph-3tuple.png" style="height: 200px;">
    <figcaption><p style="color: grey; font-size: smaller;">Source: Battaglia et al. (2018)</p></figcaption>
  </div> -->
  <div style="text-align: center;">
    <img src="https://hasosh.github.io/hasanevci.github.io/images/graph-3tuple.png" style="height:auto; width:50%;">
    <figcaption><p style="color: grey; font-size: smaller;">Source: Battaglia et al. (2018)</p></figcaption>
  </div>
  </div>
</details>

The central concept of GNs is the ability to propagate information across the edges and nodes of a graph. Unlike traditional neural networks that operate on fixed-sized inputs, GNs dynamically adapt to varying graph structures. They achieve this through a series of computation steps that involve message-passing and aggregation. GNs process information at different levels, starting from individual edges and nodes, then aggregating information at higher levels, ultimately leading to a global understanding of the entire graph. The power of GNs lies in their capacity to capture relational inductive biases, making them versatile for various tasks.

The following sections about graph networks draw from the insights of Battaglia et al.'s (2018) paper *"Relational inductive biases, deep learning, and graph networks"*.

## GN block

The primary computational unit within the GN framework is the GN block, which can be described as a "graph-to-graph" module. This module accepts a graph comprised of edge, node, and global elements as its input, conducts computations based on the graph's structure, and produces another graph as its output.

A GN block comprises three **update functions** denoted as $$\phi$$ and three **aggregation functions** denoted as $$\rho$$:

$$
\begin{align*}
  e'_k &= \phi^e(e_k, v_{r_k}, v_{s_k}, u) \qquad & \bar{e}'_i &= \rho^{e \to v}(E'_i)\\ 
  v'_k &= \phi^v(\bar{e}'_i, v_i, u) \qquad & \bar{e}' &= \rho^{e \to u}(E')\\
  u' &= \phi^u(\bar{e}', \bar{v}', u) \qquad & \bar{v}' &= \rho^{v \to u}(V')
\end{align*}
$$

where 

$$
\begin{align*}
  E'_i &= \{(e'_k, r_k, s_k)\}_{r_k = i, k = 1:N^e} \\ \nonumber
  V' &= \{v'_i\}_{i = 1:N^v} \\
  E' &= \bigcup_i E'_i = \{(e'_k, r_k, s_k)\}_{k = 1:N^e}
\end{align*}
$$

These equations describe the operations within the GN block. 

The $$\phi^e$$ function is applied to all edges to calculate per-edge updates, $$\phi^v$$ is applied to all nodes to compute per-node updates, and $$\phi^u$$ is applied once to perform a global update. The $$\rho$$ functions, on the other hand, operate on sets of elements and reduce them to single values that represent aggregated information. Importantly, the $$\rho$$ functions must remain **invariant to permutations** of their inputs and should be able to handle variable numbers of arguments, such as elementwise summation, averaging, or finding the maximum value, among other possibilities.

When a graph, denoted as G, is provided as input to a Graph Neural (GN) block, the computations within the block follow a specific sequence. Most of the time, they are going from the edge level to the node level, and finally to the global level. 

This sequence of computations is visualized in the following figure, where aggregations are already done implicitly.

<!-- <div style="text-align: center;">
  <img src='/images/GN-updates.png'>
  <figcaption><p style="color: grey; font-size: smaller;">Source: Battaglia et al. (2018)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/GN-updates.png' style="height:auto; width:90%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Battaglia et al. (2018)</p></figcaption>
</div>

It's worth noting that while this sequence of steps is assumed here, the order is not strictly enforced. Depending on the specific application and requirements, it is possible to reverse the order of update functions to proceed from the global level to per-node and per-edge updates, offering flexibility in the GN block's design.


## Composing multiple GN blocks

GNs revolutionize architecture composition, enabling the seamless assembly of multiple GN blocks into sophisticated structures. This makes the creation of complex, interconnected models possible that can be used for a wide range of tasks. Here's an overview of how multiple GN blocks can be composed:

1. **Basic Composition**: In its simplest form, two GN blocks, $GN_1$ and $GN_2$, can be combined by passing the output of the first block as input to the second, resulting in $G' = GN_2(GN_1(G))$. This allows for the sequential application of GN blocks.

2. **Arbitrary Composition**: Multiple GN blocks can be composed arbitrarily, enabling the creation of more intricate architectures. These blocks can either be **unshared** (having different functions and parameters, akin to layers in a Convolutional Neural Network), $GN_1 \neq GN_2 \neq \ldots \neq GN_M$, or **shared** (using the same functions and parameters, resembling an unrolled Recurrent Neural Network), $GN_1 = GN_2 = \ldots = GN_M$.

3. **Encode-Process-Decode Architecture**: A common architecture design involves the encode-process-decode configuration. In this setup, an encoder block $GN_{enc}$ takes an input graph $G_{inp}$ and transforms it into a latent representation $G_0$. A core block $GN_{core}$ (shared or unshared) is then applied $M$ times to generate $G_M$. Finally, an output graph $G_{out}$ is decoded by a GN block $GN_{dec}$. This design is suitable for tasks such as predicting the dynamics of a system over time. Note that this is also the architecture that is used in the GNS framework that we will see later.

4. **Recurrent GN-Based Architectures**: Recurrent GN-based architectures maintain a hidden graph $G_{hid}^t$, which takes an observed graph $G_{inp}^t$ as input and produces an output graph $G_{out}^t$ at each step. This architecture is useful for predicting sequences of graphs, such as the trajectory of a dynamical system over time.

You can see the latter three composition architectures in (a), (b), and (c) of the next figure, respectively.

<!-- <div style="text-align: center;">
  <img src='/images/GN-types.png'>
  <figcaption><p style="color: grey; font-size: smaller;">Source: Battaglia et al. (2018)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/GN-types.png' style="height:auto; width:90%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Battaglia et al. (2018)</p></figcaption>
</div>

## Message passing

Message passing in GNs is fundamentally the **iterative application of a GN block** (Gilmer et al., 2017). This process is named 'message passing' due to the way information, or "messages" are exchanged between nodes in the graph. Each application of the GN block facilitates the communication and state updates of these nodes, based on the information received from their connections. This mechanism is central to the functionality of GNs, enabling the network to effectively propagate and process information across the graph structure.

Note that in the context of GNs, **message passing** and **message passing neural network** are often used interchangeably because the update functions $\phi$ introduced in the GN block section are usually implemented as neural networks.

An example of the message passing process is visualized in the following figure:

<!-- <div style="text-align: center;">
  <img src='/images/message-passing-example.png'>
  <figcaption><p style="color: grey; font-size: smaller;">Source: Battaglia et al. (2018)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/message-passing-example.png' style="height:auto; width:90%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Battaglia et al. (2018)</p></figcaption>
</div>

In this example of message passing, each row illustrates the information diffusion starting from a particular node. The top row represents a scenario where the node of interest is in the upper right, while the bottom row depicts the node of interest in the bottom right. Shaded nodes indicate how far information from the original node can propagate in $m$ steps of message passing, and bolded edges indicate which edges allow information to traverse. It's important to note that during the full message passing procedure, this propagation of information happens simultaneously for all nodes and edges in the graph, not just the two nodes and edges shown in this illustration.

Crucially, if we exclude the global feature (which aggregates information across the entire graph), the accessible information to a node after $m$ steps is limited to the nodes and edges within an $m$-hop distance. This aspect of message passing effectively decomposes complex computations into smaller, more manageable steps. Moreover, these steps can represent sequential time periods, allowing the GN to model temporal sequences and dynamics. Thus, message passing in GNs not only facilitates information exchange but also enables a nuanced breakdown of intricate computations and the modeling of sequential processes.

# Graph Network-based Simulators (GNS) framework
------

The Graph Network-based Simulators (GNS) framework is a machine learning framework designed for simulating complex physical systems involving fluids, rigid solids, and deformable materials interacting with each other. It represents the state of the system as nodes in a graph and computes dynamics using learned message-passing techniques. The framework demonstrates the ability to generalize across different initial conditions, timesteps, and particle quantities during testing, showing robustness to hyperparameter variations. 

The next sections will cover the main idea behind GNS, explain what a learnable simulation means, and describe the framework of the model.

## Main idea of GNS

The main idea of GNS can be explained with the help of the figure shown below:

<!-- <div style="text-align: center;">
  <img src='/images/GNS_idea.png' style="width: 600px;">
  <figcaption><p style="color: grey; font-size: smaller;">Inspired by Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/GNS_idea.png' style="height:auto; width:80%;">
  <figcaption><p style="color: grey; font-size: smaller;">Inspired by Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

Given a constellation of some material, e.g. a water block, and an environment (see (1)). This block serves as the starting point from which we select a set of representative particles. Furthermore, particles that are close to each other are connected with the idea that they can influence each other more than distant particles. Using the *particles as nodes* and their *connections as edges*, they are transformed into a mathematical structure—a *graph*—that allows for computational manipulation and operation (see (2)).

The core of this simulation is what the paper terms a **'learned simulator'**, which is the GNS framework in this case. Through this simulator, we process the graph of water particles, and it yields an updated graph that encapsulates the state of the system after a time increment $\Delta t$ (see (3)). This step is where the simulator's innovative capabilities come into play, as it employs weights, denoted as $\theta$, that have been refined through deep learning techniques. These weights are crucial because they govern how the simulator interprets and predicts the fluid dynamics of the water block over time.

The final phase of the simulation involves reconstructing the updated graph into a visually coherent image that represents the simulated state of the water (see (4)). This rendering step brings the mathematical predictions to life, providing a tangible output from the theoretical framework.

The transition from the initial water block to the selection of particles, and subsequently from the simulation to the reconstruction, might seem direct. However, the true ingenuity of this research lies in the application of the learned simulator. This tool is not merely a static piece of software but a dynamic model that evolves and learns, which is precisely why this paper is a significant stride forward in the field of fluid simulation.

## What is a learnable simulation?

In the following discourse, we shall present a precise formal definition of how a learnable simulation can be characterized. 

<details>
  <summary>
    <span class="info-icon">&#x2139;</span> 
    <b>Terminology</b>
    <span class="arrow"></span>
  </summary>
  <div class="content">
  In order to explain the components and functionality of the GNS framework, it is essential to introduce certain terminology.
  <ul>
    <li> <b>Environment</b>: refers to the system or environment being simulated or modeled. It represents the state of the physical or conceptual system under consideration at a specific point in time.</li>
    <li> <b>Physical Dynamics</b>: refers to the principles and laws of physics that govern how the state of a system changes over time. These physical dynamics encompass the fundamental rules that describe the behavior of objects and systems in the real world, such as the laws of motion, conservation of energy, and other physical principles.</li>
    <li> <b>Dynamics Information</b>: refers to data or knowledge that characterizes how the current state of the system is changing over time. It encompasses the quantitative and qualitative details regarding the evolution of the system's state as it progresses from one time step to the next. More specifically, "dynamics information" is the information that simulators gather and process to understand and predict how the system's state will transition from its current state to a future state. This information can include variables like velocities, accelerations, forces, or any other parameters that describe the rate and direction of change in the system's properties.</li>
  </ul>
  </div>
</details>

We assume that $X^t \in X$ represents the state of the system at time $t$. Employing physical dynamics over $K$ discrete time steps results in a **trajectory of states** denoted as 

$$
  X^{t_0:K} = (X^{t_0}, \ldots, X^{t_K}) \nonumber
$$

A **simulator**, $s: \mathcal{X} \rightarrow \mathcal{X}$, models these dynamics by mapping preceding states to consequent future states in a causal manner. We refer to this simulated sequence as 

$$
  \tilde{X}^{t_0:K} = ({X}^{t_0}, \tilde{X}^{t_1}, \ldots, \tilde{X}^{t_K}) \nonumber
$$

which is computed iteratively as 

$$
  \tilde{X}^{t_{k+1}} = s(\tilde{X}^{t_k}) \nonumber
$$

for each time step. Simulators calculate dynamic information that represents how the current state is evolving and use this information to update the current state to predict future states. 

A **trainable simulator**, denoted as $s_{\theta}$, calculates dynamic information using a parameterized function approximator, $d_{\theta}: \mathcal{X} \rightarrow \mathcal{Y}$, where $\theta$ represents the parameters that can be optimized based on a specific training objective. The $Y \in \mathcal{Y}$ signifies the **dynamic information**, the meaning of which is defined by the update mechanism. The update mechanism can be perceived as a function that takes the state $\tilde{X}^{t_k}$ and utilizes $d_{\theta}$ to forecast the subsequent state 

$$
  \tilde{X}^{t_{k+1}} = \text{Update}(\tilde{X}^{t_k}, d_{\theta}) \nonumber
$$

In this context, we assume a straightforward update mechanism, specifically an **Euler integrator**, and $\mathcal{Y}$ represents accelerations. However, one can also use more sophisticated integrators.

<details>
  <summary>
    <span class="info-icon">&#x2139;</span> 
    <b>Euler integrator</b>
    <span class="arrow"></span>
  </summary>
  <div class="content">
  The Euler integrator is a simple numerical method used for solving ordinary differential equations (ODEs). It approximates the next state of a system by using the current state and the derivatives of the state variables. The formula for the Euler integrator is:
$$
  X_{k+1} = X_k + h \cdot f(X_k, t_k) \nonumber
$$
  where:
  <ul>
    <li> $X_k$ represents the current state of the system at time $t_k$.</li>
    <li> $X_{k+1}$ is the estimated next state at time $t_{k+1}$.</li>
    <li> $h$ is the time step, which determines the size of the time intervals between calculations.</li>
    <li> $f(X_k, t_k)$ represents the derivative of the state variables at the current time $t_k$.</li>
  </ul>

  In essence, the Euler integrator updates the state by taking a small step in time $h$ in the direction of the derivative $f(X_k, t_k)$ at the current time $t_k$. While it is a straightforward method, it may introduce errors, especially when dealing with complex or highly nonlinear systems, and more advanced numerical integration methods are often used for greater accuracy. <br><br>
  </div>
</details>

## Model framework

### Input and output representation

The state of the physical system can be represented via a tuple of $N$ particles $x_i$:

$$
  X = (x_0, \ldots, x_N) \nonumber
$$

Each particle $x_i$ in the physical system can be expressed as a vector. This vector contains the particle's position $p_i^{t_k}$, the particle's 5 previous velocities $\dot{p}_i^{t_k}$, and features $f_i$ that describe static material properties, e.g. storing what type of particle it is (water, sand, goop), storing whether it is flexible or not, and storing whether the particle is at the boundary of the environment. The following figure summarizes the input vector representation of a particle.

<!-- <div style="text-align: center;">
  <img src='/images/input-representation.png' style="height: 300px;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Created by myself</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/input-representation.png' style="height:auto; width:60%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Created by myself</p></figcaption>
</div>

The targets for supervised learning in this context are the **average acceleration** of each particle, symbolized as $$\ddot{p}_i$$. To get the position and velocity of each particle from the acceleration, a semi-implicit **euler integration** update mechanism is used:

$$
  \dot{p}^{t_{k+1}} = \dot{p}^{t_{k}} + \Delta t \cdot \ddot{p}^{t_{k}} \\ \nonumber
  p^{t_{k+1}} = p^{t_{k}} + \Delta t \cdot \dot{p}^{t_{k+1}} \nonumber
$$

By assumption it holds that $\Delta t = 1$ ($\Delta t$ is fixed). Also, the authors argue that they did not use forward Euler ($p^{t_{k+1}} = p^{t_{k}} + \Delta t \cdot \dot{p}^{t_{k}}$) because they wanted the acceleration $\ddot{p}^{t_{k}}$ to directly influence the new position $p^{t_{k+1}}$.

### Architecture
The whole architecture of the GNS framework is depicted in the following figure:

<!-- <div style="text-align: center;">
  <img src='/images/GNS-framework.png'>
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/GNS-framework.png' style="height:auto; width:90%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

We will explain the architecture step by step, going from the coarse level to the granular level (denoted by the letters a-e). Given the state $X^{t_k}$, we perform one simulation step with our learned simulator $s_{\theta}$ to transition to the next state $X^{t_{k+1}}$ (see (a)). Note that $s_{\theta}$ is the GNS framework and that a simulation step can also be regarded as an update of a state. As already mentioned, we use an Euler integration scheme to do this update. 

To perform a state update, we need to learn a function approximator $d_{\theta}$. This is realized via an **encode-process-decode** paradigm (see (b)), similar to the structure outlined in the section discussing graph network variants. For each state transition in this framework, the encoder is employed at the outset of a simulation to transform the initial state into a graph representation. This is followed by the repeated application of the processor throughout the simulation, which iteratively updates the graph to reflect the evolving dynamics of the system. Finally, at the end of the simulation, the decoder is utilized to translate the final graph state into per-particle average accelerations.

In the following, we explain the encoder, processor and decoder in more detail.

<span style="font-size: 24px;">Encoder</span>

The **ENCODER** $$\mathcal{X} \to \mathcal{G}$$ takes a particle-based state $X$ as input and maps it to a latent graph $G^0$, where $$G = (V, E, u), v_i \in V, e_{i,j} \in E$$ (see (b,c)). 

$$
  G^0 = ENCODER(X) \nonumber
$$

The **node embeddings** $v_i$ represent learned functions of the particles' states, denoted as 

$$
  v_i = \varepsilon^v(x_i) \nonumber
$$

The **edge embeddings** $e_{i,j}$ are functions that have been learned from the pairwise characteristics of the associated particles

$$
  e_{i,j} = \varepsilon^e(r_{i,j}) \nonumber
$$ 

These characteristics $r_{i,j}$ can include factors like the displacement between their positions, the spring constant, and so on.

In total, the encoder constructs a locally connected latent graph $G_0$. This means that it adds edges between particles that are within a specific **"connectivity radius"**, denoted as $R$. This radius reflects local interactions among the particles and remains constant for all simulations at the same resolution. Standard nearest neighbor algorithms were employed for this purpose.

Beyond that, the authors tested two different variants of encoders: **absolute** and **relative encoder**. The absolute encoder uses absolute position information of the particles, thus the input to $\varepsilon^v$ is the $x_i$ described above in the input representation. As absolute positions are stored for each input particle, the edge embeddings $\varepsilon^e$ do not carry any valuable information here and were consequently discarded for the absolute encoder. On the other hand, the relative encoder uses relative position information of the particles, i.e. the relative distance to each neighbor of a particle. Therefore, the node embeddings $\varepsilon^v$ ignored position information $p_i$ inside $x_i$ by masking it out. Instead, the edge embeddings utilized the relative positional displacement and its magnitude $r_{i,j} = [(p_i - p_j), \| p_i - p_j\|]$. Lastly, in both version, global properties were always appended to $x_i$ prior to inputting them to $\varepsilon^v$.

<span style="font-size: 24px;">Processor</span>

The **PROCESSOR** $$\mathcal{G} \to \mathcal{G}$$ maps a latent graph $G$ to another latent graph $G$, by performing interactions among nodes through $M$ steps of learned **message-passing** (thus $M$ GN blocks). This results in a series of updated latent graphs, represented as $G = (G^1, ..., G^M)$, where each $G^{m+1}$ is generated based on the previous graph $G^m$ using the function $G^{m+1} = GN^{m+1}(G^m)$. Ultimately, the processor yields the final graph

$$
  G^M = PROCESSOR(G^0) \nonumber
$$

which is obtained by applying the processor to the initial graph $G^0$ (see (b,d)). The Processor furthermore uses residual connections between the input and output latent nodes and edge attributes.

<span style="font-size: 24px;">Decoder</span>

The **DECODER** $$\mathcal{G} \to \mathcal{Y}$$ takes the final latent graph $G^M$ as input and from this it extracts the average acceleration from the nodes

$$
  Y = DECODER(G^M) \nonumber
$$

To extract acceleration for a single node $v_i^M$, a learned function $\delta^v$ is applied

$$
  y_i = \delta^v(v_i^M) \nonumber
$$

where $\delta^v$ is learned through an MLP (see (b, e)).

Note that the output of the decoder is the per-particle average acceleration, so the output is not a graph and is also not the final positions of the particles! Following the decoder step, the future positions and velocities are computed using an **Euler integrator**. Therefore, $y_i$ represents accelerations, denoted as $\ddot{p}_i$, which can have either 2D or 3D dimensions, depending on the specific physical domain.

Also note that we do not learn a decoder function $\delta^e$ for the edges, since it is not required for determining the final accelerations of the particles. The edges solely serve the purpose of passing messages from one node to another.

<span style="font-size: 24px;">Multilayer Perceptron (MLP)</span>

Now one question might be: Well how exactly are these embedding functions $\varepsilon^v$ and $\varepsilon^e$ created or learned for the nodes and edges, respectively? Also, how are node and edge embeddings updated in the processor and how is the decoding function $\delta^v$ learned?

For this purpose, multiple **Multilayer Perceptrons (MLP)** are trained separately i.e. for $\varepsilon^v$, $\varepsilon^e$ and $\delta^v$, a MLP separately. For $\varepsilon^v$ and $\varepsilon^e$, they encode the node features and edge features into latent vectors $v_i$ and $e_{i,j}$ of size 128.

All MLPs used for the encoder, processor, and decoder have the following architecture (in this order):

- two hidden layers with ReLU activations
- one output layer with no activation
- one LayerNorm layer (exception: output decoder)

# Experiments and Results
------

The code and data of the GNS framework can be found at this <a href="https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate" style="color: blue;">GitHub repository</a>.

The research on the GNS model reveals several key findings. 
1. Firstly, the GNS model excels in learning accurate and high-resolution **simulations of diverse materials**, including fluids, deformables, and rigid solids. This versatility is crucial in understanding and predicting the behavior of different physical systems, from everyday substances (e.g. water and sand) to complex industrial materials (e.g. goop).
2. Secondly, it demonstrates a remarkable ability to **generalize its learning** to handle much more extensive and complex scenarios than those it was initially trained on (e.g. adding obstacles or mixing materials). 
3. In a comparative analysis, the **GNS model outperforms two recent similar models**, proving to be simpler, more universally applicable, and more accurate.

We will discuss these key findings in more detail in the following sections.

## Data

In their experiments, various simulators were employed to generate trajectories, each tailored to specific domains. These simulators included those based on Smoothed Particle Hydrodynamics (**SPH**), Material Point Method (**MPM**), and Particle-Based Dynamics (**PBD**) engines. SPH is often employed for simulating water, MPM for scenarios involving sand and goop, and PBD is only used for the Boxbath-domain (see later). Additionally, they created a high-resolution 3D water scenario akin to existing work. Their datasets consisted of 1000 training, 100 validation, and 100 test trajectories, each running for 300-2000 timesteps, with the duration tailored to meet stability requirements specific to different materials.

The datasets only consisted of position vectors $p_i$. As the GNS framework needs velocities $\dot{p}$ for its input features into the encoder and accelerations $\ddot{p}$ as ground truth data for the decoder, they are computed via **finite differences**:

$$
  \begin{align*}
  &\dot{p}^{t_{k}} \equiv p^{t_{k}} - p^{t_{k-1}} \\ \nonumber
  &\ddot{p}^{t_{k}} \equiv \dot{p}^{t_{k+1}} - \dot{p}^{t_{k}} \\[5px] \nonumber
  \implies &\ddot{p}^{t_{k}} \equiv p^{t_{k+1}} - 2p^{t_{k}} + p^{t_{k-1}} \nonumber
  \end{align*}
$$

Note that the constant $\Delta t$ is omitted for simplicity.

For more detailed information about the dataset, readers are directed to the accompanying figure.

<!-- <div style="text-align: center;">
  <img src='/images/dataset.png' style="width: 600px;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/dataset.png' style="height:auto; width:90%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

## Training

This chapter presents an in-depth examination of the critical training methodologies for the complex simulations: 

1. **Training Noise**: Models are trained using ground-truth one-step data, which lacks accumulated noise found in real-world scenarios. When generating rollouts by feeding the model with its own noisy predictions, the out-of-training-distribution inputs can lead to substantial errors and **error accumulation**. To address this, the model's input velocities are corrupted with **random-walk noise** $$\mathcal{N(\mu_v, \sigma_v=0.0003)} $$ during training, making the training distribution closer to that seen during rollouts. The effectiveness of this specific noise level was confirmed through an ablation study, which we will see later in the chapter 'Key architectural choices'.

2. **Normalization**: All input and target vectors are normalized elementwise to have zero mean and unit variance. This normalization is performed using online statistics computed during training, improving training speed but not significantly affecting the final performance.

3. **Loss Function and Optimization**: The loss function is based on **one-step** $L_2$ **loss**, calculated on predicted per-particle accelerations from sampled particle state pairs $(x_i^{t_k}, x_i^{t_{k+1}})$: $ L(x_i^{t_k}, x_i^{t_{k+1}}; \theta) = \|d_{\theta}(x_i^{t_k}) - \ddot{p}_i^{t_k} \|^2 $.

4. **Hyperparameters Consistency**: Across the various experiments, a consistent set of hyperparameters is applied. This includes a **relative encoder** variant, **unshared GN parameters** in the processor, and **10 steps of message-passing**. Again, these hyperparameters are validated through an ablation study.

Model parameters are optimized using the Adam optimizer with a small mini-batch size. Training involves up to 20 million gradient update steps with a gradual learning rate decay. This approach maintains consistency across datasets and facilitates fair comparisons across settings.

Model performance is evaluated during training by conducting full-length rollouts on held-out validation trajectories. The training stops when negligible decreases in Mean Squared Error (MSE) are observed. The training duration varies depending on the dataset complexity, ranging from a few hours for simpler datasets to up to a week for more complex ones.

In summary, the training process for complex simulation systems involves strategies such as adding training noise, normalization, and common optimization procedures. Regular evaluation and stopping criteria ensure the models achieve the desired performance.

## GNS' generalization capability: from basic to complex experiments

In the following, we will see an incremental approach, where the GNS model is applied to easy environments first and to more complex environments later on.

The carried out training experiments are summarized in the following figure:

<!-- <div style="text-align: center;">
  <img src='/images/training-experiments.png' style="height: 400px;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/training-experiments.png' style="height:auto; width:80%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

First, they trained a separate GNS model for each material (see (a-c)). As can be seen, the GNS model simulations of goop, water and sand is almost indistiguishable from the ground truth. Then, the authors increased the difficulty of the environment by adding obstacles to the environment or by mixing different materials in the same environment (see (d-e)). Here, they trained one GNS model in environments with obstacles and multiple materials. The GNS model was capable of simulating materials interacting with complicated static obstacles such as ramps. What stands out even more is that it also simulated the interaction between different materials (e.g. water, sand and goop) quite accurately. This effectively means that the model learned the product space of different interactions (e.g. water-water, water-sand, water-goop, ...).

Next, we delve into the GNS model's extraordinary capability for **generalization**, a step up in complexity and a true testament to its adaptability. The model, already proficient in handling basic environments and interactions, was now tested in scenarios far beyond its initial training parameters. These more complex outside training experiments are summarized in the next figure:

<!-- <div style="text-align: center;">
  <img src='/images/outside-training-experiments.png' style="height: 400px;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/outside-training-experiments.png' style="height:auto; width:65%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

In a remarkable demonstration of this generalization, the model, initially trained on the *WATERRAMPS* scenario (described before), was subjected to a variety of novel and challenging environments (see (f-h)). For instance, in one scenario, instead of having a static body of water, an "inflow" condition was introduced (see (f)). Here, water particles were continuously added during the simulation, significantly increasing the complexity of the task. This dynamic addition of particles resulted in scenarios where the model had to manage up to 28k particles, compared to the 2.5k particles in its training set. Yet, the model excelled, accurately predicting highly chaotic dynamics that were never encountered during its training. The rollouts from these simulations, visually similar to the ground truth, underscored the model's robust generalization capabilities.

But the testing did not stop there. The model's adaptability was further stretched in a much larger domain featuring multiple inflows over a complex arrangement of slides and ramps (see (h)). Here, the test domain was 32 times larger than the training area, and the number of particles increased to 85k. Despite these vast differences, the model's performance remained very accurate, handling long simulations that extended eight times the duration of its training scenarios. In a final, rigorous test, the model was applied to a custom domain with varied material inflows and shapes. This not only proved the model's capacity to understand and simulate frictional behavior between different materials but also its ability to adapt to completely novel shapes and dynamics, a feature crucial for practical real-world applications.

Through these incremental steps, from basic material simulations to handling multi-material interactions and then to adapting to vastly different and more complex environments, the GNS model has shown an impressive range of capabilities. It's not just a model that simulates physical processes; it's a model that learns, adapts, and excels, even in situations far removed from its initial training grounds.

A detailed list of all the experimental domains and their further specifications such as node count, sequence length and the 1-step / rollout accuracy (as MSE) can be seen in the following table:

<!-- <div style="text-align: center;">
  <img src='/images/experiment-domains.png' style="width: 400px;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/experiment-domains.png' style="height:auto; width:50%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

If you want to see pretty rollout videos of the experiments above, click <a href="https://sites.google.com/view/learning-to-simulate" style="color: blue;">here</a>.

## Key architectural choices

In their study, the authors conducted a comprehensive analysis of the architectural choices within their GNS model to assess their impact on performance. While various hyperparameters, such as the number of MLP layers and encoder/decoder functions, were examined, it was found that these had minimal effects on performance.

However, several factors were identified that significantly influenced the GNS model's performance:

1. **The number of message-passing steps** $$M$$ was observed to have a substantial impact. Increasing M improved performance in both one-step and rollout accuracy, allowing the model to capture longer-range and more complex interactions among particles. It was recommended to choose the smallest M that met the desired performance criteria, considering computational time.

2. The choice between **shared and unshared parameters in the processor** had a noticeable effect. Unshared parameters, despite resulting in more parameters, yielded better accuracy, especially for rollouts. Shared parameters introduced a strong inductive bias similar to recurrent models, whereas unshared parameters resembled a deep architecture.

3. **The connectivity radius** $$R$$ was another influential factor. Larger R values led to lower error rates as they facilitated longer-range communication among nodes. However, larger R values required more computational resources and memory. Therefore, it was advisable to use the minimal R that met the desired performance criteria.

4. **The scale of noise added to inputs during training** had an impact on rollout accuracy. Intermediate noise scales resulted in the best rollout accuracy, in line with the motivation for introducing noise. However, one-step accuracy decreased with increasing noise scale, as noise made the training distribution less similar to the uncorrupted distribution used for one-step evaluation.

5. The choice between the **relative and absolute encoder** showed that the relative version was superior. This preference likely arose from the fact that the underlying physical processes being learned were invariant to spatial position, aligning with the inductive bias imposed through the relative encoder.

These decisions were guided by the experimental results, as illustrated in following figure, which depicts the effect of different ablations against their model on one-step and rollout errors. The bars represent the median seed performance averaged across the entire *GOOP* test dataset, with error bars displaying lower and higher quartiles for the default parameters.

<!-- <div style="text-align: center;">
  <img src='/images/ablation-study.png'>
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/ablation-study.png' style="height:auto; width:90%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

## Comparison with other models

In their comparison analysis, the researchers contrasted their GNS model with two recent particle-based fluid simulation approaches.

The GNS model was applied to the *BOXBATH* domain, known from the **dynamic particle interaction network (DPI) from Y. Li et al. (2018)**. Unlike DPI, which required distinct architectures for different simulations and a specialized hierarchical mechanism for rigid particles, the GNS model managed to simulate a rigid box floating in water without any modifications. The GNS's relative encoder and training noise were sufficient to maintain the stiff relative displacements among rigid particles, simplifying the process.

The GNS model was also compared to the convolutional network **(CConv) from Ummenhofer et al. (2020)**, which is specifically tailored for fluid dynamics with distinct features like SPH-like local kernels and different sub-networks for various particle types. While CConv showed proficiency in water simulations, it struggled with more complex materials and preserving the shape of rigid objects in simulations like the *BOXBATH* domain. In contrast, the GNS model displayed better rollout accuracy across multiple domains without needing such tailored approaches. An example of this observation can be seen here:

<!-- <div style="text-align: center;">
  <img src='/images/cconv-comparison.png' style="width: 400px;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/cconv-comparison.png' style="height:auto; width:60%;">
  <figcaption><p style="color: grey; font-size: smaller;">Source: Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

Overall, the GNS model demonstrated a more versatile and accurate performance in these comparisons, handling a variety of simulations more effectively than the specialized methods of DPI and CConv.

## Further application of the GNS framework

The GNS framework is a versatile tool that can be applied to numerous domains and scenarios, especially where complex interactions and relationships are best represented through graph-based models. While the discourse in this paper primarily centers on the utilization of mesh-free particle methods, it's noteworthy that the versatility of the GNS framework extends to encompass mesh-based methods as well. This expanded application has been explored and presented in a subsequent paper by the same team of authors. For readers seeking to delve deeper, the work titled <a href="https://iclr.cc/virtual/2021/spotlight/3542" style="color: blue;">"Learning Mesh-Based Simulation with Graph Networks"</a> by Tobias Pfaff is a key resource. It was presented at the International Conference on Learning Representations (ICLR) in 2021 and offers an expanded application of the GNS framework.

# Discussion
------

This chapter delves into a comprehensive analysis of the study, segmenting our focus into three critical sections: 'Why does the GNS framework work so well?', 'Limitations', and 'Review'. Each segment aims to provide an evaluation of the study, from understanding its inherent constraints, to offering a balanced critique of its overall contributions and implications in the field.

## Why does the GNS framework work so well?

The success of the GNS framework can be significantly attributed to its use of **inductive biases**. As elucidated by Battaglia et al. (2018), an "inductive bias allows a learning algorithm to prioritize one solution (or interpretation) over another, independent of the observed data." This is to say, inductive biases guide the learning process, offering a way to inject domain knowledge into the model which helps in generalization, especially when the available data is scarce or noisy.

In the case of the GNS framework, its inductive biases are deeply intertwined with the representation and processing of physical systems which explain its efficacy. The first such bias is the **representation of the physical system as a graph**. This allows the model to consider that particles in close proximity have a more significant influence on each other's behavior. This leads to a more focused computation, where the algorithm mainly works on the most important interactions and avoids unnecessary calculations on distant particles that don't interact.

Another bias introduced in the GNS framework is the use of **shared node and edge functions**. By treating the dynamics as uniform across all particles, every particle effectively becomes a 'sample' for training the model. This approach not only amplifies the amount of training data, as each particle serves as an observational point, but it also ensures that the learned dynamics are universally applicable across the system, enhancing the model's ability to generalize from limited data.

Lastly, the GNS framework employs **relative encoding**, which posits that the differences in position relative to neighboring particles are sufficient for the model's purposes. This bias towards relative position, rather than absolute coordinates, helps prevent overfitting. By focusing on the relative positions, the model learns to understand and predict the system's dynamics based on local structures and interactions, which are often more generalizable and less prone to overfitting than absolute positions.

These inductive biases in the GNS framework combine to form a robust and efficient model. They serve to embed a priori knowledge about physical interactions into the learning process, thus enabling the framework to make accurate predictions, learn from each particle interaction, and generalize across various scenarios. The framework does all of this while maintaining computational efficiency and reducing the risk to overfit to the training data.

## Limitations

While "Learning to Simulate Complex Physics with Graph Networks" by Sanchez-Gonzalez et al. presents innovative approaches to simulating complex physical systems using GNs, it also has several limitations:

Firstly, the authors prioritized the precision of simulations, which, while yielding high-quality predictions, resulted in **considerable inference times**. This may be limiting rapid deployment in time-sensitive applications. Thus, this trade-off underlines a crucial area for future optimization to enhance the framework's practicality in scenarios where computational speed is of the essence.

Secondly, the **assumption of a constant time step size** in the model can be limiting. Different physical scenarios might require varying time scales, necessitating retraining of the model for each desired time step, which can be impractical and resource-intensive.

<!-- <div style="text-align: center;">
  <img src='/images/fixed-time-step.png' style="width: 400px;">
  <figcaption><p style="color: grey; font-size: smaller;">Inspired by Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div> -->
<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/fixed-time-step.png' style="height:auto; width:60%;">
  <figcaption><p style="color: grey; font-size: smaller;">Inspired by Sanchez-Gonzalez et al. (2020)</p></figcaption>
</div>

Furthermore, the framework's stability is **less reliable for elastic materials** due to its training focus on substances like water, sand, and goop. These materials conform to basic classical mechanics, where position, velocity, and acceleration largely govern particle behavior. Elastic materials, however, present complex stress-strain interactions that go beyond these simple mechanics. As a consequence, the model, not being trained with data specific to the nuanced behaviors of elasticity, struggles to accurately simulate the dynamics of elastic materials.

Lastly, the effectiveness of the GNs, as highlighted in the paper, **hinges on the availability of large-scale, high-quality datasets** specifically tailored for particle-based predictions. However, such datasets are often scarce and challenging to procure, leading to a reliance on other simulation tools to generate the necessary data.

It's important to note that these limitations do not diminish the significance of the paper's contributions but rather highlight areas for further research and consideration when applying these techniques to real-world problems.

## Review

From the standpoint of a reviewer, this paper presents a groundbreaking machine learning framework using Graph Network-based Simulators (GNS) for simulating complex systems, marking a notable advancement in physics-based simulations. The single GNS architecture has demonstrated exceptional capabilities, paving the way for a variety of applications in simulating complex physical phenomena.

The framework's strength lies in its ability to produce high-quality simulations, leveraging **inductive biases** to ensure focused computations and generalizable results. These biases, including the representation of physical systems as graphs and the use of shared node and edge functions, are essential in the framework's success. However, these advantages are counterbalanced by **limitations**, such as considerable inference times and challenges in simulating elastic materials. The assumption of a constant time step size and the dependency on extensive datasets for training also pose significant constraints.

The **versatility of the GNS framework** is evident from its potential application across various domains, ranging from fluid dynamics to solid mechanics. This flexibility suggests that the framework could be adapted to a wide range of complex physical systems, making it a valuable tool in diverse research and industrial settings.

Perhaps the most revolutionary aspect of this work is that it acts as a **proof of concept that AI can simulate physics without explicit equations** describing the underlying phenomena. This represents a paradigm shift, moving beyond traditional equation-based models to data-driven, learned simulations. Such capability is a significant stride in AI, demonstrating its potential to infer complex physical behaviors from data alone.

The paper suggests potential enhancements to the framework by **incorporating more robust, generic physical knowledge**, like Hamiltonian mechanics and architecturally imposed symmetries. These suggestions hint at ways to potentially improve the accuracy and flexibility of simulations.

**Efficiency optimization** emerges as a critical area, especially given the high computational demands of the GNS framework. Future work should focus on optimizing the parameterization and implementation of GNS computations, taking advantage of advances in parallel computing hardware. Such developments could greatly enhance the practicality and scalability of the framework.

Moreover, the potential application of these learned, differentiable simulators in **solving inverse problems** broadens the framework's utility, offering solutions to a wide range of real-world challenges. This approach goes beyond just predicting outcomes, enabling more extensive problem-solving capabilities.

In summary, this paper marks a considerable advancement in the field of AI, particularly in enhancing capabilities for physical reasoning and simulation. It equips researchers with sophisticated tools for understanding and simulating complex physical systems. As research progresses, these developments hold the promise of providing scientists and engineers with more efficient and accurate methods for analyzing and modeling a wide range of physical phenomena.

# Conclusion
------

In summary, this blog post covers the paper “Learning to Simulate Complex Physics with Graph Networks” by Sanchez-Gonzalez et al., a significant work presented at the International Conference on Machine Learning (ICML) in 2020. Our review began with an examination of how Graph Networks (GNs) are used to simulate complex physical phenomena. The authors of this paper skillfully used GNs to model complex physical systems, moving away from traditional simulation methods.

A key aspect of this paper is the effective use of GNs for capturing the dynamics of complex physics. Through message-passing techniques and a particle-based approach, the authors showed how their method excels in simulating various physical situations. This breakthrough in research highlights the role of machine learning in scientific computing and leads to simulations that are more accurate and efficient for complex real-world phenomena.

To conclude, the paper by Sanchez-Gonzalez et al. is an important step in bringing together machine learning and physics. It offers significant potential for improving our understanding and predictions of complex physical systems.


# References
------

<span style="color: grey; font-size: 100%; text-decoration: underline;">
Primary Sources
</span>

<span style="color: grey; font-size: 85%;">
A. Sanchez-Gonzalez et al., <a href="https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html" style="color: blue;">"Learning to simulate complex physics with graph networks"</a>, in Proc. Int. Conf. Machine Learning, PMLR, 2020. 
</span>

<span style="color: grey; font-size: 85%;">
P. W. Battaglia et al., <a href="https://arxiv.org/abs/1806.01261" style="color: blue;">"Relational inductive biases, deep learning, and graph networks"</a>, arXiv preprint arXiv:1806.01261, 2018. 
</span>

<span style="color: grey; font-size: 85%;">
Pfaff, Tobias, et al. <a href="https://iclr.cc/virtual/2021/spotlight/3542" style="color: blue;">"Learning Mesh-Based Simulation with Graph Networks"</a>, 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, OpenReview.net, 2021.
</span>

<span style="color: grey; font-size: 85%;">
Gilmer, Justin, et al. <a href="https://proceedings.mlr.press/v70/gilmer17a" style="color: blue;"> "Neural message passing for quantum chemistry"</a>, International conference on machine learning. PMLR, 2017. 
</span>

<span style="color: grey; font-size: 85%;">
Y. Li et al., <a href="https://arxiv.org/abs/1810.01566" style="color: blue;">"Learning particle dynamics for manipulating rigid bodies, deformable objects, and fluids"</a>, arXiv preprint arXiv:1810.01566, 2018.
</span>

<span style="color: grey; font-size: 85%;">
B. Ummenhofer et al., <a href="https://openreview.net/forum?id=B1lDoJSYDH" style="color: blue;">"Lagrangian fluid simulation with continuous convolutions"</a>, International Conference on Learning Representations, 2019.
</span>

<span style="color: grey; font-size: 85%;">
J.J. Monaghan, <a href="https://www.annualreviews.org/doi/pdf/10.1146/annurev.aa.30.090192.002551" style="color: blue;">"Smoothed particle hydrodynamics"</a>, Annual review of astronomy and astrophysics, vol. 30, no. 1, pp. 543-574, 1992.
</span>

<span style="color: grey; font-size: 85%;">
M. Müller, B. Heidelberger, M. Hennix, and J. Ratcliff, <a href="https://www.sciencedirect.com/science/article/abs/pii/S1047320307000065" style="color: blue;">"Position based dynamics"</a>, Journal of Visual Communication and Image Representation, vol. 18, no. 2, pp. 109-118, 2007.
</span>

<span style="color: grey; font-size: 85%;">
D. Sulsky, S.-J. Zhou, and H.L. Schreyer, <a href="https://www.sciencedirect.com/science/article/abs/pii/0010465594001707" style="color: blue;">"Application of a particle-in-cell method to solid mechanics"</a>, Computer physics communications, vol. 87, no. 1-2, pp. 236-252, 1995.
</span>

<span style="color: grey; font-size: 100%; text-decoration: underline;">
Additional Sources
</span>

<span style="color: grey; font-size: 85%;">
A. Sanchez-Gonzalez et al., <a href="https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate" style="color: blue;">"Learning to simulate"</a>, Github Repository, 2022. (for the paper "Learning to simulate complex physics with graph networks" from A. Sanchez-Gonzalez et al., 2020)
</span>

<span style="color: grey; font-size: 85%;">
A. Sanchez-Gonzalez et al., <a href="https://sites.google.com/view/learning-to-simulate" style="color: blue;">"Learning to simulate complex physics with graph networks"</a>, Google Sites, 2020. 
</span>

<span style="color: grey; font-size: 85%;">
ML Explained - Aggregate Intellect - AI.SCIENCE, <a href="https://www.youtube.com/watch?v=8v27_jzNynM" style="color: blue;">"Tobias Pfaff (DeepMind): Learning to Simulate Complex Physics with Graph Networks"</a>, 2020.
</span>

<span style="color: grey; font-size: 85%;">
Sanchez-Lengeling, et al., <a href="https://distill.pub/2021/gnn-intro/" style="color: blue;">"A Gentle Introduction to Graph Neural Networks"</a>, Distill, 2021.
</span>

All links were last followed on February 10, 2024.

-----









