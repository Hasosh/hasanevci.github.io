---
title: "Learning to Simulate Complex Physics with Graph Networks"
date: 2023-11-20
permalink: /posts/2023/11/seminar-blog-post/
excerpt: "This is a blog post summarizing the paper 'Learning to Simulate Complex Physics with Graph Networks' (2020) by Sanchez-Gonzalez, Alvaro, et al. published in the International conference on machine learning. <img src='https://hasosh.github.io/hasanevci.github.io/images/front-page-stretched.png' style='width:500px'>"
# excerpt: "This is a blog post summarizing the paper 'Learning to Simulate Complex Physics with Graph Networks' (2020) by Sanchez-Gonzalez, Alvaro, et al. published in the International conference on machine learning. <img src='/images/front-page-stretched.png' style='width:500px'>"
collection: Machine Learning
tags:
  - Graph Network
  - Message Passing
  - Particle-based System
  - Complex Physics Modeling
  - Graph-based Simulation
  - Deep Learning in Scientific Computing
---


Ever pondered the fascinating realm of predicting the future positions of particles and the ideal models to achieve such a feat? If you've ever been intrigued by this scientific enigma, you're in for a treat!

<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/intro.png' alt="Introductory example" style="height:240px">
</div>
<!-- <div style="text-align: center;">
  <img src='/images/intro.png' alt="Introductory example" style="height:240px">
</div> -->

<style>
    /* CSS styles go here */
    .gif-container {
      position: relative;
      width: 720px; /* Set the width and height according to your GIF dimensions */
      height: 360px;
      overflow: hidden;
    }

    #hover-gif {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      opacity: 0;
    }

    .hover-box {
      position: absolute;
      top: 0;
      right: -50px; /* Adjust the distance from the left as needed */
      background-color: rgba(0, 0, 0, 0.5);
      color: white;
      padding: 10px;
      cursor: pointer;
    }
</style>

<div class="gif-container" style="text-align: center;">
  <img src="https://hasosh.github.io/hasanevci.github.io/images/water_ramps_rollout.gif" alt="Water ramps rollout" id="hover-gif">
  <img src="https://hasosh.github.io/hasanevci.github.io/images/water_ramps_rollout.png" alt="Water ramps rollout" id="overlay-image">
</div>

<script>
    // JavaScript code goes here
    var gif = document.getElementById('hover-gif');
    var overlayImage = document.getElementById('overlay-image');

    gif.addEventListener('mouseenter', function() {
      this.style.opacity = '1'; // Show the GIF on hover
      this.play(); // Play the GIF
      overlayImage.style.opacity = '0'; // Hide overlay image
    });

    gif.addEventListener('mouseleave', function() {
      this.style.opacity = '0'; // Hide the GIF when not hovering
      this.pause(); // Pause the GIF
      overlayImage.style.opacity = '1'; // Show overlay image
    });
  </script>

---

# Introduction
------

Welcome to this blog post where I'll provide an overview and insights into the paper titled ***'Learning to Simulate Complex Physics with Graph Networks'*** authored by *Sanchez-Gonzalez et al.*, which was presented at the International Conference on Machine Learning (ICML) in 2020. In this paper, the authors delve into the exciting realm of simulating complex physical phenomena using Graph Networks. They explore how these innovative techniques can be applied to enhance our understanding and prediction of complex physics, bridging the gap between deep learning and numerical methods.

As a student passionate about using machine learning solve problems in the sciences, I find this paper particularly fascinating. It showcases the potential for mathematical concepts and deep learning to synergize, leading to significant advancements in our ability to model and simulate intricate physical systems. This post provides a comprehensive exploration of the paper's key findings and contributions in the context of enhancing our understanding of complex physics through innovative neural network architectures. It also draws inspiration from the <a href="https://www.youtube.com/watch?v=8v27_jzNynM" style="color: blue;">author's talk</a>.

## Why do we even need to simulate complex physics?

The need for simulating complex physics arises across various scientific and engineering disciplines. These realistic simulators serve as invaluable tools that enable us to gain a profound understanding of intricate physical phenomena. Whether it's designing cutting-edge aerospace technologies, optimizing fluid dynamics in engineering, or unraveling the behavior of biological systems, simulations provide a controlled and accessible environment to explore and experiment.

In the absence of such simulations, many scientific inquiries and engineering endeavors would be severely hindered. Complex physics simulations offer the means to predict and visualize the behavior of systems that may be difficult or impossible to study directly. They empower us to test hypotheses, predict outcomes, and optimize designs, ultimately advancing our knowledge and technological capabilities. However, the traditional methods of creating these simulations present significant challenges and limitations.

## What is the problem of traditional simulators?

While the importance of simulating complex physics is undeniable, the traditional approach to building simulators poses several daunting challenges. These conventional simulators can be exceptionally costly, both in terms of time and resources. Developing a simulator often demands years of painstaking engineering effort, from designing accurate models to implementing computational algorithms.

One of the critical issues is the trade-off between generality and accuracy. Traditional simulators tend to excel in specific, narrowly defined settings, sacrificing the ability to adapt to a broader range of scenarios. Moreover, constructing high-quality simulators necessitates substantial computational resources, rendering scalability a formidable hurdle.

Even the best traditional simulators may fall short due to inherent limitations. Insufficient knowledge of the underlying physics or the complexities of approximating critical parameters can lead to inaccuracies. As a result, there is a compelling need for alternative approaches that can overcome these challenges and revolutionize the way we simulate complex physics. One such alternative is the utilization of machine learning, which has the potential to train simulators directly from observed data. Graph Networks (GNs), a specialized class of machine learning architectures, have emerged as a particularly promising approach in this context.

## How can you simulate complex physics with graph networks ?

Simulating complex physics with Graph Networks (GNs) involves representing physical systems as graphs, with nodes and edges denoting entities and their interactions. GNs excel in capturing the complex relationships and dynamics within these structured data. By training GNs on observed data, they can learn the underlying physics and predict how physical properties evolve over time. This adaptability makes GNs a powerful tool for simulating a wide range of complex physical phenomena, from fluid dynamics to structural mechanics, without relying on predefined mathematical models. In essence, GNs revolutionize complex physics simulation by leveraging the flexibility and capacity of machine learning within the framework of graph-like representations.

The paper "Learning to Simulate Complex Physics with Graph Networks" by Sanchez-Gonzalez et al. significantly advances the application of Graph Networks (GNs) in simulating complex physics. In their pioneering work, the authors propose a novel Graph Network Simulation (GNS) framework that harnesses the inherent graph-like nature of physical systems. The GNS framework leverages GNs to represent physical entities and their interactions as nodes and edges within the graph, enabling the network to grasp the intricate relationships and dependencies that define complex physical phenomena. Through extensive training on empirical data, as exemplified in their research, the GNS framework teaches GNs to uncover the underlying physics and simulate the evolution of physical properties accurately. By doing so, the paper exemplifies how GNs can be effectively employed to revolutionize the field of complex physics simulation, transcending the limitations of traditional approaches and opening new avenues for understanding and predicting complex physical systems.

The remainder of this post is organized as follows:
2. In the following chapter about **Graph Networks**, we introduce you to  the fundamental principles and concepts of Graph Networks, providing you with a solid understanding of the underlying framework. 
3. After laying the foundations of graph networks, we delve into the components and functionalities of the **Graph Network-based Simulators (GNS) framework**, where we explore how GNS leverages Graph Networks to simulate complex physical systems with remarkable accuracy and generality.
4. In the section on **Experiments and Results**, we present the empirical findings and outcomes of applying the GNS framework, highlighting its effectiveness in simulating diverse physical phenomena.
5. Moving forward, we explore **Further Applications of the GNS framework**, shedding light on the potential extensions and real-world applications of this transformative approach beyond the scope of the original paper.
6. In the **Discussion** section, we engage in a critical examination of the paper's contributions, discussing its strengths, limitations, and implications within the broader context of machine learning and physics simulations.
7. Finally, we conclude with a summary of key takeaways and insights in the **Conclusion** chapter, reflecting on the significance of the paper and its potential to reshape the landscape of complex physics simulation through Graph Networks.

The following chapters assume knowledge about some terminology, a foundational understanding of graph theory and familiarity with the Euler integrator. If you require a refresher on these topics, it is advisable to refer to the [Appendix](#appendix) before proceeding.

# Graph networks
------

For the next sections about graph networks, we use the information provided in Battaglia et al. (2018) paper *"Relational inductive biases, deep learning, and graph networks"*.

## Idea of graph networks

Graph Networks (GNs) are a class of neural network architectures designed to handle data with complex relational structures, such as graphs. The fundamental idea behind GNs is to enable neural networks to effectively reason about and process data points that are interconnected, where the relationships among these data points are critical for understanding the underlying patterns.

The central concept of GNs is the ability to propagate information across the edges and nodes of a graph. Unlike traditional neural networks that operate on fixed-sized inputs, GNs dynamically adapt to varying graph structures. They achieve this through a series of computation steps that involve message-passing and aggregation. GNs process information at different levels, starting from individual edges and nodes, then aggregating information at higher levels, ultimately leading to a global understanding of the entire graph. The power of GNs lies in their capacity to capture relational inductive biases, making them versatile for various tasks.

## Graph Network (GN) block

The primary computational unit within the GN framework is the GN block, which can be described as a "graph-to-graph" module. This module accepts a graph as its input, conducts computations based on the graph's structure, and produces another graph as its output.

A GN block comprises **three "update" functions** denoted as $$\phi$$:

$$
  e'_k = \phi^e(e_k, v_{r_k}, v_{s_k}, u) \\ \nonumber
  v'_k = \phi^v(\bar{e}'_i, v_i, u) \\
  u' = \phi^u(\bar{e}', \bar{v}', u)
$$

and **three "aggregation" functions** denoted as $$\rho$$:

$$
  \bar{e}'_i = \rho^{e \to v}(E'_i) \\ \nonumber
  \bar{e}' = \rho^{e \to u}(E') \\
  \bar{v}' = \rho^{v \to u}(V')
$$

where 

$$
  E'_i = \{(e'_k, r_k, s_k)\}_{r_k = i, k = 1:N^e} \\ \nonumber
  V' = \{v'_i\}_{i = 1:N^v} \\
  E' = \bigcup_i E'_i = \{(e'_k, r_k, s_k)\}_{k = 1:N^e}
$$

These equations describe the operations within the GN block. 

The $$\phi^e$$ function is applied to all edges to calculate per-edge updates, $$\phi^v$$ is applied to all nodes to compute per-node updates, and $$\phi^u$$ is applied once to perform a global update. The $$\rho$$ functions, on the other hand, operate on sets of elements and reduce them to single values that represent aggregated information. Importantly, the $$\rho$$ functions must remain **invariant to permutations** of their inputs and should be able to handle variable numbers of arguments, such as elementwise summation, averaging, or finding the maximum value, among other possibilities.

When a graph, denoted as G, is provided as input to a Graph Neural (GN) block, the computations within the block follow a specific sequence, going from the edge level to the node level, and finally to the global level. Here's an explanation of each step in this computation process:

1. Edge-Level Computation
2. Node-Level Aggregation
3. Node-Level Computation
4. Edge-Level Aggregation
5. Global-Level Aggregation
6. Global-Level Computation

This sequence of computations are visualized in the following figure, where aggregations are already done implicitly.

<img src='https://hasosh.github.io/hasanevci.github.io/images/GN-updates.png'>
<!-- <img src='/images/GN-updates.png'> -->

It's worth noting that while this sequence of steps is assumed here, the order is not strictly enforced. Depending on the specific application and requirements, it is possible to reverse the order of update functions to proceed from the global level to per-node and per-edge updates, offering flexibility in the GN block's design.


## Composing multiple GN blocks

The concept of Graph Networks (GNs) involves the ability to compose multiple GN blocks to construct complex architectures. This composability allows for the creation of intricate models for various applications. Here's an overview of how multiple GN blocks can be composed:

1. **Basic Composition**: In its simplest form, two GN blocks, GN1 and GN2, can be combined by passing the output of the first block as input to the second, resulting in G0 = GN2(GN1(G)). This allows for the sequential application of GN blocks.

2. **Arbitrary Composition**: Multiple GN blocks can be composed arbitrarily, enabling the creation of more intricate architectures. These blocks can either be unshared (having different functions and parameters, akin to layers in a Convolutional Neural Network) or shared (using the same functions and parameters, resembling an unrolled Recurrent Neural Network).

3. **Encode-Process-Decode Architecture**: A common architecture design involves the encode-process-decode configuration. In this setup, an input graph, G_inp, is transformed into a latent representation, G0, through an encoder, GN_enc. A shared core block, GN_core, is then applied M times to generate GM. Finally, an output graph, G_out, is decoded by GN_dec. This design is suitable for tasks such as predicting the dynamics of a system over time. Note that this is also the architecture that is used in the GNS-framework that we will see later.

4. **Recurrent GN-Based Architectures**: Recurrent GN-based architectures maintain a hidden graph, G_t_hid, which takes an observed graph, G_t_inp, as input and produces an output graph, G_t_out, at each step. This architecture is useful for predicting sequences of graphs, such as the trajectory of a dynamical system over time.

You can see the last three composition architectures in (a), (b), (c) of the next figure, respectively.

<img src='https://hasosh.github.io/hasanevci.github.io/images/GN-types.png'>
<!-- <img src='/images/GN-types.png'> -->


# Graph Network-based Simulators (GNS) framework
------

The GNS (Graph Network-based Simulators) framework is a machine learning framework designed for simulating complex physical systems involving fluids, rigid solids, and deformable materials interacting with each other. It represents the state of the system as nodes in a graph and computes dynamics using learned message-passing techniques. The framework demonstrates the ability to generalize across different initial conditions, timesteps, and particle quantities during testing, showing robustness to hyperparameter variations.

## What is a learnable simulation?
In the following discourse, we shall present a precise formal definition of how a learnable simulation can be characterized. 

We assume that $X_t \in X$ represents the state of the system at time $t$. Employing physical dynamics over $K$ discrete time steps results in a **trajectory of states** denoted as 

$$
  X^{t_0:K} = (X^{t_0}, \ldots, X^{t_K}) \nonumber
$$

A **simulator**, $s: \mathcal{X} \rightarrow \mathcal{X}$, models these dynamics by mapping preceding states to consequent future states in a causal manner. We refer to this simulated sequence as 

$$
  \tilde{X}^{t_0:K} = ({X}^{t_0}, \tilde{X}^{t_1}, \ldots, \tilde{X}^{t_K}) \nonumber
$$

which is computed iteratively as 

$$
  \tilde{X}_{t_{k+1}} = s(\tilde{X}_{t_k}) \nonumber
$$

for each time step. Simulators calculate dynamic information that represents how the current state is evolving and use this information to update the current state to predict future states. 

A **trainable simulator**, denoted as $s_{\theta}$, calculates dynamic information using a parameterized function approximator, $d_{\theta}: \mathcal{X} \rightarrow \mathcal{Y}$, where $\theta$ represents the parameters that can be optimized based on a specific training objective. The $Y \in \mathcal{Y}$ signifies the **dynamic information**, the meaning of which is defined by the update mechanism. The update mechanism can be perceived as a function that takes the state $X_k$ and utilizes $d_{\theta}$ to forecast the subsequent state 

$$
  \tilde{X}_{k+1} = \text{Update}(\tilde{X}_k, d_{\theta}) \nonumber
$$

In this context, we assume a straightforward update mechanism, specifically an **Euler integrator**, and $\mathcal{Y}$ represents accelerations.

For further clarity, an illustrative figure is presented, encapsulating this formal definition through a practical example for enhanced comprehension. In the example, each blue circle represents a particle and the glas box resembles the boundaries of the environment. Given the initial positions of the particles, the simulator's task is to "rollout" the trajectory of the next particle states.


<img src='https://hasosh.github.io/hasanevci.github.io/images/simulator-example.png'> 
<!-- <img src='/images/simulator-example.png'> -->

## How do we create graphs for particle-based systems?

In particle-based systems, the concept of a graph $G = (V, E, u)$ can be applied as follows:

Particles within the physical system represent the nodes of the graph. Each particle is analogous to a node in the graph, and the collection of all particles forms the set of nodes denoted as $V$.

Interactions or connections between particles, such as pairwise forces, influences, or constraints, can be represented as edges in the graph. These edges define the relationships and dependencies between particles and are typically denoted as the set of edges, $E$. For example, if particle A interacts with particle B, there would be an edge connecting the corresponding nodes in the graph: $r_{A,B}=(v_A, v_B) \in E$. The crucial aspect here is that it adds edges between particles that are within a specific **"connectivity radius"**, denoted as $R$. This radius reflects local interactions among the particles and remains constant for all simulations at the same resolution.

Additionally, global properties of the physical system that affect all particles uniformly, such as external forces like gravity or magnetic fields, can be represented as global attributes associated with the entire graph. These global properties are captured by the graph's global attribute, denoted as $u$.

## How can we pass messages in a graph?

tbd.

<img src='https://hasosh.github.io/hasanevci.github.io/images/message-passing.png'>
<!-- <img src='/images/message-passing.png'> -->


## GNS architecture 

The state of the physical system can be represented via a tuple of $N$ particles $x_i$:

$$
  X = (x_0, \ldots, x_N) \nonumber
$$

The GNS architecture can be described via a **encode-process-decode architecture** as we have already seen in the section about graph network variants, where an encoder is applied **once** at the beginning of a simulation, a processor is applied **several times** during the simulation, and the decoder is applied **once** at the end of a simulation.

<img src='https://hasosh.github.io/hasanevci.github.io/images/encoder-processor-decoder.png'>
<!-- <img src='/images/encoder-processor-decoder.png'> -->

Each of the three components plays a different role for the graph network.

**Encoder**

The **encoder** takes a particle-based state $X$ as input and maps it to a latent graph $G^0$, where $$G = (V, E, u), v_i \in V, e_{i,j} \in E$$. 

$$
  G^0 = ENCODER(X) \nonumber
$$

Therefore, it has to create embeddings for both the nodes and the edges. The **node embeddings** $v_i$ represent learned functions of the particles' states, denoted as 

$$
  v_i = \varepsilon^v(x_i) \nonumber
$$

The **edge embeddings** $e_{i,j}$ are functions that have been learned from the pairwise characteristics of the associated particles

$$
  e_{i,j} = \varepsilon^e(r_{i,j}) \nonumber
$$ 

These characteristics can include factors like the displacement between their positions, the spring constant, and so on.

The following figure visualizes how the encoder transforms the input of particles into a latent graph.

<div style="text-align: center;">
  <img src="https://hasosh.github.io/hasanevci.github.io/images/encoder.png" alt="Encoder" style="height: 200px;">
</div>
<!-- <div style="text-align: center;">
  <img src="/images/encoder.png" alt="Encoder" style="height: 200px;">
</div> -->

**Processor**

The processor maps a latent graph $G$ to another latent graph $G$, by performing interactions among nodes through $M$ steps of learned **message-passing**. This results in a series of updated latent graphs, represented as 

$$
  G = (G^1, ..., G^M) \nonumber
$$

where each $G^{m+1}$ is generated based on the previous graph $G^m$ using the function $G^{m+1} = GN^{m+1}(G^m)$. Ultimately, the processor yields the final graph

$$
  G^M = PROCESSOR(G^0) \nonumber
$$

which is obtained by applying processor to the initial graph $G^0$.

To perform $M$ message-passing steps, a stack of $M$ GN blocks are used.

A visualization of the processor operating on latent graphs can be seen here:

<div style="text-align: center;">
  <img src="https://hasosh.github.io/hasanevci.github.io/images/processor.png" alt="Processor" style="height: 200px;">
</div>
<!-- <div style="text-align: center;">
  <img src="/images/processor.png" alt="Processor" style="height: 200px;">
</div> -->

**Decoder**

The decoder takes the final latent graph $G^M$ as input and from this extracts dynamics information from the nodes such as acceleration

$$
  Y = DECODER(G^M) \nonumber
$$

To extract dynamics information for a single node $v_i^M$ a learned function $\delta^v$ is applied

$$
  y_i = \delta^v(v_i^M) \nonumber
$$

where $\delta^v$ is learned through an MLP.

Note that the output of the decoder is not a graph and is also not the final positions of the particles! Following the decoder step, the future positions and velocities are gained using an **Euler integrator**. Therefore, $y_i$ represents accelerations, denoted as $\ddot{p}_i$, which can have either 2D or 3D dimensions, depending on the specific physical domain.

Also note that we do not learn a decoder function $\delta^e$ for the edges, as we do not need it to get the final accelerations of the particles. The edges solely serve the purpose of passing messages from on node to another.

The following figure visualizes how the dynamics information $y_i$ are extracted from the final latent graph $G^M$ 

<div style="text-align: center;">
  <img src="https://hasosh.github.io/hasanevci.github.io/images/decoder.png" alt="Decoder" style="height: 200px;">
</div>
<!-- <div style="text-align: center;">
  <img src="/images/decoder.png" alt="Decoder" style="height: 200px;">
</div> -->

**Multilayer Perceptron (MLP)**

Now the question might be: Well how exactly are these embedding functions $\varepsilon^v$ and $\varepsilon^e$ created or learned for the nodes and edges, respectively? Also how are node and edge embeddings updated in the processor and how is the decoding function $\delta^v$ learned?

For this purpose, mutliple **Multilayer Perceptrons (MLP)** are trained separately e.g. for $\varepsilon^v$, $\varepsilon^e$ and $\delta^v$, a MLP separately. For $\varepsilon^v$ and $\varepsilon^e$, they encode the node features and edge features into latent vectors $v_i$ and $e_{i,j}$ of size 128.

All MLPs used for the encoder, processor and decoder have the following architecture (in this order):

- two hidden layers with ReLU activations
- one output layer with no activation
- one LayerNorm layer

## Input and Output Representation

Each particle $x_i$ in the physical system can be expressed as a vector. This vector contains the particle's position $p_i^{t_k}$, the particle'
s 5 previous velocities $\dot{p}_i^{t_k}$ and features $f_i$ that describe static material properties, e.g. storing what type of particle it is (water, sand, goop), storing whether it is flexible or not, and storing whether the particle is at the boundary of the environment. The following figure summarizes the input vector representation of a particle.

<div style="text-align: center;">
  <img src='https://hasosh.github.io/hasanevci.github.io/images/input-representation.png' style="height: 300px;">
</div>
<!-- <div style="text-align: center;">
  <img src='/images/input-representation.png' style="height: 300px;">
</div> -->

## Training

The training process for a complex and chaotic simulation system involves several strategies to improve model performance. The main challenges include **mitigating error accumulation** during long rollouts and **handling noisy inputs**.

1. **Training Noise**: Models are trained using ground-truth one-step data, which lacks accumulated noise found in real-world scenarios. When generating rollouts by feeding the model with its own noisy predictions, the out-of-training-distribution inputs can lead to substantial errors and error accumulation. To address this, the model's input velocities are corrupted with **random-walk noise** $$\mathcal{N(\mu_v, \sigma_v=0.0003)} $$ during training, making the training distribution closer to that seen during rollouts.

2. **Normalization**: All input and target vectors are normalized elementwise to have zero mean and unit variance. This normalization is performed using online statistics computed during training, improving training speed but not significantly affecting the final performance.

3. **Loss Function and Optimization**: The loss function is based on $L_2$ **loss**, calculated on predicted per-particle accelerations from sampled particle state pairs $(x_i^{t_k}, x_i^{t_{k+1}})$
$$ 
  L(x_i^{t_k}, x_i^{t_{k+1}}; \theta) = \|d_{\theta}(x_i^{t_k}) - \ddot{p}_i^{t_k} \|^2 \nonumber
$$

Model parameters are optimized using the Adam optimizer with a small mini-batch size. Training involves up to 20 million gradient update steps with a gradual learning rate decay. This approach maintains consistency across datasets and facilitates fair comparisons across settings.

Model performance is evaluated during training by conducting full-length rollouts on held-out validation trajectories. The training stops when negligible decreases in Mean Squared Error (MSE) are observed. The training duration varies depending on the dataset complexity, ranging from a few hours for simpler datasets to up to a week for more complex ones.

In summary, the training process for complex simulation systems involves strategies such as adding training noise, normalization, specific loss functions, and optimization procedures. Regular evaluation and stopping criteria ensure the models achieve the desired performance.

## Summary

So now after all the information about the Graph Network-based Simulators (GNS) framework, how can we summarize it neatly? 

The authors provide a pretty accurate short description of their framework:

<blockquote style="background-color: #f2f2f2; padding: 10px; border-left: 5px solid #ccc;">
  "Our framework imposes strong inductive biases, where rich physical states are represented by graphs of interacting particles, and complex dynamics are approximated by learned message-passing among nodes."
</blockquote>

From the citation, we can extract the following parts of information:

1. **Strong inductive bias**: By using Graph Networks, the Euler integration to update states and the one step loss as training objective, a strong inductive bias is imposed on the framework.
2. **Rich physical states**: The states of the particles consists of several components including their position, velocities and further features.
3. **Graphs of interacting particles**: A particle-based system is processed in their graph network using particles for the nodes and establishing edges between nodes that are close to each other.
4. **Complex dynamics**: The dynamics of the physical system are complex as they involve a wide range of intricate and often nonlinear interactions among particles.
5. **Message passing**: For the prediction of the next state, information between particles must be exchanged withing the created graph to correctly process the ongoing dynamics of the current state.



# Experiments and Results
------

Code and data of the GNS framework can be found at this <a href="https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate" style="color: blue;">GitHub repository</a>.

## Experimental Domains

tbd.
<img src='https://hasosh.github.io/hasanevci.github.io/images/experiment-domains.png'>
<!-- <img src='/images/experiment-domains.png'> -->

## Generalization
tbd.

## Key Architectural Choices

In their study, the authors conducted a comprehensive analysis of the architectural choices within their Graph Neural Simulator (GNS) to assess their impact on performance. While various hyperparameters, such as the number of MLP layers and encoder/decoder functions, were examined, it was found that these had minimal effects on performance (for more details, refer to Supplementary Materials C).

However, several factors were identified that significantly influenced the GNS model's performance:

1. **The number of message-passing steps** (M) was observed to have a substantial impact. Increasing M improved performance in both one-step and rollout accuracy, allowing the model to capture longer-range and more complex interactions among particles. It was recommended to choose the smallest M that met the desired performance criteria, considering computational time.

2. The choice between **shared and unshared parameters in the PROCESSOR** had a noticeable effect. Unshared parameters, despite resulting in more parameters, yielded better accuracy, especially for rollouts. Shared parameters introduced a strong inductive bias similar to recurrent models, whereas unshared parameters resembled a deep architecture.

3. **The connectivity radius** (R) was another influential factor. Larger R values led to lower error rates as they facilitated longer-range communication among nodes. However, larger R values required more computational resources and memory. Therefore, it was advisable to use the minimal R that met the desired performance criteria.

4. **The scale of noise added to inputs during training** had an impact on rollout accuracy. Intermediate noise scales resulted in the best rollout accuracy, in line with the motivation for introducing noise. However, one-step accuracy decreased with increasing noise scale, as noise made the training distribution less similar to the uncorrupted distribution used for one-step evaluation.

5. The choice between the **relative and absolute ENCODER** showed that the relative version was superior. This preference likely arose from the fact that the underlying physical processes being learned were invariant to spatial position, aligning with the inductive bias of the relative ENCODER.

These decisions were guided by the experimental results, as illustrated in Figure 4, which depicts the effect of different ablations against their model on one-step and rollout errors. The bars represent the median seed performance averaged across the entire GOOP test dataset, with error bars displaying lower and higher quartiles for the default parameters.

<img src='https://hasosh.github.io/hasanevci.github.io/images/ablation-study.png'>
<!-- <img src='/images/ablation-study.png'> -->

In summary, the study revealed that certain architectural choices, such as the number of message-passing steps, shared vs. unshared parameters, connectivity radius, noise scale, and the type of ENCODER, had significant impacts on the performance of the GNS model. These findings offer valuable insights for optimizing the model for various applications in the GOOP domain.

# Another Application of the GNS framework
------

In the realm of further applications of Graph Networks (GNS), it's noteworthy to introduce the paper titled "Learning mesh-based simulation with Graph Networks" by Tobias Pfaff. This paper, published at the International Conference on Learning Representations (ICLR) in 2021, explores an intriguing avenue where Graph Networks are employed to revolutionize mesh-based simulations.

**Overview of the Paper "Learning mesh-based simulation with Graph Networks"**

Tobias Pfaff's paper dives into the fascinating domain of mesh-based simulations. Mesh-based simulations are a common technique used to model and simulate complex physical systems, particularly in fields such as computational fluid dynamics, structural mechanics, and computer graphics. These simulations rely on dividing the simulation domain into a mesh, typically consisting of interconnected vertices and edges, and solving partial differential equations (PDEs) on this mesh to simulate the behavior of the system over time.

Traditionally, mesh-based simulations involve solving intricate PDEs that describe how physical properties, such as temperature, pressure, or deformation, propagate and interact within the mesh. These simulations are computationally intensive and often require domain-specific knowledge to set up and fine-tune.


**Applications of Mesh-Based Simulation**

Mesh-based simulations find applications in a wide range of fields, as can be seen in the following figure:

<img src='https://hasosh.github.io/hasanevci.github.io/images/mesh-based-simulation-examples.png'>
<!-- <img src='/images/mesh-based-simulation-examples.png'> -->

<ol style="list-style-type: lower-alpha;">
  <li><b>Flag Waving in the Wind</b>: The model accurately simulates the movement of a flag as it ripples and sways in response to the wind.</li>
  <li><b>Deforming Plate</b>: In this scenario, the model demonstrates its capability to predict the deformation of a plate, highlighting its potential in structural mechanics applications. A color map represents the von-Mises stress, providing insights into the stress distribution across the plate.</li>
  <li><b>Flow of Water</b> around a Cylinder Obstacle: The model successfully replicates the flow of water around a cylinder-shaped obstacle, showcasing its applicability in fluid dynamics simulations.</li>
  <li><b>Aircraft Wing Dynamics</b>: The authors also exhibit the model's proficiency in modeling the aerodynamic behavior of air around the cross-section of an aircraft wing. This has significant implications for aerospace engineering. In this case, the color map represents the x-component of the velocity field, offering insights into the airflow patterns.</li>
</ol>

Incorporating Graph Networks into mesh-based simulations holds the promise of making these simulations more accessible, efficient, and adaptable to a broader range of applications, thereby advancing our capabilities in understanding and modeling complex physical systems.

# Discussion
------

## Shortcomings / Open Questions
tbd.
- Very limited information about the message passing algorithm
- Comparison to other models comes too short

## Limitations

While "Learning to Simulate Complex Physics with Graph Networks" by Sanchez-Gonzalez et al. presents innovative approaches to simulating complex physical systems using Graph Networks (GNs), it also has several limitations:

1. **Data Dependency**: The paper relies on large-scale, high-quality datasets for training the Graph Networks effectively. This data requirement may limit the applicability of the approach in scenarios where collecting such datasets is challenging or expensive.

2. **Complexity and Scalability**: Implementing and training Graph Networks can be computationally expensive and complex, especially for large-scale simulations. This could hinder practical adoption, particularly for researchers and engineers with limited computational resources.

3. **Domain Expertise**: The paper's approach might require substantial domain expertise to fine-tune and adapt to specific physical systems or scenarios. This could limit its accessibility to those without a deep understanding of both machine learning and the physics involved.

4. **Robustness to Noise**: The robustness of the learned models to noisy or imperfect data is not extensively discussed. In real-world scenarios, data often contains noise, and understanding how the approach handles such noise is crucial.

It's important to note that these limitations do not diminish the significance of the paper's contributions but rather highlight areas for further research and consideration when applying these techniques to real-world problems.

## Outlook

From a reviewer's standpoint, the machine learning framework presented in this paper for simulating complex systems using Graph Networks (GNS) is undeniably groundbreaking. The experimental results have not only showcased the exceptional capabilities of the single GNS architecture but have also opened up exciting possibilities for future research and applications in the realm of physics-based simulations.

One avenue for exploration, which warrants further investigation, is the extension of the GNS approach to data represented using meshes, such as finite-element methods. This expansion could potentially broaden the scope of the framework, making it even more versatile and adaptable to diverse simulation scenarios.

The paper hints at the incorporation of stronger, generic physical knowledge, including Hamiltonian mechanics and architecturally imposed symmetries. As a reviewer, I believe that delving deeper into these areas could substantially enhance the accuracy and flexibility of the simulations and would be a valuable direction for future work.

Efficiency optimization is paramount, especially considering the computational demands of large-scale simulations. It is imperative to explore strategies for improving the parameterization and implementation of GNS computations while capitalizing on advances in parallel compute hardware. This optimization could significantly impact the practicality and scalability of the framework.

Furthermore, as a reviewer, I see immense potential in the application of learned, differentiable simulators to solve inverse problems. Going beyond strict forward prediction to address inverse objectives could broaden the utility of the framework and find applications in a myriad of real-world challenges.

In a broader context, this paper's contribution represents a significant step forward in enhancing generative models and equipping the AI toolkit with enhanced physical reasoning capabilities. As we look to the future, these research directions offer promising opportunities to empower scientists and engineers with an even more powerful tool for understanding and simulating complex physical systems.

# Conclusion
------

In summary, this analysis has focused on the paper "Learning to Simulate Complex Physics with Graph Networks" authored by Sanchez-Gonzalez et al., a notable contribution presented at the International Conference on Machine Learning (ICML) in 2020. We commenced our exploration by delving into the innovative application of Graph Networks in simulating complex physical phenomena. The paper's authors adeptly harnessed the power of Graph Networks to model intricate physical systems, a departure from traditional simulation methods.

Notably, the utilization of Graph Networks introduced an effective means of capturing and understanding the dynamics of complex physics. By leveraging message passing mechanisms and a particle-based system, the authors demonstrated their approach's superiority in simulating various physical scenarios. This research breakthrough not only underscores the potential of machine learning techniques in scientific computing but also paves the way for more accurate and efficient simulations of complex real-world phenomena.

In conclusion, "Learning to Simulate Complex Physics with Graph Networks" by Sanchez-Gonzalez et al. marks a significant stride in the convergence of machine learning and physics. This work holds promise for enhancing our comprehension and predictive prowess in the domain of intricate physical systems.

# References
------

A. Sanchez-Gonzalez et al., <a href="https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html" style="color: blue;">"Learning to simulate complex physics with graph networks"</a>, in Proc. Int. Conf. Machine Learning, PMLR, 2020.

P. W. Battaglia et al., <a href="https://arxiv.org/abs/1806.01261" style="color: blue;">"Relational inductive biases, deep learning, and graph networks"</a>, arXiv preprint arXiv:1806.01261, 2018.

Pfaff, Tobias, et al. <a href="https://iclr.cc/virtual/2021/spotlight/3542" style="color: blue;">"Learning Mesh-Based Simulation with Graph Networks"</a>, 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, OpenReview.net, 2021

Gilmer, Justin, et al. <a href="https://proceedings.mlr.press/v70/gilmer17a" style="color: blue;"> "Neural message passing for quantum chemistry"</a>, International conference on machine learning. PMLR, 2017. 

<a href="https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate" style="color: blue;">Github Repository</a> for A. Sanchez-Gonzalez et al., "Learning to simulate complex physics with graph networks"

https://www.youtube.com/watch?v=8v27_jzNynM

https://www.youtube.com/watch?v=2Bw5f4vYL98

https://distill.pub/2021/gnn-intro/ (figure)

https://en.wikipedia.org/wiki/Graph_theory (figure)

https://en.wikipedia.org/wiki/Euler_method (figure)

-----



<a name="appendix"></a>

# Appendix

## Terminology

In order to explain the components and functionality of the GNS framework, it is essential to introduce certain terminology, thereby laying the foundational framework for our subsequent discussion. 
- **World / Environment**: refers to the system or environment being simulated or modeled. It represents the state of the physical or conceptual system under consideration at a specific point in time.
- **Physical Dynamics**: refers to the principles and laws of physics that govern how the state of a system or the world (see above) changes over time. These physical dynamics encompass the fundamental rules that describe the behavior of objects and systems in the real world, such as the laws of motion, conservation of energy, and other physical principles.
- **Dynamics Information**: refers to data or knowledge that characterizes how the current state of the system is changing over time. It encompasses the quantitative and qualitative details regarding the evolution of the system's state as it progresses from one time step to the next. More specifically, "dynamics information" is the information that simulators gather and process to understand and predict how the system's state will transition from its current state to a future state. This information can include variables like velocities, accelerations, forces, or any other parameters that describe the rate and direction of change in the system's properties.

## Graph Theory

In the context of graph theory and graph networks, a graph is formally defined as $G = (V, E, u)$, where:

- $$V$$ is the set of nodes, denoted as $$V = \{v_i \| v_i \in \{1, \ldots, N\}\}$$, where each $v_i$ represents an individual node within the graph. For example, in a social network, $V$ could represent individuals, and in a transportation network, $V$ could represent locations or cities.

- $E$ represents the set of edges, which defines the connections or relationships between the nodes. It is typically defined as $$E \subseteq \{e_{i,j}=(v_i, v_j) \| v_i, v_j \in V\}$$, indicating that edges are pairs of nodes from the node set. In a social network, the edges might represent friendships between individuals, while in a computer network, they could signify network connections between devices.

- $u$ represents global properties or attributes associated with the entire graph, i.e. properties that may apply to all nodes and edges. These global properties can vary widely depending on the context of the graph. For instance, in a financial network, $u$ could represent the overall economic stability of a region, and in a recommendation system, $u$ might represent user preferences or trends that apply to the entire user base.

This formal definition encapsulates the fundamental components of a graph: nodes, edges, and global properties, providing a basis for understanding and analyzing graph structures in various applications, including graph networks.

The following figure depicts a graph $G=(V,E)$ with the following properties:

- $$V = \{1, 2, 3, 4, 5, 6\}$$
- $$E = \{(1, 2), (2, 1), (1, 5), (5, 1), (2, 5), (5, 2), (2,3), (3,2), (5,4), (4,5), (3,4), (4,3), (4,6), (6,4)\}$$

Note that $u$ is left out as the example graph does not have any global properties or attributes.

<div style="text-align: center;">
  <img src="https://hasosh.github.io/hasanevci.github.io/images/graph-example.png" alt="Graph Example" style="height: 200px;">
</div>
<!-- <div style="text-align: center;">
  <img src="/images/graph-example.png" alt="Graph Example" style="height: 200px;">
</div> -->

## Euler integrator

The Euler integrator is a simple numerical method used for solving ordinary differential equations (ODEs). It approximates the next state of a system by using the current state and the derivatives of the state variables. The formula for the Euler integrator is:

$X_{k+1} = X_k + h \cdot f(X_k, t_k)$

Where:

- $X_k$ represents the current state of the system at time $t_k$.
- $X_{k+1}$ is the estimated next state at time $t_{k+1}$.
- $h$ is the time step, which determines the size of the time intervals between calculations.
- $f(X_k, t_k)$ represents the derivative of the state variables at the current time $t_k$.

In essence, the Euler integrator updates the state by taking a small step in time $h$ in the direction of the derivative $f(X_k, t_k)$ at the current time $t_k$. While it is a straightforward method, it may introduce errors, especially when dealing with complex or highly nonlinear systems, and more advanced numerical integration methods are often used for greater accuracy. 

The following fiture illustrates the application of the euler method on an example.

<div style="text-align: center;">
  <img src="https://hasosh.github.io/hasanevci.github.io/images/euler_method.png" alt="Euler Method" style="height: 300px;">
</div>
<!-- <div style="text-align: center;">
  <img src="/images/euler_method.png" alt="Euler Method" style="height: 300px;">
</div> -->


The blue curve represents the unknown curve while the red curve shows the approximation computed by the Euler integration. From the given example it is apparent that the more time steps must be approximated in advance the higher chance there is for errors to accumulate.