# HPC KANNet – Rust implementation of the KANNet module  
[![Srpski](img.shields.io)](README.md)
[![English](img.shields.io)](README.en.md)

Original module structure: https://github.com/Lija321/KANNet

#### Basic Information
Student: Dejan Lisica  
The project will be implemented with the goal of achieving the highest grade (10).

#### Problem Description
The project focuses on the implementation of a 2D convolution-like block which, instead of relying on the classical universal approximation theorem, is based on the Kolmogorov–Arnold representation theorem. Almost all modern image recognition networks rely on convolutional blocks; however, the KANNet block replaces linear combinations with a sequence of ReLU functions in order to approximate a multivariate function for each block, enabling the learning of more complex relationships between pixels.

Our architecture is based on the KAN layer proposed in:  
https://arxiv.org/abs/2406.02075  

The parameters $e$ and $s$ are trainable parameters learned during training. The core of each block is a network layer defined by the following decomposition:

- $A = \text{ReLU}(E - x^T)$
- $B = \text{ReLU}(x^T - S)$
- $D = r \times A \cdot B$
- $F = D \cdot D$
- $y = \mathbf{W} \bigotimes F$

Our block can be of arbitrary size $n$, for which we construct a corresponding network layer with $n^2$ inputs. Similar to a standard convolutional block, the block “slides” over the image, and the corresponding pixels are mapped to the appropriate inputs of the layer.

The motivation of this project is not to evaluate classification accuracy or learning performance, but rather to analyze the computational characteristics and performance implications of KAN-based convolution-like blocks, especially in comparison to classical convolutional implementations, with a strong focus on high-performance computing aspects.

#### Implementation Description
The input to the KANNet block will be randomly generated single-channel images. As in the classical convolution case, multiple input sizes will be considered. All input matrices and parameters will be randomly generated at the beginning of the experiment and stored in files, ensuring that sequential and parallel implementations operate on identical data.

For each spatial position of the sliding window (of size $n \times n$), the corresponding pixels are flattened into a vector $x \in \mathbb{R}^{n^2}$. This vector serves as the input to the KAN layer. The operations defined above are then applied to compute the output value for that spatial location.

Padding strategies will be considered to ensure well-defined behavior at image boundaries and to control the output dimensionality, analogously to classical convolutional blocks. The output of the KAN block can optionally be followed by additional operations such as:
- element-wise activation (if required),
- pooling (e.g., max pooling or average pooling),  
in order to maintain comparability with standard convolutional pipelines.

The primary focus of the implementation will be on efficiency, memory access patterns, and parallelization strategies rather than numerical optimization or training stability.

#### Sequential Solution
The sequential implementation will closely follow the mathematical definition of the KAN layer and the sliding-window mechanism described above. Different input sizes and block sizes $n$ will be evaluated, and execution times will be recorded.

The reference implementation will be written in Rust, emphasizing:
- explicit memory management,
- cache-friendly data layouts,
- avoidance of unnecessary allocations.

This sequential version will serve as the baseline for all performance comparisons.

#### Parallel Solution
The parallel solution will be based on spatial decomposition of the input matrix. The input image will be divided into independent subregions, each processed by a separate thread. Care will be taken to ensure that each sliding window is fully contained within a single subregion, avoiding data dependencies between threads.

Key aspects of the parallelization strategy include:
- partitioning the image such that no KAN block spans multiple partitions,
- careful handling of padding regions,
- minimizing synchronization overhead.

The implementation will use Rust threads and synchronization primitives where necessary, with an emphasis on embarrassingly parallel execution. The impact of block size $n$, input size, and number of threads on performance will be systematically evaluated.

#### Strong and Weak Scaling Experiments
Both strong and weak scaling experiments will be conducted:
- **Strong scaling:** the input size is fixed while the number of threads is increased, measuring speedup and efficiency.
- **Weak scaling:** the input size increases proportionally with the number of threads, evaluating how well the implementation maintains constant execution time.

The results will be analyzed in the context of Amdahl’s and Gustafson’s laws, highlighting the limits and potential of parallelization for KAN-based convolution-like blocks.

#### Visualization of Results
The results will be visualized using plots that show:
- execution time versus input size for sequential and parallel implementations,
- speedup and efficiency as functions of the number of threads,
- comparisons between different block sizes $n$.

For visualization in Rust, one of the following libraries will be used:

- `Plotters`
- `Rerun`

The visualizations will support a clear performance-oriented analysis and provide insight into the scalability and computational behavior of KANNet blocks in an HPC context.
