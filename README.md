
# Topological Benchmark

A work-in-progress benchmarking dataset for evaluating topological methods.

For more information, see: [topological-benchmark.github.io
](topological-benchmark.github.io
).

## Benchmarking Dataset

The first iteration of the project consists of a synthetic point cloud created by sampling from predetermined shapes.

### Shapes

The set of shapes includes:

- Ellipsoid
- Torus
- Point
- 2-sphere
- Ball
- Hopf link
- 2-sphere inside a 2-sphere
- Ball inside a 2-sphere
- A link of a filled torus and a hollow torus

### Dataset Parameters

The following geometric and sampling parameters can be varied:

- Rescaling of shapes, including stretching
- Sampling density
- Separation or minimal distance between objects
- Thickening
- Noise: iid point noise plus smooth random vector-field displacement, both scaled by object size
- Sampling texture, meaning introducing non-uniform density along the shape


### Data representation

 - Point clouds
 - Mesh
 - 3D image

### Tasks

The benchmark is intended for the following tasks:

- Determine the homology of all objects in all degrees 
- Multi-Label Shape Classification
- "Balanced" Betti numbers (different Betti numbers should appear)


## Project Status

The specifications of the dataset and the dataset are under development.
