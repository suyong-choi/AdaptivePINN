# Adaptive PINN
Exercise of self-adaptive physics informed neural network (https://arxiv.org/pdf/2009.04544.pdf). In this example, Laplace's equation is solved in 2-D for Dirichlet boundary conditions. Imposing large initial weights for BC loss is needed for satisfactory convergence. For a general class of solutions, you should try to use TensorDiffEq (https://tensordiffeq.io).
