# Mixture of Linear Experts Model

## Model

Model the data as a combination of linear regressions. Each linear expert specialize to a soft region of the input space.
The input, gating and outpu nodes can have arbitrary dimension.

## Training

The training data is generated from a random instance of the model itself. The parameters are then reinitialized and re-learned.

The training_mode variable allows to specify the optimization method used in the maximization step:
    1 -> Newton-Raphson
    2 -> Gradient Ascent


