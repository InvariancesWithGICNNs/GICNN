# Input-Invex Neural Networks
Code accompanying the ICML22 Submission.
Includes the Fashion-MNIST experiment implemented in Tensorflow 2.4.

- `fmnist_experiment.py` is the main code. This includes Network Architecture, Training & Plots.
- `lib/` contains the keras model, used layers, and other utilities.

You can set up the conda environment and run the Fashion-MNIST experiment with the script provided in `run_fmnist.sh`.

After using `conda` for setting up the environment specified in `env.yml`,
the experiment can be run with `python fmnist_experiment.py`.
A single `Ctrl+C` will interrupt the training and continue with the plotting without killing the script.
If training takes too long, feel free to use this for seeing intermediate results. 
After ~20 Epochs the reconstructions look reasonably well.
