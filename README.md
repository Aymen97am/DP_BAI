# DP-BAI Repository

This repository contains an implementation of best-arm identification algorithms with differential privacy (DP-BAI). The algorithms aim to solve the best-arm identification problem while preserving the privacy of sensitive data using differential privacy techniques.


## Usage

To run the experiments and simulations, execute the `experiment.py` file located in the root directory. This script performs multiple simulations for different combinations of hyperparameters and saves the results in separate folders.

Make sure you have the required dependencies installed. You can install them by running:

```
pip install -r requirements.txt
```

Once the dependencies are installed, you can run the experiments by executing:

```
python experiment.py
```

The script will generate results and save them in separate folders for each algorithm. After running the experiments, the results are read from the generated CSV files, and a plot is generated to visualize the sample complexity of the algorithms.

## Results

The generated results are stored in separate folders for each algorithm under the `experiments` directory. The folders are named as follows:

- `/UCBTT` for the TT-UCB algorithm.
- `/AdaPTT` for the AdaPTopTwo algorithm.
- `/DPSE` for the DP-SE algorithm.

Each algorithm folder contains CSV files with the experiment results, including epsilon values, taus, and other relevant data. The `read_results` function in the `utils.py` file is used to read and process these results.

## Figures

The generated figures are saved under the `figures` directory. The figures visualize the sample complexity of each algorithm based on the experiment results. The figure files are named according to the experiment ID.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.
