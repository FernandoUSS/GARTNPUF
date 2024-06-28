## Genetic Algorithm to improve the reliability of PUFS 

In this code, a genetic algorithm is used to improve the reliability of PUFs. The parameters are:

n_gen          # Number of generations

mutation_rate  # Mutation rate

pob_size       # Poblation size

n_offspring    # Number of children 

n_runs         # Runs of the algorithm 

t_MCF_20       # t_MCF selected for 20ºC

t_meas_20      # Aperture of data

T              # Temperatures

n_ttos         # Number of transistors

n_pairs        # Number of pairs

n_meas_T       # Number of MCF measurements per temperature

n_meas         # Number of MCF measurements for all temperatures

comp_offset    # Comparator offset

t_MCF_adp      # t_MCF adaptation according to temperature

stopping_crit  # Stopping criteria

stop_limit     # Stop limit

k_b            # Boltzmann constant to adapt to temperature

Ea_adp         # Mean activation energy to adapt to temperature

fitness        # Fitness function (NSP or Rel)

P              # Probability value for the NSP fitness function

## Installation

To run the Python scripts in this repository using Conda, follow these steps:

### 1. Clone the Repository

First, clone this repository to your local machine using `git`:

```bash
git clone https://github.com/FernandoUSS/.git
```

### 2. Navigate to the Repository

Navigate into the cloned repository directory:

```bash
cd trng_rtn
```

### 3. Create, Install Required Packages, and Activate Conda Environment
Create a new Conda environment (e.g., named myenv), install the required Python packages listed in requirements.txt, and activate it:

```bash
conda create --name myenv --file requirements.txt
conda activate myenv
```

### 4. Run the Script
You can now run the Python script. For example, to run smacd24_DMCF_viewer.py, use the following command:

```bash
python -m plots.smacd24_DMCF_viewer
```

Replace `smacd24_DMCF_viewer` with the name of your Python script. Ensure that you are in the root directory of the repository when running the command.

## Code Sections

- `/data`: Contains data files used in to analize or plot
- `/lib`: Contains common functionalities library codes
- `/plot`: Contains scripts for plotting data
- `/tests`: Contains unit test files

## Contributing

We welcome pull requests. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
