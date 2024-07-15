# Genetic Algorithm to improve the reliability of PUFs 

In this code, a genetic algorithm is used to improve the reliability of PUFs. To illustrate the operation, an RTN-based PUF is considered.

## Installation

To run the Python scripts in this repository using Conda, follow these steps:

### 1. Clone the Repository

First, clone this repository to your local machine using `git`:

```bash
git clone https://github.com/FernandoUSS/GARTNPUF.git
```

### 2. Navigate to the Repository

Navigate into the cloned repository directory:

```bash
cd GARTNPUF
```

### 3. Create, Install Required Packages, and Activate Conda Environment
Create a new Conda environment (e.g., named myenv), install the required Python packages listed in requirements.txt, and activate it:

```bash
conda create --name myenv --file requirements.txt
conda activate myenv
```

### 4. Run the Script
You can now run the Python script. For example, to run Main.py, use the following command:

```bash
python -u Main.py
```

Replace `Main.py` with the name of your Python script. Ensure that you are in the root directory of the repository when running the command.

## Code Sections

- `/data`: Contains data files used in to analize or plot
- `/optimization_algorithm`: Contains the code with the optimization algorithm and the file configGA.ini to choose the parameters
- `/plot`: Contains scripts for plotting data
- `/evaluation`: Contain a script to evaluate the results from the optimization algorithm

## Contributing

We welcome pull requests. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
