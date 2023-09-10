# Boilerplate for Python script

This is a sample template that you can use for development in Python. You may refer to this template on how you can create structure your python project, create virtual environment, run unit testing.

## Getting Started

### Installing

You can clone this folder from the git repository 

```bash 
```

### Prerequisites

Install the `virtualenv` if you do not have it

```bash 
pip install virtualenv
```

Create a new virtual environment name `my_env`

```bash
virtualenv my_env
```

Activate the environment via

```bash
source ./my_env/bin/activate
```

Lastly, install the listed packages in `requirements.txt` file

```bash
pip install -r requirements.txt
```
To deactivate the environment, run the following command:

```bash
deactivate
```
If you wish to export the list of dependencies into `requirements.txt`, use the following command:
```bash
pip freeze > requirements.txt
```

### Usage Example

To run the program with default settings
```bash
python boilerplate.py
```

The output files will be in
```bash
./output/
```

More details on the arguments can shown if you run
```bash
python boilerplate.py --help
```

```bash
sage: boilerplate.py [-h] [-i INPUT_FILE] [-o OUT_FILE_PATH] [-d DATE_FORMAT]

options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        Specify input file
  -o OUT_FILE_PATH, --output-folder OUT_FILE_PATH
                        Specify output directory
  -d DATE_FORMAT, --date DATE_FORMAT
                        Specify date in YYYYmmdd UTC format
```

## Directory structure

### Folder Structure

```bash
├── config
│   ├── log.config (config for logger)
│   ├── config.json (config for URL)
├── data
├── requirements.txt
├── tests
├── logs 
├── boilerplate.py
├── README.md
└── .gitignore
```

## Config

This section describes what is the purpose of the config.

```json
{
  "config": {
    "prefix_url": "https://portal.nccs.nasa.gov/datashare/GlobalFWI/v2.0/fwiCalcs.GEOS-5/Default/GPM.EARLY.v5"
  }
}
```

## Linting
Flake8 is a python linting tool that helps you to check your Python code for errors, styling issues and complexity. It is a very popular and easy-to-use tool. You can install it via pip package manager:

```bash
pip install flake8
```
Once you have installed flake8, you can run it via the following command:

```bash
flake8 boilerplate.py
```

In our case, there are a few minor styling issues that pop up

```python
boilerplate.py:33:80: E501 line too long (82 > 79 characters)
boilerplate.py:35:80: E501 line too long (83 > 79 characters)
boilerplate.py:36:80: E501 line too long (84 > 79 characters)
```
## Running the tests
You can run `pytest` simply via the following command:

```bash
pytest
```

`pytest` will run all the unit tests under the folder of `tests`

If the tests are successful, it should show something like this 


```python
=============================================================================== test session starts ===============================================================================
platform darwin -- Python 3.10.9, pytest-7.4.0, pluggy-1.2.0
rootdir: /Users/songhanwong/Library/CloudStorage/GoogleDrive-songhan89@gmail.com/My Drive/Projects/python-script
collected 1 item                                                                                                                                                                  

tests/test_loader.py .                                                                                                                                                      [100%]

================================================================================ 1 passed in 0.92s ================================================================================
```


## Authors








