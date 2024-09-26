## Prepare Development Environment

1. Install [Python](https://www.python.org/downloads/) (3.12 was used in testing) 

2. Install [FFMPEG](https://www.ffmpeg.org/download.html) 

    Windows

    To simplify setup ffmpeg.exe and ffprode.exe have been included in this repo.
    
    Mac or Linux

    ``` bash
    apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
    ```

3. Create a Virtual Environment using a Linux shell or **git bash** in Windows

    Windows

    ``` bash
    pip install virtualenv
    python -m venv ./.venv
    source ./.venv/Scripts/activate
    pip install -r requirements.txt
    ```

    Mac or Linux

    ``` bash
    pip install virtualenv
    python -m venv ./.venv
    source ./.venv/bin/activate
    pip install -r requirements.txt
    ```

    Test Environment
    
    Run the command below to make sure the virtual environment is activated.
    
    ``` bash
    which python
    ```

    Create an OpenAI Account and Obtain a Key
    
    1. Follow the instructions [here](https://platform.openai.com/docs/quickstart) to create your key.
    2. Create the file [key.py](./key.py) and add the following text using your key as indicated.
    ``` python
    openai_key = 'put_your_key_here'
    ```

## Launch Notebook

Run the command below from the virtual environment to launch this notebook in a browser then continue executing the cells below.

``` bash
jupyter notebook assistant.ipynb
```