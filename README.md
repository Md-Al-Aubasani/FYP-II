# FYP-II
This is the FYP-II repo of S2102848.

For most files, make sure to replace the value `YOUR_API_KEY` with your Mistral API key.

### `Creating_Vector_Store.ipynb`
This notebook focuses on how from a given dataset of `jsonl` files, a vector store can be generated. The medical `textbooks` from MedRAG's Huggingface datasets (https://huggingface.co/MedRAG) were used as an example, as it was less than 15GB and could easily be stored in someone's G-Drive.

### `FYP_Framework_Drafts.ipynb`
This notebook focuses on how the framework works with the user-based context and the medically derived context.

### `FYP_Testing.ipynb`
This notebook contains the evaluations of similarity. It also displays how this framework can be evaluated on the MMLU-Med and BIO-ASQ benchmarks, here it uses the `textbooks` dataset from MedRAG as an example.

### `app-test.py`
This python file contains the completed Streamlit app. It can be run on Google Colab following the steps below:
1. Clone the git repository
2. Install the required libraries from `requirements.txt`
3. Make sure to replace the value `YOUR_API_KEY` with your Mistral API key.
4. Import your G-drive that contains the FAISS index files
```
from google.colab import drive
drive.mount('/content/drive')
```
5. Run the below code to get your tunnel password.
```
!wget -q -O - ipv4.icanhazip.com
```
6. Run the below code, to obtain a tunnel website link.
```
!streamlit run /content/FYP-II/app-test.py &>/content/logs.txt &
!npx localtunnel --port 8501
```
7. Click on the link and put in the tunnel password. You'll be able to run the Streamlit app in your browser with no problems.

### `csv_folder`
This folder contains the csv files that were generated and used in `FYP_Testing.ipynb`.

### `dataset_folder`
This folder contains 74 patient-doctor transcriptions obtained from: https://github.com/otto-dev/HealthLLM
