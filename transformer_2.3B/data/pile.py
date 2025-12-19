from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", "default")

ds.to_csv("fineweb-edu.csv")