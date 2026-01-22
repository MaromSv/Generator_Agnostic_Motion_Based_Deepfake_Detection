import os
from modelscope.msdatasets import MsDataset

# Dataset Download
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, "data")
os.makedirs(data_folder, exist_ok=True)

print(f"Downloading dataset to: {data_folder}")
ds = MsDataset.load("cccnju/GenVideo-100K", cache_dir=data_folder)

print(f"✓ Dataset downloaded successfully to {data_folder}")
print(f"Dataset info: {ds}")

# Save dataset to disk in the data folder
output_path = os.path.join(data_folder, "GenVideo-100K")
print(f"Saving dataset to: {output_path}")
ds.save_to_disk(output_path)
print(f"✓ Dataset saved to {output_path}")
