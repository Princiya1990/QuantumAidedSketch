import os
# Train SRGAN
train = "python train.py"
print("Training SRGAN...")
os.system(train)

os.makedirs("results", exist_ok=True)

test = "python train.py --mode eval"
print("Running inference on LR dataset...")
os.system(test)
