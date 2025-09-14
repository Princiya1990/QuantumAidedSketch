import os
os.chdir("code")
combine="python combine_A_and_B.py --fold_A ./../data/dataset/photos --fold_B ./../data/dataset/sketches --fold_AB ./../data/dataset/face2sketch"
os.system(combine)
train = "python train.py --dataroot ./../data/dataset --name results --model pix2pix --direction AtoB"
# Execute the command
os.system(train)
test = "python train.py --dataroot ./../data/dataset --name results --model pix2pix --direction AtoB"
# Execute the command
os.system(test)
