import os
dirs = ['class_one', 'class_two']
for dir in dirs:
	i = -1;
	for file in os.listdir(dir):
		i = i + 1
		os.rename(os.path.join(dir, file), os.path.join(dir, '%03d.jpg' %i))
