import csv
import os

f = open('data/trainLabels.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
	if line[0] != 'id':
		file_name = line[0] + '.png'
		file_target = line[1]

		file_path = './data/train/'+file_target+'/'
		print('./data/train/'+file_name, file_path+file_name)
		try:
			os.rename('./data/train/'+file_name, file_path+file_name)
		except FileNotFoundError:
			os.mkdir(file_path)
			os.rename('./data/train/'+file_name, file_path+file_name)
		except:
			pass
		# break
f.close()