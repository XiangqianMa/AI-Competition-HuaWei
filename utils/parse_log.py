
log_file = '/media/mxq/project/Projects/competition/HuaWei/AI-Competition-HuaWei/log.txt'
predict = []
with open(log_file, 'r') as f:
    for line in f.readlines():
        if 'result:' in line:
            line = line.split('result:')[1].strip()
            predict.append(line)

print("There are %d images in teset set." % len(predict))
label_to_number = {}
for label in predict:
    if label not in label_to_number.keys():
        label_to_number[label] = 1
    else:
        label_to_number[label] += 1
print(label_to_number)