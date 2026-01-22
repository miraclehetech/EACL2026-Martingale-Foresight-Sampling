import json
from collections import Counter

def find_last_uppercase(input_str):
    """Extract the last uppercase letter (A-E) from response."""
    input_str = input_str.split("answer is")[-1]
    for char in input_str:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char
    return ""

# Mapping from letter to index
map_dict = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}

# Load predictions
prediction = []
with open(f"/mnt/d/phi-Decoding/result/ours-llama31-reclor-test-08.json") as file:
    for line in file:
        prediction.append(json.loads(line))

# Load ground truth
ground_truth = json.load(open("/mnt/d/phi-Decoding/data/reclor_val.json"))
ground_truth = [item['label'] for item in ground_truth]

# Evaluate predictions using majority voting
correct_num = 0
output = []
deduct = 0

for i in range(len(prediction)):
    response = prediction[i]['all_responses']
    answers = [find_last_uppercase(a) for a in response if a is not None]
    judge = Counter(answers).most_common(1)[0][0]
    
    if judge == "":
        deduct += 1
        continue
    
    pred = judge
    gt = ground_truth[i]
    
    if map_dict[pred] == gt:
        correct_num += 1
    else:
        output.append({
            "id": i,
            "ground_truth": gt,
            "pred": map_dict[pred]
        })

# Save incorrect predictions
with open("/mnt/d/phi-Decoding/result/ours-reclor-test-wrong.json", "w") as f:
    for item in output:
        f.write(json.dumps(item) + "\n")

# Print results
print(correct_num)
print(correct_num / (len(prediction) - deduct))
print(len(prediction))
print(deduct)
