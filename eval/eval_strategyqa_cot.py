import json

def getAnswer(response):
    """Extract Yes/No answer from response."""
    pred = response.split("answer is")[-1]
    if "Yes" in pred:
        return "Yes"
    elif "No" in pred:
        return "No"
    return ""

# Load predictions
prediction = []
with open(f"/cpfs01/user/xufangzhi/o1/infer/results/strategyqa_test_241202-3_sir_no_replace_rollout_0_foresight_0.json") as file:
    for line in file:
        prediction.append(json.loads(line))

print(len(prediction))

# Evaluate predictions
correct_num = 0
for i in range(len(prediction)):
    response = prediction[i]['response'].strip()
    pred = getAnswer(response)
    gt = prediction[i]['ground_truth']

    try:
        if gt[pred] == 1:
            correct_num += 1
    except:
        continue

# Print accuracy
print(correct_num / len(prediction))
