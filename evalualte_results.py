import json
import itertools
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd


def aggregate_results(results_path, af_path):
    with open(results_path, 'r') as f:
        results = json.load(f)

    with open(af_path, 'r') as f:
        af_dict = json.load(f)

    final_results = {}
    for key in af_dict.keys():
        final_results[key] = []

    for dp in results['decisions']:
        for fact in dp:
            for key, atoms in af_dict.items():
                if fact["atom"] in atoms:
                    final_results[key].append(fact["is_supported"])
                    break

    return final_results


def get_gt(data, pred_sentences, data_name):
    gt = None
    if data_name == "selfcheck":
        annotations = [x["annotation"] for x in data]
        annotations = list(itertools.chain.from_iterable(annotations))
        sentences = [x["gpt3_sentences"] for x in data]
        sentences = list(itertools.chain.from_iterable(sentences))
        for idx, sentence in enumerate(sentences):
            if sentence not in pred_sentences:
                sentences.pop(idx)
                annotations.pop(idx)
        gt = [True if x == "accurate" else False for x in annotations]
        return gt
    elif data_name == "factscore":
        annotations = []
        all_atoms = []
        no_annotations = []
        for idx, datapoint in data.iterrows():
            try:
                current_label = []
                atoms_dp = []
                for sent in datapoint["annotations"]:
                    if sent["human-atomic-facts"] is None:
                        continue
                    for atom in sent["human-atomic-facts"]:
                        current_label.append(atom["label"])
                        atoms_dp.append(atom["text"])

                # current_label = [atom["label"] for sent in datapoint["annotations"] for atom in
                #                  sent["human-atomic-facts"]]
                # atoms_dp = [atom["text"] for sent in datapoint["annotations"] for atom in sent["human-atomic-facts"]]
                annotations.append(current_label)
                all_atoms.append(atoms_dp)
            except Exception as e:
                # print(e)
                no_annotations.append(idx)
                annotations.append([])

        all_atoms = list(itertools.chain.from_iterable(all_atoms))
        annotations = list(itertools.chain.from_iterable(annotations))
        for idx, pred_s in enumerate(pred_sentences):
            if pred_s not in all_atoms:
                print(pred_s)
                # all_atoms.pop(idx)
                # annotations.pop(idx)

        gt = [True if x == "S" else False for x in annotations]

        return gt, no_annotations


data_name = "selfcheck"
aggreagation = "soft"  # soft, strict

if data_name == "selfcheck":
    results_path = "/home/palfib/factscore/results/dataset_selfcheck_factscore_output.json"
    af_path = "/home/palfib/factscore/results/selfcheck_af.json"
    data = json.load(open("/home/palfib/factscore/data/dataset_selfcheck.json", 'r'))

    final_results = aggregate_results(results_path, af_path)

    # for all results, if the dictionary contains false then it is not supported
    decisions = {}
    if aggreagation == 'strict':
        for key, value in final_results.items():
            if False in value:
                decisions[key] = False
            else:
                decisions[key] = True
    else:
        for key, value in final_results.items():
            num_true = value.count(True)
            num_false = value.count(False)
            decisions[key] = True if num_true > num_false and num_false < 2 else False
    pred = list(decisions.values())

    with open(af_path, 'r') as f:
        af_dict = json.load(f)
    pred_sentence = af_dict.keys()

    gt = get_gt(data, list(pred_sentence), data_name)

else:
    results_path = "results/ChatGPT_factscore_output.json"

    # af_path = "/home/palfib/factscore/results/selfcheck_af.json"
    data = pd.read_json("/home/palfib/factscore/data/labeled/ChatGPT.jsonl", lines=True)
    with open(results_path, 'r') as f:
        results = json.load(f)
    pred = []
    atoms_pred = []
    for dp in results['decisions']:
        for fact in dp:
            pred.append(fact["is_supported"])
            atoms_pred.append(fact["atom"])

    gt, no_annotation = get_gt(data, atoms_pred, data_name)
    # pred = [x for idx, x in enumerate(pred) if idx not in no_annotation]
    pred = [x for idx, x in enumerate(pred)]

# if gt and pred are not the same, print the decision key
# for idx, (gt_val, pred_val) in enumerate(zip(gt, pred)):
#     if gt_val != pred_val:
#         print(list(decisions.keys())[idx], gt_val, pred_val)
#         a = 2

(precision, recall, f1, _) = precision_recall_fscore_support(gt, pred, average='binary', pos_label=True,
                                                             zero_division=1.0)
accuracy = accuracy_score(gt, pred)

print("Prec: ", precision)
print("Rec: ", recall)
print("F1: ", f1)
print("Acc: ", accuracy)
print("Done")
