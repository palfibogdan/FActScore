import json
import itertools
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd


def get_refcheck_pred():
    with open("/home/palfib/refchecker/results/refchecker_selfcheck_output.json", "r") as f:
        results = json.load(f)

    pred_hard = [True if x['Y'] == "Entailment" else False for x in results]
    triplets = [x["triplets"] for x in results]
    pred_triplets = [x["ys"] for x in results]
    responses = [x["response"] for x in results]

    return pred_hard, triplets, pred_triplets, responses


def aggregate_results(results_path, af_path):
    with open(results_path, 'r') as f:
        results = json.load(f)

    results = list(itertools.chain.from_iterable(results['decisions']))
    atom_list = [x["atom"] for x in results]
    atom_list_copy = atom_list.copy()
    results = [x["is_supported"] for x in results]

    with open(af_path, 'r') as f:
        af_dict = json.load(f)

    final_results = {}
    for key in af_dict.keys():
        final_results[key] = []

    atoms_dict = []
    for key, atoms in af_dict.items():
        if key.startswith("In 2010, Hurley founded the heavy metal"):
            print("HERE")
        atoms_dict.append(atoms)
        num_atoms = len(atoms)

        if len(atom_list) > 0 and len(atoms) > 0:
            while atom_list[0] != atoms[0]:
                atom_list.pop(0)
                results.pop(0)

        for iter in range(num_atoms):
            atom = atom_list.pop(0)
            final_results[key].append(results.pop(0))

    atoms_dict = list(itertools.chain.from_iterable(atoms_dict))
    if atom_list_copy == atoms_dict:
        print("TRUE")
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
aggreagation = "strict"  # soft, strict

if data_name == "selfcheck":
    results_path = "/home/palfib/factscore/results/dataset_selfcheck_factscore_zephyr_output.json"
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

    ext_knowledge = []
    for dp in data:
        ext_knowledge.append(dp["wiki_bio_text"])

    with open("/home/palfib/factscore/data/selfcheck_ext_knowledge.json", 'w') as f:
        json.dump(ext_knowledge, f, indent=4)

    refcheck_pred, triplets, pred_triplets, responses = get_refcheck_pred()
    # gt.pop(1115)
    # pred.pop(1115)
    # for idx, (gt_val, pred_val) in enumerate(zip(gt, pred)):
    #     if gt_val != pred_val and pred_val == False:
    #         sentence = list(decisions.keys())[idx]
    #         idx_ref = responses.index(sentence)
    #         print(idx, sentence, "GT: ", gt_val, " FAct: ", pred_val, " Ref:", refcheck_pred[idx_ref])
    #         print(af_dict[sentence])
    #         print(final_results[sentence])
    #
    #         print(triplets[idx_ref])
    #         print(pred_triplets[idx_ref])
    #         print("\n")

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
(precision, recall, f1, _) = precision_recall_fscore_support(gt, pred, average='binary', pos_label=True,
                                                             zero_division=1.0)
accuracy = accuracy_score(gt, pred)

print("Prec: ", precision)
print("Rec: ", recall)
print("F1: ", f1)
print("Acc: ", accuracy)
print("Done")

with open("/home/palfib/factscore/results/dataset_selfcheck_factscore_output.json", 'r') as f:
    results_old = json.load(f)
