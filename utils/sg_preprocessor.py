import json
import os
import argparse
import sys

task_list = ["center_embed", "center_embed_mod", "cleft", "cleft_mod", "fgd_subject", "fgd_object", "fgd_pp", "fgd_embed3", \
    "fgd_embed4", "fgd_hierarchy", "mvrr", "mvrr_mod", "nn_nv_rpl", "npi_orc_any", "npi_orc_ever", "npi_src_any", \
    "npi_src_ever", "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod", "number_orc", "number_prep", "number_src", \
    "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem", "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc", \
    "subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src"]


task_2_tag = {
    "center_embed": [{"plaus":[6,7], "implaus":[6,7]}], 
    "center_embed_mod": [{"plaus":[7,8], "implaus":[7,8]}], 
    "cleft": [{"np_mismatch":[6], "np_match":[6], "vp_mismatch":[5,6], "vp_match":[5,6]}], 
    "cleft_mod": [{"np_mismatch":[7], "np_match":[7], "vp_mismatch":[6,7], "vp_match":[6,7]}], 
    "fgd_subject": [{"what_nogap":[3], "that_nogap":[3], "what_gap":[4], "that_gap":[4]}], 
    "fgd_object": [{"what_nogap":[5], "that_nogap":[5], "what_gap":[6], "that_gap":[6]}], 
    "fgd_pp": [{"what_nogap":[7], "that_nogap":[7], "what_gap":[8], "that_gap":[8]}], 
    "fgd_embed3": [{"what_no-gap":[6], "that_no-gap":[6], "what_gap":[7], "that_gap":[7]}], 
    "fgd_embed4": [{"what_no-gap":[6], "that_no-gap":[6], "what_gap":[7], "that_gap":[7]}],
    "fgd_hierarchy": [{"what_nogap":[6], "that_nogap":[6], "what_subjgap":[6], "that_subjgap":[6]}, {"what_nogap":[9], "that_nogap":[9], "what_subjgap":[6], "that_subjgap":[6]}], 
    "mvrr": [{"reduced_ambig":[5], "unreduced_ambig":[5], "reduced_unambig":[5], "unreduced_unambig":[5]}], 
    "mvrr_mod": [{"reduced_ambig":[6], "unreduced_ambig":[6], "reduced_unambig":[6], "unreduced_unambig":[6]}], 
    "nn_nv_rpl": [{"nn_ambig":[5], "nn_unambig":[5]}, {"nv_ambig":[5], "nv_unambig":[5]}], 
    "npi_orc_any": [{"neg_pos":[8], "pos_pos":[8], "neg_neg":[8], "pos_neg":[8]}], 
    "npi_orc_ever": [{"neg_pos":[8], "pos_pos":[8], "neg_neg":[8], "pos_neg":[8]}], 
    "npi_src_any": [{"neg_pos":[8], "pos_pos":[8], "neg_neg":[8], "pos_neg":[8]}],
    "npi_src_ever": [{"neg_pos":[8], "pos_pos":[8], "neg_neg":[8], "pos_neg":[8]}], 
    "npz_ambig": [{"ambig_nocomma":[5], "ambig_comma":[5], "unambig_nocomma":[5], "unambig_comma":[5]}], 
    "npz_ambig_mod": [{"ambig_nocomma":[6], "ambig_comma":[6], "unambig_nocomma":[6], "unambig_comma":[6]}], 
    "npz_obj": [{"no-obj_no-comma":[5], "no-obj_comma":[5], "obj_no-comma":[5], "obj_comma":[5]}], 
    "npz_obj_mod": [{"no-obj_no-comma":[6], "no-obj_comma":[6], "obj_no-comma":[6], "obj_comma":[6]}], 
    "number_orc": [{"match_sing":[7], "mismatch_sing":[7], "match_plural":[7], "mismatch_plural":[7]}],
    "number_prep": [{"match_sing":[6], "mismatch_sing":[6], "match_plural":[6], "mismatch_plural":[6]}],
    "number_src": [{"match_sing":[7], "mismatch_sing":[7], "match_plural":[7], "mismatch_plural":[7]}], 
    "reflexive_orc_fem": [{"match_sing":[8], "mismatch_sing":[8], "match_plural":[8], "mismatch_plural":[8]}], 
    "reflexive_orc_masc": [{"match_sing":[8], "mismatch_sing":[8], "match_plural":[8], "mismatch_plural":[8]}],
    "reflexive_prep_fem": [{"match_sing":[7], "mismatch_sing":[7], "match_plural":[7], "mismatch_plural":[7]}],
    "reflexive_prep_masc": [{"match_sing":[7], "mismatch_sing":[7], "match_plural":[7], "mismatch_plural":[7]}],
    "reflexive_src_fem": [{"match_sing":[8], "mismatch_sing":[8], "match_plural":[8], "mismatch_plural":[8]}], 
    "reflexive_src_masc": [{"match_sing":[8], "mismatch_sing":[8], "match_plural":[8], "mismatch_plural":[8]}], 
    "subordination": [{"sub_no-matrix":[3], "no-sub_no-matrix":[3], "sub_matrix":[3],  "no-sub_matrix":[3]}], 
    "subordination_orc-orc": [{"sub_no-matrix":[5], "no-sub_no-matrix":[5], "sub_matrix":[5],  "no-sub_matrix":[5]}], 
    "subordination_pp-pp": [{"sub_no-matrix":[5], "no-sub_no-matrix":[5], "sub_matrix":[5],  "no-sub_matrix":[5]}], 
    "subordination_src-src": [{"sub_no-matrix":[5], "no-sub_no-matrix":[5], "sub_matrix":[5],  "no-sub_matrix":[5]}]
}


class sg_testsuite():
    def __init__(self, task_name, json_path, tokenizer, output_path=None):
        self.task_name = task_name
        assert self.task_name in task_list
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.output_path = output_path
        self.loaddata()
        self.output = self.processdata()
        self.writefile()

    def loaddata(self):
        import json
        with open(self.json_path, 'r') as file:
            data = json.load(file)
        self.raw_data = data
        self.formulalist = [item["formula"] for item in self.raw_data["predictions"]]

    # output should be a list of list, each sublist means a iter and contains several dicts, each dict means a condition and contains condiction_name, tokenized_input, tokenized_tag
    def processdata(self):
        output = {}
        output["formula"] = self.formulalist
        output["data"] = []
        for iteration in self.raw_data["items"]:
            sublist = []
            for condition in iteration["conditions"]:
                cond_dict = {}
                cond_dict["condition_name"] = condition["condition_name"]
                cond_dict["input"] = ""
                cond_dict["tag"] = []
                for evaltype in range(len(task_2_tag[self.task_name])):
                    cond_dict["tag"].append([])
                for region in condition["regions"]:
                    if region["content"] == "":
                        continue
                    extented = " " + region["content"].strip()
                    cond_dict["input"] += extented
                    # print("input: ", region["content"])
                    # print("tokenizer_input:", self.tokenizer.encode(extented))
                    leng = len(self.tokenizer.encode(extented))
                    for evaltype in range(len(task_2_tag[self.task_name])):
                        if cond_dict["condition_name"] in task_2_tag[self.task_name][evaltype].keys() and region["region_number"] in task_2_tag[self.task_name][evaltype][cond_dict["condition_name"]]:
                            cond_dict["tag"][evaltype] += [1] * leng
                        else:
                            cond_dict["tag"][evaltype] += [0] * leng
                        # print("evaltype:", cond_dict["tag"][evaltype])
                cond_dict["input"] = cond_dict["input"].strip()
                sublist.append(cond_dict)
            output["data"].append(sublist)
        return output

    def writefile(self):
        if self.output_path != None:
            with open(self.output_path, "w") as json_file:
                json.dump(self.output, json_file, indent=2)

if __name__ == "__main__":
    cmd = argparse.ArgumentParser("Arguments for sg preprocessor")
    cmd.add_argument('--tokenizer_config_path', required=True, type=str, help='config for tokenizer')
    cmd.add_argument('--sg_dir', required=True, type=str, help='directory for sg dataset')
    cmd.add_argument('--output_dir', required=True, type=str, help='output directory for preprocessed sg dataset')

    args = cmd.parse_args(sys.argv[1:])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_config_path)
    
    for task_name in task_list:
        json_path = os.path.join(args.sg_dir, f"{task_name}.json")
        output_path = os.path.join(args.output_dir, f"{task_name}.json")
        sg_testsuite(task_name, json_path, tokenizer, output_path=output_path)