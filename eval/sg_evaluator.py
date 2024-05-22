import torch
import re
import json

task_list = ["center_embed", "center_embed_mod", "cleft", "cleft_mod", "fgd_subject", "fgd_object", "fgd_pp", "fgd_embed3", \
    "fgd_embed4", "fgd_hierarchy", "mvrr", "mvrr_mod", "nn_nv_rpl", "npi_orc_any", "npi_orc_ever", "npi_src_any", \
    "npi_src_ever", "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod", "number_orc", "number_prep", "number_src", \
    "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem", "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc", \
    "subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src"]


formula_dict = {
    'center_embed': ['[ (6;%plaus%) + (7;%plaus%) ] < [ (6;%implaus%) + (7;%implaus%) ]'],
    'center_embed_mod': ['[ (7;%plaus%) + (8;%plaus%) ] < [ (7;%implaus%) + (8;%implaus%) ]'],
    'cleft': ['[(6;%np_mismatch%)-(6;%np_match%)]+[[(5;%vp_mismatch%)+(6;%vp_mismatch%)]-[(5;%vp_match%)+(6;%vp_match%)]]>0'],
    'cleft_mod': ['[(7;%np_mismatch%)-(7;%np_match%)]+[[(6;%vp_mismatch%)+(7;%vp_mismatch%)]-[(6;%vp_match%)+(7;%vp_match%)]]>0'],
    'fgd_subject': ['[(3;%what_nogap%) > (3;%that_nogap%) ] & [(4;%what_gap%) < (4;%that_gap%) ] '],
    'fgd_object': ['[(5;%what_nogap%) > (5;%that_nogap%) ] & [(6;%what_gap%) < (6;%that_gap%) ] '],
    'fgd_pp': ['[(7;%what_nogap%) > (7;%that_nogap%) ] & [(8;%what_gap%) < (8;%that_gap%) ] '],
    'fgd_embed3': ['[(6;%what_no-gap%)>(6;%that_no-gap%)] & [(7;%what_gap%)<(7;%that_gap%)]'],
    'fgd_embed4': ['[(6;%what_no-gap%)>(6;%that_no-gap%)] & [(7;%what_gap%)<(7;%that_gap%)]'],
    'fgd_hierarchy': ['[(6;%what_nogap%) >  (6;%that_nogap%)] & [(6;%what_subjgap%) <  (6;%that_subjgap%)]','[(9;%what_nogap%) =  (9;%that_nogap%)]& [(6;%what_subjgap%) =  (6;%that_subjgap%)]'],
    'mvrr': ['[(5;%reduced_ambig%) > (5;%unreduced_ambig%)] & [(5;%reduced_ambig%) > (5;%reduced_unambig%)] & [[(5;%reduced_ambig%) - (5;%unreduced_ambig%)] > [(5;%reduced_unambig%) - (5;%unreduced_unambig%)]]'],
    'mvrr_mod': ['[(6;%reduced_ambig%) > (6;%unreduced_ambig%)] & [(6;%reduced_ambig%) > (6;%reduced_unambig%)] & [[(6;%reduced_ambig%) - (6;%unreduced_ambig%)] > [(6;%reduced_unambig%) - (6;%unreduced_unambig%)]]'],
    'nn_nv_rpl': ['(5;%nn_ambig%)>(5;%nn_unambig%)', '(5;%nv_ambig%)>(5;%nv_unambig%)'], 
    'npi_orc_any': ['[ (8;%neg_pos%) < (8;%pos_pos%) ] & [ (8;%neg_neg%) < (8;%pos_neg%) ] & [ (8;%neg_pos%) < (8;%pos_neg%) ]'],
    'npi_orc_ever': ['[ (8;%neg_pos%) < (8;%pos_pos%) ] & [ (8;%neg_neg%) < (8;%pos_neg%) ] & [ (8;%neg_pos%) < (8;%pos_neg%) ]'],
    'npi_src_any': ['[ (8;%neg_pos%) < (8;%pos_pos%) ] & [ (8;%neg_neg%) < (8;%pos_neg%) ] & [ (8;%neg_pos%) < (8;%pos_neg%) ]'],
    'npi_src_ever': ['[ (8;%neg_pos%) < (8;%pos_pos%) ] & [ (8;%neg_neg%) < (8;%pos_neg%) ] & [ (8;%neg_pos%) < (8;%pos_neg%) ]'], 
    'npz_ambig': ['[(5;%ambig_nocomma%) > (5;%ambig_comma%) ] &  [(5;%ambig_nocomma%) > (5;%unambig_nocomma%) ]  & [[(5;%ambig_nocomma%) - (5;%ambig_comma%) ] > [(5;%unambig_nocomma%) - (5;%unambig_comma%) ]]'],
    'npz_ambig_mod': ['[(6;%ambig_nocomma%) > (6;%ambig_comma%) ] &  [(6;%ambig_nocomma%) > (6;%unambig_nocomma%) ]  & [[(6;%ambig_nocomma%) - (6;%ambig_comma%) ] > [(6;%unambig_nocomma%) - (6;%unambig_comma%) ]]'],
    'npz_obj': ['[(5;%no-obj_no-comma%) > (5;%no-obj_comma%) ] &  [(5;%no-obj_no-comma%) > (5;%obj_no-comma%) ] & [[(5;%no-obj_no-comma%) - (5;%no-obj_comma%) ] > [(5;%obj_no-comma%) - (5;%obj_comma%) ]]'],
    'npz_obj_mod': ['[(6;%no-obj_no-comma%) > (6;%no-obj_comma%) ] &  [(6;%no-obj_no-comma%) > (6;%obj_no-comma%) ] & [[(6;%no-obj_no-comma%) - (6;%no-obj_comma%) ] > [(6;%obj_no-comma%) - (6;%obj_comma%) ]]'],
    'number_orc': ['[(7;%match_sing%) < (7;%mismatch_sing%)] & [(7;%match_plural%) < (7;%mismatch_plural%)]'],
    'number_prep': ['[(6;%match_sing%) < (6;%mismatch_sing%)] & [(6;%match_plural%) < (6;%mismatch_plural%)]'],
    'number_src': ['[(7;%match_sing%) < (7;%mismatch_sing%)] & [(7;%match_plural%) < (7;%mismatch_plural%)]'],
    'reflexive_orc_fem': ['[ (8;%match_sing%) < (8;%mismatch_sing%) ] & [ (8;%match_plural%) < (8;%mismatch_plural%) ]'],
    'reflexive_orc_masc': ['[ (8;%match_sing%) < (8;%mismatch_sing%) ] & [ (8;%match_plural%) < (8;%mismatch_plural%) ]'],
    'reflexive_prep_fem': ['[ (7;%match_sing%) < (7;%mismatch_sing%) ] & [ (7;%match_plural%) < (7;%mismatch_plural%) ]'], 
    'reflexive_prep_masc': ['[ (7;%match_sing%) < (7;%mismatch_sing%) ] & [ (7;%match_plural%) < (7;%mismatch_plural%) ]'],
    'reflexive_src_fem': ['[ (8;%match_sing%) < (8;%mismatch_sing%) ] & [ (8;%match_plural%) < (8;%mismatch_plural%) ]'], 
    'reflexive_src_masc': ['[ (8;%match_sing%) < (8;%mismatch_sing%) ] & [ (8;%match_plural%) < (8;%mismatch_plural%) ]'],
    'subordination': ['[(3;%sub_no-matrix%) > (3;%no-sub_no-matrix%) ] & [(3;%sub_matrix%) < (3;%no-sub_matrix%) ]'], 
    'subordination_orc-orc': ['[(5;%sub_no-matrix%) > (5;%no-sub_no-matrix%) ] & [(5;%sub_matrix%) < (5;%no-sub_matrix%) ]'], 
    'subordination_pp-pp': ['[(5;%sub_no-matrix%) > (5;%no-sub_no-matrix%) ] & [(5;%sub_matrix%) < (5;%no-sub_matrix%) ]'], 
    'subordination_src-src': ['[(5;%sub_no-matrix%) > (5;%no-sub_no-matrix%) ] & [(5;%sub_matrix%) < (5;%no-sub_matrix%) ]']
}


new_formula_dict = {
    'center_embed': ['[ (%plaus%) ] < [ (%implaus%) ]'],
    'center_embed_mod': ['[ (%plaus%) ] < [ (%implaus%) ]'],
    
    'cleft': ['[ (%np_mismatch%) - (%np_match%) ] + [ [ (%vp_mismatch%) ] - [ (%vp_match%) ] ]>0'],
    'cleft_mod': ['[ (%np_mismatch%) - (%np_match%) ]+[ [ (%vp_mismatch%) ] - [ (%vp_match%) ] ]>0'],
    
    'fgd_subject': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_object': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_pp': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_embed3': ['[ (%what_no-gap%) > (%that_no-gap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_embed4': ['[ (%what_no-gap%) > (%that_no-gap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_hierarchy': ['[ (%what_nogap%) > (%that_nogap%)] & [ (%what_subjgap%) <  (%that_subjgap%) ]', '[ (%what_nogap%) = (%that_nogap%) ] & [ (%what_subjgap%) = (%that_subjgap%) ]'],
    
    'mvrr': ['[ (%reduced_ambig%) > (%unreduced_ambig%) ] & [ (%reduced_ambig%) > (%reduced_unambig%) ] & [ [ (%reduced_ambig%) - (%unreduced_ambig%) ] > [ (%reduced_unambig%) - (%unreduced_unambig%) ] ]'],
    'mvrr_mod': ['[ (%reduced_ambig%) > (%unreduced_ambig%) ] & [ (%reduced_ambig%) > (%reduced_unambig%) ] & [ [ (%reduced_ambig%) - (%unreduced_ambig%)] > [(%reduced_unambig%) - (%unreduced_unambig%)] ]'],
    
    'nn_nv_rpl': ['(%nn_ambig%)>(%nn_unambig%)', '(%nv_ambig%)>(%nv_unambig%)'], 
    
    'npi_orc_any': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_orc_ever': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_src_any': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_src_ever': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'], 
    
    'npz_ambig': ['[ (%ambig_nocomma%) > (%ambig_comma%) ] &  [ (%ambig_nocomma%) > (%unambig_nocomma%) ]  & [ [ (%ambig_nocomma%) - (%ambig_comma%) ] > [ (%unambig_nocomma%) - (%unambig_comma%) ] ]'],
    'npz_ambig_mod': ['[ (%ambig_nocomma%) > (%ambig_comma%) ] &  [ (%ambig_nocomma%) > (%unambig_nocomma%) ]  & [ [ (%ambig_nocomma%) - (%ambig_comma%) ] > [ (%unambig_nocomma%) - (%unambig_comma%) ] ]'],
    'npz_obj': ['[ (%no-obj_no-comma%) > (%no-obj_comma%) ] &  [ (%no-obj_no-comma%) > (%obj_no-comma%) ] & [ [ (%no-obj_no-comma%) - (%no-obj_comma%) ] > [ (%obj_no-comma%) - (%obj_comma%) ] ]'],
    'npz_obj_mod': ['[ (%no-obj_no-comma%) > (%no-obj_comma%) ] &  [ (%no-obj_no-comma%) > (%obj_no-comma%) ] & [ [ (%no-obj_no-comma%) - (%no-obj_comma%) ] > [ (%obj_no-comma%) - (%obj_comma%) ] ]'],
    
    'number_orc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'number_prep': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'number_src': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    
    'reflexive_orc_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_orc_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_prep_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'], 
    'reflexive_prep_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_src_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'], 
    'reflexive_src_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    
    'subordination': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_orc-orc': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_pp-pp': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_src-src': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]']
}


class Scorer():
    # load an input and get score
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def score(inputs, tags):
        pass


class Evaluator():
    # load the whole dataset of "task_name" and compute accuracy
    def __init__(self, task_path, model, device):
        self.task_path = task_path
        task_name = task_path.split('/')[-1][:-5]
        self.task_name = task_name
        assert self.task_name in task_list
        self.model = model
        self.device = device
        self.loaddata()
        self.scorer = Scorer(model, device)

    def loaddata(self):
        with open(self.task_path, 'r') as file:
            data = json.load(file)
        self.data = data

    def eval_math_expr(self, expr):
        try:
            return eval(expr)
        except:
            return math.nan
    
    def run(self, formula, input_list):
        keys = re.findall(r"%([\w|-]+)%", formula)
        keys = set(keys)

        score_dict = {}
        for item in input_list:
            score_dict[item["condition_name"]] = self.scorer.score(item["input"], item["tag"])
        # print(score_dict)
        
        for key in keys:
            formula = formula.replace(
                "(%{}%)".format(key),
                str(score_dict[key]),
                )
        formula = formula.replace("[", "(")
        formula = formula.replace("]", ")")
        # print(formula)
        return self.eval_math_expr(formula)

    def eval(self):
        # TODO: write new formula into processed json, deal with multiple formulas in one task
        formula = new_formula_dict[self.task_name][0]
        total_len = len(self.data["data"])
        po = 0
        for data in self.data["data"]:
            po += self.run(formula, data)
        acc = po / total_len
        return acc