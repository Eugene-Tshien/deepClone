import json
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gc

from deepstochlog.network import Network
from deepstochlog.utils import (
    set_fixed_seed,
)
from deepstochlog.term import Term

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t5_type = "t5-small"
except_switcher = Network(
    "except_switcher",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/type", str(64) + "_" + str(3) + "_" + str(5e-4))),
    index_list=None,
    network_type=t5_type,
    concat_tensor_input=False,
)
tab_picker = Network(
    "tab_picker",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/table", str(64) + "_" + str(5) + "_" + str(1e-3))),
    index_list=None,
    network_type=t5_type,
    concat_tensor_input=False,
)
col_picker = Network(
    "col_picker",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/column", str(32) + "_" + str(16) + "_" + str(5e-4))),
    index_list=None,
    network_type=t5_type,
    concat_tensor_input=False,
)
select_switcher = Network(
    "select_switcher",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/ss", str(64) + "_" + str(7) + "_" + str(1e-3))),
    index_list=[Term(str(e)) for e in range(10)],
    network_type=t5_type,
    concat_tensor_input=False,
)
where_switcher = Network(
    "where_switcher",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/where", str(32) + "_" + str(14) + "_" + str(5e-4))),
    index_list=[Term(str(e)) for e in range(2)],
    network_type=t5_type,
    concat_tensor_input=False,
)
op_switcher = Network(
    "op_switcher",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/op", str(64) + "_" + str(17) + "_" + str(1e-3))),
    index_list=None,
    network_type=t5_type,
    concat_tensor_input=False,
)
groupby_switcher = Network(
    "groupby_switcher",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/group_by", str(64) + "_" + str(3) + "_" + str(1e-3))),
    index_list=[Term(str(e)) for e in range(2)],
    network_type=t5_type,
    concat_tensor_input=False,
)
having_switcher = Network(
    "having_switcher",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/having", str(64) + "_" + str(2) + "_" + str(5e-4))),
    index_list=[Term(str(e)) for e in range(2)],
    network_type=t5_type,
    concat_tensor_input=False,
)
orderby_switcher = Network(
    "orderby_switcher",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/order_by", str(64) + "_" + str(7) + "_" + str(1e-3))),
    index_list=[Term(str(e)) for e in range(2)],
    network_type=t5_type,
    concat_tensor_input=False,
)
desc_switcher = Network(
    "desc_switcher",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/desc", str(32) + "_" + str(10) + "_" + str(5e-4))),
    index_list=[Term(str(e)) for e in range(2)],
    network_type=t5_type,
    concat_tensor_input=False,
)
limit_switcher = Network(
    "limit_switcher",
    T5ForConditionalGeneration.from_pretrained(os.path.join("models/limit", str(64) + "_" + str(3) + "_" + str(5e-4))),
    index_list=[Term(str(e)) for e in range(2)],
    network_type=t5_type,
    concat_tensor_input=False,
)
col_cf_picker = Network(
    "col_cf_picker",
    T5ForConditionalGeneration.from_pretrained(
        os.path.join("models/column_cf", str(32) + "_" + str(13) + "_" + str(5e-4))),
    index_list=None,
    network_type=t5_type,
    concat_tensor_input=False,
)


def build_prompt(inputs, name, domain=None):
    prompt = inputs[0] + " "
    if "switcher" in name:
        if name == "select_switcher":
            grammar_option = ["*", "COUNT(*)", "column", "COUNT(column)", "SUM(column)", "AVG(column)", "MIN(column)", "MAX(column)"]
        elif name == "groupby_switcher":
            grammar_option = ["empty", "GROUP BY"]
        elif name == "orderby_switcher":
            grammar_option = ["empty", "ORDER BY"]
        elif name == "desc_switcher":
            grammar_option = ["ASC", "DESC"]
        elif name == "except_switcher":
            grammar_option = ["empty", "EXCEPT"]
        elif name == "where_switcher":
            grammar_option = ["empty", "WHERE"]
        elif name == "having_switcher":
            grammar_option = ["empty", "HAVING"]
        elif name == "limit_switcher":
            grammar_option = ["empty", "LIMIT"]
        elif name == "op_switcher":
            if inputs[1] == 0:
                grammar_option = ["=", "!=", ">", "<", ">=", "<=", "like"]
                prompt += "WHERE column [operator], "
            elif inputs[1] == 1:
                grammar_option = ["=", ">", "<", ">=", "<="]
                prompt += "HAVING column [operator], "
        for index in range(len(grammar_option)):
            prompt += "Answer " + str(index + 1) + " for " + grammar_option[index] + ", "
    else:
        if name == "col_picker" or name == "col_cf_picker":
            ref = inputs[1]+inputs[2]
            if len(inputs) ==4:
                if inputs[3] == 0:
                    prompt += "SELECT [column], "
                elif inputs[3] == 1:
                    prompt += "WHERE [column], "
                elif inputs[3] == 2:
                    prompt += "GROUP BY [column], "
                elif inputs[3] == 3:
                    prompt += "ORDER BY [column], "
                elif inputs[3] == 4:
                    prompt += "EXCEPT [column], "
        elif name == "tab_picker":
            ref = None
            if inputs[1] == 0:
                prompt += "SELECT FROM [table], "
            elif inputs[1] == 1:
                prompt += "EXCEPT [table], "
        if ref is not None and ref not in domain:
            raise Exception(
                "Index was not found, did you include the right Term list as keys? Error item: "
                + str(ref)
                + " "
                + str(type(ref))
                + ".\nPossible values: "
                + ", ".join([str(k) for k in domain.keys()])
            )
        if ref is not None:
            domain = domain[ref]
        for index in range(len(domain)):
            prompt += "Answer " + str(int(index)+1) + " for " + domain[index] + ", "
    prompt += "the answer should be Answer "

    return prompt


def get_domain(db, type=None, tables=[], context_sensitive=True):
    with open("src/task2/schema_task2.json", 'r') as f:
        schema = json.load(f)
    schema_parts = schema[db].split("\n")
    if type == "tables":
        tables = schema_parts[0][schema_parts[0].index("[")+1:schema_parts[0].index("]")].split(", ")
        tables = [token.replace("'", "") for token in list(tables)]
        return tables
    elif type == "foreign_tables":
        tables = schema_parts[1][schema_parts[1].index("[") + 1:schema_parts[1].index("]")].split(", ")
        tables = [token.replace("'", "") for token in list(tables)]
        return tables
    elif type == "foreign_relations":
        foreign_schema = [item for item in schema_parts if item.startswith("table_foreign(") and not item == ""]
        if not context_sensitive:
            f_domain = schema_parts[1][schema_parts[1].index("[") + 1:schema_parts[1].index("]")].split(", ")
            f_domain = [item.replace("'", "") for item in f_domain]
        else:
            schema_f_index = [i for i in range(len(foreign_schema)) if foreign_schema[i].startswith("table_foreign('" + db + "', '" + tables[0] + "'")]
            f_schema = foreign_schema[schema_f_index[0]]
            f_domain = f_schema[f_schema.index("[") + 1:f_schema.index("]")].split(", ")
            f_domain = [item.replace("'", "") for item in f_domain]
        return f_domain
    elif type == "foreign_keys":
        foreign_schema = [item for item in schema_parts if item.startswith("foreign_key(") and not item == ""]
        schema_f = "foreign_key('" + tables[0] + "', '" + tables[1] + "'"
        schema_f_index = [i for i in range(len(foreign_schema)) if
                          foreign_schema[i].startswith(schema_f)]
        f_schema = foreign_schema[schema_f_index[0]]
        f_domain = f_schema[f_schema.index("(") + 1:f_schema.index(")")].split(", ")
        f_domain = [item.replace("'", "") for item in f_domain]
        return f_domain[2], f_domain[3]
    elif type == "columns":
        column_domain = {}
        if not context_sensitive:
            columns = schema_parts[2][schema_parts[2].index("[") + 1:schema_parts[2].index("]")].split(", ")
            columns = [item.replace("'", "") for item in columns]
            column_domain[db+"cf"]=columns
        else:
            c_schema = [item for item in schema_parts if item.startswith("table(") and not item == ""]
            schema_c_index = [i for i in range(len(c_schema)) if
                              c_schema[i].startswith("table('" + db + "', '" + tables[0] + "'")]
            c_schema = c_schema[schema_c_index[0]]
            columns = c_schema[c_schema.index("[") + 1:c_schema.index("]")].split(", ")
            columns = [token.replace("'", "") for token in columns]
            column_domain[db+tables[0]]=columns
        return column_domain

def greedy_nn(model, prompt, domain=None):
    model.to(device)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(t5_type)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device, dtype=torch.long)
    input_mask = inputs["attention_mask"].to(device, dtype=torch.long)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            max_length=2
        )
        pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in output_ids]
        if domain is not None:
            try:
                pred = domain[int(pred[0])-1]
            except:
                pred = domain[-1]
        else:
            pred = int(pred[0])-1
    return pred


def greedy_col(nl, db, tab, state, context_sensitive=True):
    if not context_sensitive:
        col_domain = get_domain(db, "columns", [], context_sensitive)
        col_prompt = build_prompt([nl, db, "cf", state], "col_cf_picker", col_domain)
        col = greedy_nn(col_cf_picker.neural_model, col_prompt, col_domain[db + "cf"])
    else:
        col_domain = get_domain(db, "columns", [tab])
        col_prompt = build_prompt([nl, db, tab, state], "col_picker", col_domain)
        col = greedy_nn(col_picker.neural_model, col_prompt, col_domain[db + tab])
    return col


def mapnn2or(db, tables, type, columns=None):
    with open("src/nn2or.json", 'r') as f:
        nn2or = json.load(f)
    all_res = []
    if type == "columns":
        for i in range(len(columns)):
            res = ""
            col_tab = []
            cols = nn2or[db][type].keys()
            for c in cols:
                if c.lower() == columns[i].lower():
                    for item in nn2or[db][type][c]:
                        col_tab.append(item)
            if col_tab == []:
                print("no or col match between gt query and mapping file!")
            tabs = [pair[1] for pair in col_tab]
            cols = [pair[0] for pair in col_tab]
            if tables == []:
                res = cols[0]
            else:
                for index in range(len(tabs)):
                    t = tabs[index]
                    if t == tables[i] or t.lower() == tables[i].lower():
                        res = cols[index]
            if res == "":
                print(db)
                print(columns[i])
                print(tables[i])
                print(col_tab)
                print("no or2nn col match, attention!")
            else:
                all_res.append(res)
    else:
        for table in tables:
            res = ""
            try:
                res=nn2or[db][type][table]
            except:
                tabs = nn2or[db][type].keys()
                for t in tabs:
                    if t.lower() == table.lower():
                        res=nn2or[db][type][t]
            if res == "":
                # print(db)
                # print(table)
                print("no or2nn tab match, attention!")
            else:
                all_res.append(res)
    return all_res


def greedy_evaluation(test_data=None, context_sensitive=True, values=None):
    match = 0
    pred = []

    for index in range(len(test_data)):
        print(index)
        sample = test_data[index]

        nl = sample["question"]
        db = sample["db_id"]
        gt_query = sample["query"]

        except_prompt = build_prompt([nl], "except_switcher")
        ex = greedy_nn(except_switcher.neural_model, except_prompt)
        if ex == 1:
            tab_foreign_domain = get_domain(db, "foreign_tables")
            tab_prompt = build_prompt([nl, 0], "tab_picker", tab_foreign_domain)
            tab1 = greedy_nn(tab_picker.neural_model, tab_prompt, tab_foreign_domain)
            if not context_sensitive:
                tab_prompt = build_prompt([nl, 1], "tab_picker", tab_foreign_domain)
                tab2 = greedy_nn(tab_picker.neural_model, tab_prompt, tab_foreign_domain)
                col1 = greedy_col(nl, db, "cf", 0, context_sensitive)
                col2 = greedy_col(nl, db, "cf", 4, context_sensitive)
                tabs = mapnn2or(db, [tab1, tab2], "tables")
                cols = mapnn2or(db, [], "columns", [col1, col2])
            else:
                foreign_tabs = get_domain(db, "foreign_relations", [tab1])
                if len(foreign_tabs) == 1:
                    tab2 = foreign_tabs[0]
                else:
                    tab_prompt = build_prompt([nl, 1], "tab_picker", foreign_tabs)
                    tab2 = greedy_nn(tab_picker.neural_model, tab_prompt, foreign_tabs)
                col1, col2 = get_domain(db, "foreign_keys", [tab1, tab2])
                tabs = mapnn2or(db, [tab1, tab2], "tables")
                cols = mapnn2or(db, [tabs[0], tabs[1]], "columns", [col1, col2])
            pred_query = "SELECT " + cols[0] + " FROM " + tabs[0] + " EXCEPT SELECT " + cols[1] + " FROM " + tabs[1]
            pred.append(pred_query+"\n")
            if pred_query.lower() == gt_query.lower():
                match += 1
            # else:
            #     print(pred_query)
            #     print(gt_query)

        else:
            tab_domain = get_domain(db, "tables")
            tab_prompt = build_prompt([nl, 0], "tab_picker", tab_domain)
            tab = greedy_nn(tab_picker.neural_model, tab_prompt, tab_domain)

            tab_or = mapnn2or(db, [tab], "tables")[0]
            pred_query = ["SELECT"]

            select_prompt = build_prompt([nl], "select_switcher")
            select = greedy_nn(select_switcher.neural_model, select_prompt)

            if select == 0:
                pred_query.append("*")
            elif select == 1:
                pred_query.append("COUNT(*)")
            elif select == 2:
                col = greedy_col(nl, db, tab, 0, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
            elif select == 3:
                pred_query.append("DISTINCT")
                col = greedy_col(nl, db, tab, 0, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
            elif select == 4:
                pred_query.append("COUNT(")
                col = greedy_col(nl, db, tab, 0, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
                pred_query.append(")")
            elif select == 5:
                pred_query.extend(["COUNT(", "DISTINCT"])
                col = greedy_col(nl, db, tab, 0, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
                pred_query.append(")")
            elif select == 6:
                pred_query.append("SUM(")
                col = greedy_col(nl, db, tab, 0, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
                pred_query.append(")")
            elif select == 7:
                pred_query.append("AVG(")
                col = greedy_col(nl, db, tab, 0, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
                pred_query.append(")")
            elif select == 8:
                pred_query.append("MIN(")
                col = greedy_col(nl, db, tab, 0, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
                pred_query.append(")")
            elif select == 9:
                pred_query.append("MAX(")
                col = greedy_col(nl, db, tab, 0, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
                pred_query.append(")")

            pred_query.extend(["FROM", tab_or])

            where_prompt = build_prompt([nl], "where_switcher")
            where = greedy_nn(where_switcher.neural_model, where_prompt)

            if where == 1:
                pred_query.append("WHERE")
                col = greedy_col(nl, db, tab, 1, context_sensitive)
                op_prompt = build_prompt([nl, 0], "op_switcher")
                op = greedy_nn(op_switcher.neural_model, op_prompt)
                op_domain = ["=", "!=", ">", "<", ">=", "<=", "like"]
                if index in values.keys() and "where" in values[index].keys():
                    value = values[index]["where"]
                else:
                    value = str(1)
                if not context_sensitive:
                    pred_query.extend([mapnn2or(db, [], "columns", [col])[0], op_domain[op], value])
                else:
                    pred_query.extend([mapnn2or(db, [tab_or], "columns", [col])[0], op_domain[op], value])


            groupby_prompt = build_prompt([nl], "groupby_switcher")
            groupby = greedy_nn(groupby_switcher.neural_model, groupby_prompt)

            if groupby == 1:
                pred_query.append("GROUP BY")
                col = greedy_col(nl, db, tab, 2, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
                having_prompt = build_prompt([nl], "having_switcher")
                having = greedy_nn(having_switcher.neural_model, having_prompt)
                if having == 1:
                    op_prompt = build_prompt([nl, 1], "op_switcher")
                    op = greedy_nn(op_switcher.neural_model, op_prompt)
                    op_domain = ["=", ">", "<", ">=", "<="]
                    if index in values.keys() and "having" in values[index].keys():
                        value = values[index]["having"]
                    else:
                        value = str(1)
                    pred_query.extend(["HAVING", "COUNT(*)", op_domain[op], value])

            orderby_prompt = build_prompt([nl], "orderby_switcher")
            orderby = greedy_nn(orderby_switcher.neural_model, orderby_prompt)

            if orderby == 1:
                pred_query.append("ORDER BY")
                col = greedy_col(nl, db, tab, 3, context_sensitive)
                if not context_sensitive:
                    pred_query.append(mapnn2or(db, [], "columns", [col])[0])
                else:
                    pred_query.append(mapnn2or(db, [tab_or], "columns", [col])[0])
            if not orderby == 0:
                desc_prompt = build_prompt([nl], "desc_switcher")
                desc = greedy_nn(desc_switcher.neural_model, desc_prompt)
                limit_prompt = build_prompt([nl], "limit_switcher")
                limit = greedy_nn(limit_switcher.neural_model, limit_prompt)

                if desc == 0:
                    pred_query.append("ASC")
                elif desc == 1:
                    pred_query.append("DESC")
                if limit == 1:
                    if index in values.keys() and "limit" in values[index].keys():
                        value = values[index]["limit"]
                    else:
                        value = str(1)
                    pred_query.extend(["LIMIT", value])

            pred.append((" ").join(pred_query) + "\n")
            if gt_query.lower()==(" ").join(pred_query).lower():
                match += 1
            # else:
            #     print(gt_query)
            #     print((" ").join(pred_query))
    # print(match)
    if not context_sensitive:
        with open("src/task2/T5CFG.txt", "w") as of:
            for p in pred:
                of.write(p)
    else:
        with open("src/task2/Ours.txt", "w") as of:
            for p in pred:
                of.write(p)



def get_all_values(test_data):
    values = {}
    for index in range(len(test_data)):
        sample = test_data[index]
        query = sample["query"]
        values[index] = {}
        query_list = query.lower().split(" ")
        query_list = [item for item in query_list if not item == ""]
        if "where" in query_list:
            if "'" in query or '"' in query:
                values[index]["where"] = '"'+re.findall(r'["\'](.*?)["\']', query)[0]+'"'
            else:
                values[index]["where"] = query_list[query_list.index("where")+3].split(";")[0].replace('"', "")
        if "having" in query_list:
            values[index]["having"] = query_list[query_list.index("having") + 3].split(";")[0]
        if "limit" in query_list:
            values[index]["limit"] = query_list[query_list.index("limit")+1].split(";")[0]
    return values


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    set_fixed_seed(23)
    with open("data/test_task2.json", "r") as f:
        test_data = json.load(f)
    values = get_all_values(test_data)
    # whether table-unification is used, False for T5-small+CFG baseline
    context_sensitive = True
    greedy_evaluation(test_data=test_data, context_sensitive=context_sensitive, values = values)