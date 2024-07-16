import argparse
import re
import json
from pathlib import Path
import random
random.seed(23)

data_dir = Path(__file__).parent.parent / "data" / "spider" / "spider"


with open(data_dir / "train_spider.json", 'r') as file:
    train_spider = json.load(file)
with open(data_dir / "train_others.json", 'r') as file:
    train_other = json.load(file)
train = train_spider + train_other

# use spider development set for evaluation
with open(data_dir / "dev.json", 'r') as file:
    test = json.load(file)


with open(data_dir / "tables.json", 'r') as file:
    tables = json.load(file)

# fix the wrong column name mapping in tables.json
tables[128]["column_names_original"][1][1] = 'name'
tables[128]["column_names_original"][2][1] = 'seq'
tables[128]["column_names_original"][3][1] = 'id'
tables[128]["column_names_original"][4][1] = 'name'
tables[128]["table_names_original"][0] = 'sqlite_sequence'
tables[128]["table_names_original"][1] = 'artists'


def store_name_mappings(records, db_id, c, t):
    mappings = [c, t]
    for i in range(len(records)):
        records[i][db_id] = {}
        records[i][db_id]["tables"] = mappings[i+len(records)]
        records[i][db_id]["columns"] = mappings[i]


def get_name_mapping(type, keys, values, db=None):
    dict = {}
    if type == "column":
        for index in range(len(keys)):
            key = keys[index]
            table_index = db["column_names_original"][index+1][0]
            table_name = db["table_names_original"][table_index]
            if key in dict.keys():
                dict[key].append((values[index], table_name))
            else:
                dict[key] = [(values[index], table_name)]
    if type == "table":
        for index in range(len(keys)):
            key = keys[index]
            if key in dict.keys():
                print("Repeated table name!")
            else:
                dict[key] = values[index]
    return dict


def remove_ident_parens(idents):
    if isinstance(idents, list):
        return ", ".join(ele for ele in idents).replace("(", "").replace(")", "")
    else:
        return idents.replace("(", "").replace(")", "")

def schema(task):
    prolog = {}
    or2nns = [{}]
    nn2ors = [{}]
    for database in tables:
        db_id = database["db_id"]
        column_names_nn = [pair[1] for pair in database["column_names"]][1:]
        column_names_original = [pair[1] for pair in database["column_names_original"]][1:]
        column_names_prolog = ["'" + name + "'" for name in column_names_nn]
        c_or2nn = get_name_mapping("column", column_names_original, column_names_nn, database)
        c_nn2or = get_name_mapping("column", column_names_nn, column_names_original, database)

        table_names_nn = database["table_names"]
        table_names_original = database["table_names_original"]
        table_names_prolog = ["'" + name + "'" for name in table_names_nn]
        t_or2nn = get_name_mapping("table", table_names_original, table_names_nn)
        t_nn2or = get_name_mapping("table", table_names_nn, table_names_original)

        store_name_mappings(or2nns, db_id, c = c_or2nn, t = t_or2nn)
        store_name_mappings(nn2ors, db_id, c=c_nn2or, t=t_nn2or)

        table_column_match_nn = []
        for index in range(len(table_names_nn)):
            table_column_match_nn.append([])
            for index_c in range(len(database["column_names"])-1):
                column = database["column_names"][index_c+1]
                if column[0] == index:
                    table_column_match_nn[index].append("'" + column[1] + "'")
        if task == 2:
            foreign_key_nn = []
            db_tab_f = []
            tabs_f = {}
            for pair in database["foreign_keys"]:
                c1 = database["column_names"][pair[0]][1]
                t1_index = database["column_names"][pair[0]][0]
                t1 = database["table_names"][t1_index]
                c2 = database["column_names"][pair[1]][1]
                t2_index = database["column_names"][pair[1]][0]
                t2 = database["table_names"][t2_index]
                foreign_key_nn.append([[t1, c1], [t2, c2]])
                foreign_key_nn.append([[t2, c2], [t1, c1]])
                if not t1 in db_tab_f:
                    db_tab_f.append(t1)
                if not t2 in db_tab_f:
                    db_tab_f.append(t2)
                if t1 in tabs_f.keys():
                    if not t2 in tabs_f[t1]:
                        tabs_f[t1].append(t2)
                else:
                    tabs_f[t1] = [t2]
                if t2 in tabs_f.keys():
                    if not t1 in tabs_f[t2]:
                        tabs_f[t2].append(t1)
                else:
                    tabs_f[t2] = [t1]

        prolog_program = ""
        prolog_program = prolog_program + "database_tables('" + remove_ident_parens(db_id) + "', [" + remove_ident_parens(table_names_prolog) + "]).\n"
        if task == 2:
            prolog_program = prolog_program + "database_tables_foreign('" + remove_ident_parens(db_id) + "', [" + remove_ident_parens(db_tab_f) + "]).\n"
        prolog_program = prolog_program + "database_columns('" + remove_ident_parens(db_id) + "', [" + remove_ident_parens(column_names_prolog) + "]).\n"
        for index in range(len(table_names_nn)):
            prolog_program = prolog_program + "table('" + remove_ident_parens(db_id) + "', " + remove_ident_parens(table_names_prolog[
                index]) + ", [" + remove_ident_parens(table_column_match_nn[index]) + "]).\n"
        if task == 2:
            for key, value in tabs_f.items():
                prolog_program = prolog_program + "table_foreign('" + remove_ident_parens(db_id) + "', " + "'"+ remove_ident_parens(key) +"'" + ", [" + remove_ident_parens(value) + "]).\n"
            for pair in foreign_key_nn:
                prolog_program = prolog_program + "foreign_key('" + remove_ident_parens(pair[0][0]) + "', '" + remove_ident_parens(pair[1][0]) + "', '" + remove_ident_parens(pair[0][1]) + "', '" + remove_ident_parens(pair[1][1]) + "').\n"
        if db_id not in prolog.keys():
            prolog[db_id] = prolog_program
        else:
            print("This db is already in schema list!")


    if task == 2:
        with open("src/task2/schema_task2.json", "w") as of_schema:
            json.dump(prolog, of_schema, indent=2)
    else:
        with open("src/task1/schema_task1_1.json", "w") as of_schema:
            json.dump(prolog, of_schema, indent=2)
    print("task"+str(task)+" schema done")
    # store the mapping between orginal and nn names
    # orginal names are used in final sql queries
    # nn names used to help language model understand the schema
    with open("src/or2nn.json", "w") as of_name:
        json.dump(or2nns[0], of_name, indent=2)
    with open("src/nn2or.json", "w") as of_name:
        json.dump(nn2ors[0], of_name, indent=2)



def mapor2nn(or2nn, db, table, type, column=None):
    res = ""
    if type == "columns":
        col_tab = []
        cols = or2nn[db][type].keys()
        for c in cols:
            if c.lower() == column.lower():
                for item in or2nn[db][type][c]:
                    col_tab.append(item)
        if col_tab == []:
            print("no or col match between gt query and mapping file!")
        tabs = [pair[1] for pair in col_tab]
        cols = [pair[0] for pair in col_tab]
        for index in range(len(tabs)):
            t = tabs[index]
            if t == table or t.lower() == table.lower():
                res = cols[index]
        if res == "":
            print(db)
            print(column)
            print(table)
            print(col_tab)
            print("no or2nn col match, attention!")
    else:
        try:
            res = or2nn[db][type][table]
        except:
            tabs = or2nn[db][type].keys()
            for t in tabs:
                if t.lower() == table.lower():
                    res = or2nn[db][type][t]
        if res == "":
            # print(db)
            # print(table)
            print("no or2nn tab match, attention!")
    return res

def get_prompt_index(text, domain, target, type=None, state=None):
    prompt = text
    target_id = -100
    if type == "table":
        if state == 0:
            prompt += " SELECT FROM [table],"
        elif state == 1:
            prompt += " JOIN [table],"
        elif state == 2:
            prompt += " SELECT [table],"
        elif state == 3:
            prompt += " WHERE [table],"
        elif state == 4:
            prompt += " GROUP BY [table],"
        elif state == 5:
            prompt += " ORDER BY [table],"
        elif state == 6:
            prompt += " EXCEPT [table],"
    if type == "column":
        if state== 0:
            prompt += " SELECT [column],"
        elif state == 1:
            prompt += " WHERE [column],"
        elif state==2:
            prompt += " GROUP BY [column],"
        elif state==3:
            prompt += " ORDER BY [column],"
    for index in range(len(domain)):
        if target_id == -100:
            if target == domain[index] or target.lower() == domain[index].lower():
                target_id = index + 1
        prompt = prompt + " Answer " + str(index+1) + " for " + domain[index] + ", "
    prompt = prompt + "the answer should be Answer "
    if target_id == -100:
        print(domain)
        print(target)
        print("Cannot find target in domain!")
    return prompt, target_id


# remove the queries that are not covered in current task2 grammar
def out_grammar_remove(id_list, queries):
    remove_list = []
    for i in id_list:
        remove = 0
        query = queries[i]
        if not query.lower().find(" having ") == -1 and query.lower().find(" having count(*) ") == -1:
            remove_list.append(i)
            remove = 1
        if remove == 0:
            check_count = query[query.lower().find(" order by "):].lower()
            if not check_count.find("(") == -1:
                remove_list.append(i)
                remove = 1
            if remove == 0:
                for func in ["max(", "min(", "avg()", "sum("]:
                    if not check_count.find(func) == -1:
                        remove_list.append(i)
                        remove = 1
        if remove == 0:
            if not query.lower().find(" between ") == -1:
                remove_list.append(i)
                remove = 1
        if remove == 0:
            if not query.lower().find(" not like ") == -1:
                remove_list.append(i)
                remove = 1
        if remove == 0:
            if not query.lower().find("current_address_id != permanent_address_id") == -1:
                remove_list.append(i)
                remove = 1
    for i in remove_list:
        id_list.remove(i)
    return id_list



def get_task2(queries, keywords):
    query_features = []
    for query in queries:
        f = []
        for j in range(len(keywords)):
            key = keywords[j].lower()
            indices = [i for i in range(len(query)) if query.lower().startswith(key+" ", i)]
            if query.lower().endswith(" "+key):
                indices.append(1)
            f.append(len(indices))
        if "," in query:
            f.append(1)
        else:
            f.append(0)
        query_features.append(f)
    for index in range(len(queries)):
        query = queries[index]
        if "distinct(" in query:
            query_features[index][keywords.index("DISTINCT")] = 1

    op_index = []
    for index in range(len(queries)):
        query = queries[index]
        if " - " in query or "+" in query or "/" in query:
            op_index.append(index)


    nested_index = []
    columnlist_index = []
    join_index = []
    as_index = []
    and_index = []
    or_index = []
    for index in range(len(query_features)):
        feature = query_features[index]
        if feature[keywords.index("SELECT")] > 1:
            nested_index.append(index)
        if feature[-1] > 0:
            columnlist_index.append(index)
        if feature[keywords.index("JOIN")] > 0:
            join_index.append(index)
        if feature[keywords.index("AS")] > 0:
            as_index.append(index)
        if feature[keywords.index("AND")] > 0:
            and_index.append(index)
        if feature[keywords.index("OR")] > 0:
            or_index.append(index)
    nextlevel_set = set(nested_index + columnlist_index + join_index + as_index + op_index + and_index + or_index)

    part1 = []
    part2 = []
    part3 = []
    for index in range(len(query_features)):
        if index not in nextlevel_set:
            feature = query_features[index]
            if sum(feature) == 1:
                part1.append(index)
            else:
                part2.append(index)
    part2 = out_grammar_remove(part2, queries)
    for index in nextlevel_set:
        feature = query_features[index]
        if sum(feature)-(feature[keywords.index("EXCEPT")] + feature[keywords.index("SELECT")])==0 and index not in op_index:
            part3.append(index)
    return part1 + part2 + part3

def task2_process_write(data, ids, of):
    if "train" in of:
        random.shuffle(ids)
    else:
        ids = sorted(ids)
    samples = []
    for id in ids:
        samples.append(data[id])
    with open(of, "w") as json_file:
        json.dump(samples, json_file, indent=2)
    return samples


def train_type(train, test):
    type_train_t5 = []
    type_test_t5 = []

    all_samples = train + test
    for index in range(len(all_samples)):
        text = all_samples[index]["question"]
        query = all_samples[index]["query"]
        sample_out = {}
        sample_out["prompt"] = text + " Answer 1 for empty, Answer 2 for EXCEPT, the answer should be Answer "
        if " except " in query.lower():
            sample_out["target"] = str(2)
        else:
            sample_out["target"] = str(1)
        sample_out["query"] = query
        if index < len(train):
            type_train_t5.append(sample_out)
        else:
            type_test_t5.append(sample_out)

    with open("data/lms_task2/train_type_task2.json", "w") as file:
        json.dump(type_train_t5, file, indent=2)
    with open("data/lms_task2/test_type_task2.json", "w") as file:
        json.dump(type_test_t5, file, indent=2)



def get_select_col(query):
    from_index = query.lower().find(" from")
    selection = query[len("select") + 1:from_index].lower()
    column = None
    if not ("*" in selection):
        if "(" in selection:
            column = selection[selection.find("(") + 1:selection.find(")")]
            column = column.strip()
        else:
            column = selection.strip()
        column_list = column.split(" ")
        if len(column_list) > 1:
            column = column_list[1].strip()
    return column


def train_table(train, test, or2nn, db_col_tab):
    t_train_t5 = []
    t_test_t5 = []

    with open("src/task2/schema_task2.json", 'r') as f:
        schema = json.load(f)

    for index in range(len(train + test)):
        if index < len(train):
            text = train[index]["question"]
            db = train[index]["db_id"]
            query = train[index]["query"]
        else:
            test_index = index - len(train)
            text = test[test_index]["question"]
            db = test[test_index]["db_id"]
            query = test[test_index]["query"]
        query_list = query.lower().split(" ")
        query_list = [item for item in query_list if not item==""]
        if "except" in query_list:
            select1 = query_list[:query_list.index("except")]
            select2 = query_list[query_list.index("except")+1:]
            table_f = select1[select1.index("from")+1]
            table_e = select2[select2.index("from")+1].split(";")[0]
            schema_parts = schema[db].split("\n")
            e_schemas = [item for item in schema_parts if item.startswith("table_foreign(") and not item == ""]
            f_schema = [item for item in schema_parts if item.startswith("database_tables_foreign(") and not item == ""][0]
            f_domain = f_schema[f_schema.index("[") + 1:f_schema.index("]")].split(", ")
            f_domain = [item.replace("'", "") for item in f_domain]
            schema_e_index = [i for i in range(len(e_schemas)) if
                              e_schemas[i].startswith(
                                  "table_foreign('" + db + "', '" + mapor2nn(or2nn, db, table_f, "tables") + "'")]
            e_schema = e_schemas[schema_e_index[0]]
            e_domain = e_schema[e_schema.index("[") + 1:e_schema.index("]")].split(", ")
            e_domain = [item.replace("'", "") for item in e_domain]
            # add to table classifier training if schema info not sufficient to determine
            if not table_e == table_f:
                prompt, target = get_prompt_index(text, f_domain,
                                                  mapor2nn(or2nn, db, table_f, "tables"), "table", 0)
                if index < len(train):
                    t_train_t5.append(
                        {"prompt": prompt, "target": str(target), "table": mapor2nn(or2nn, db, table_f, "tables"),
                         "db": db, "query": query})
                else:
                    t_test_t5.append(
                        {"prompt": prompt, "target": str(target), "table": mapor2nn(or2nn, db, table_f, "tables"),
                         "db": db, "query": query})
                if not len(e_domain) == 1:
                    prompt, target = get_prompt_index(text, e_domain,
                                                      mapor2nn(or2nn, db, table_e, "tables"), "table", 6)
                    if index < len(train):
                        t_train_t5.append(
                            {"prompt": prompt, "target": str(target), "table": mapor2nn(or2nn, db, table_e, "tables"),
                             "db": db, "query": query})
                    else:
                        t_test_t5.append(
                            {"prompt": prompt, "target": str(target), "table": mapor2nn(or2nn, db, table_e, "tables"),
                             "db": db, "query": query})
        else:
            from_index = query_list.index("from")
            table_f = query_list[from_index + 1].split(";")[0]
            prompt, target = get_prompt_index(text, db_col_tab[db]["tables_nn"],
                                              mapor2nn(or2nn, db, table_f, "tables"), "table", 0)
            if index < len(train):
                t_train_t5.append(
                    {"prompt": prompt, "target": str(target), "table": mapor2nn(or2nn, db, table_f, "tables"),
                     "db": db, "query": query})
            else:
                t_test_t5.append(
                    {"prompt": prompt, "target": str(target), "table": mapor2nn(or2nn, db, table_f, "tables"),
                     "db": db, "query": query})



    with open("data/lms_task2/train_table_task2.json", "w") as file:
        json.dump(t_train_t5, file, indent=2)
    with open("data/lms_task2/test_table_task2.json", "w") as file:
        json.dump(t_test_t5, file, indent=2)


def train_column(train, test, or2nn, context_sensitive):
    c_train_t5 = []
    c_test_t5 = []

    with open("src/task2/schema_task2.json", 'r') as f:
        schema = json.load(f)

    for index in range(len(train + test)):
        if index < len(train):
            text = train[index]["question"]
            db = train[index]["db_id"]
            query = train[index]["query"]
        else:
            test_index = index - len(train)
            text = test[test_index]["question"]
            db = test[test_index]["db_id"]
            query = test[test_index]["query"]
        query_list = query.lower().split(" ")
        query_list = [item for item in query_list if not item == ""]

        if not "except" in query_list:
            from_index = query_list.index("from")
            table_f = query_list[from_index + 1].split(";")[0]

            c_exist = [None, None, None, None]
            c_exist[0] = get_select_col(query)
            if "where" in query_list:
                c_exist[1] = query_list[query_list.index("where")+1]
            if "group" in query_list:
                c_exist[2] = query_list[query_list.index("group")+2]
            if "order" in query_list and not ("*" in query[query.lower().find(" order by "):]):
                c_exist[3] = query_list[query_list.index("order")+2]


            schema_parts = schema[db].split("\n")
            table_schema = [item for item in schema_parts if item.startswith("table(") and not item == ""]
            if not context_sensitive:
                cf_domain = schema_parts[2][schema_parts[2].index("[") + 1:schema_parts[2].index("]")].split(", ")
                cf_domain = [item.replace("'", "") for item in cf_domain]
            else:
                schema_f_index = [i for i in range(len(table_schema)) if
                                table_schema[i].startswith("table('" + db + "', '" + mapor2nn(or2nn, db, table_f, "tables") + "'")]
                f_schema = table_schema[schema_f_index[0]]
                f_domain = f_schema[f_schema.index("[") + 1:f_schema.index("]")].split(", ")
                f_domain = [item.replace("'", "") for item in f_domain]


            for i in range(len(c_exist)):
                c = c_exist[i]
                if c is not None:
                    if ";" in c:
                        c = c.split(";")[0]
                    c = c.strip()
                    if not context_sensitive:
                        domain = cf_domain
                    else:
                        domain = f_domain
                    prompt, target = get_prompt_index(text, domain, mapor2nn(or2nn, db, table_f, "columns", c), "column", i)
                    if index < len(train):
                        c_train_t5.append(
                                    {"prompt": prompt, "target": str(target),
                                     "column": mapor2nn(or2nn, db, table_f, "columns", c),
                                     "db": db, "query": query})
                    else:
                        c_test_t5.append(
                                    {"prompt": prompt, "target": str(target),
                                     "column": mapor2nn(or2nn, db, table_f, "columns", c),
                                     "db": db, "query": query})
    if not context_sensitive:
        task = "column_cf"
    else:
        task = "column"
    with open("data/lms_task2/train_"+task+"_task2.json", "w") as file:
        json.dump(c_train_t5, file, indent=2)
    with open("data/lms_task2/test_"+task+"_task2.json", "w") as file:
        json.dump(c_test_t5, file, indent=2)



def train_select_switcher(train, test):
    all_samples = train + test
    train_ss = []
    test_ss = []
    for index in range(len(all_samples)):
        text = all_samples[index]["question"]
        query = all_samples[index]["query"]
        if not " except " in query.lower():
            from_index = query.lower().find(" from")
            selection = query[len("select")+1:from_index]
            selection_list = selection.split(" ")
            selection_list = [item for item in selection_list if not item==""]
            selection = "".join(selection_list).lower()
            sample_out = {}
            sample_out["prompt"] = text + " Answer 1 for *, Answer 2 for COUNT(*), Answer 3 for column, Answer 4 for DISTINCT column, Answer 5 for COUNT(column), Answer 6 for COUNT(DISTINCT column), Answer 7 for SUM(column), Answer 8 for AVG(column), Answer 9 for MIN(column), Answer 10 for MAX(column), the answer should be Answer "
            if "*" in selection:
                if "count(*)" in selection:
                    sample_out["target"] = str(2)
                else:
                    sample_out["target"] = str(1)
            elif "count(" in selection:
                if "distinct" in selection:
                    sample_out["target"] = str(6)
                else:
                    sample_out["target"] = str(5)
            elif "sum(" in selection:
                sample_out["target"] = str(7)
            elif "avg(" in selection:
                sample_out["target"] = str(8)
            elif "min(" in selection:
                sample_out["target"] = str(9)
            elif "max(" in selection:
                sample_out["target"] = str(10)
            else:
                if "distinct" in selection:
                    sample_out["target"] = str(4)
                else:
                    sample_out["target"] = str(3)
            sample_out["select"] = selection
            sample_out["query"] = query
            if index < len(train):
                train_ss.append(sample_out)
            else:
                test_ss.append(sample_out)

    with open("data/lms_task2/train_ss_task2.json", "w") as json_file:
        json.dump(train_ss, json_file, indent=2)
    with open("data/lms_task2/test_ss_task2.json", "w") as json_file:
        json.dump(test_ss, json_file, indent=2)


def train_clause_switcher(train, test, type):
    all_samples = train + test
    train_clause = []
    test_clause = []
    for index in range(len(all_samples)):
        text = all_samples[index]["question"]
        query = all_samples[index]["query"]
        if not " except " in query.lower():
            sample_out = {}
            sample_out["query"] = query
            if type == " where ":
                sample_out["prompt"] = text + " Answer 1 for empty, Answer 2 for WHERE, the answer should be Answer "
            elif type == " group by ":
                sample_out["prompt"] = text + " Answer 1 for empty, Answer 2 for GROUP BY, the answer should be Answer "
            elif type == " having ":
                sample_out["prompt"] = text + " Answer 1 for empty, Answer 2 for HAVING, the answer should be Answer "
            elif type == " order by ":
                sample_out["prompt"] = text + " Answer 1 for empty, Answer 2 for ORDER BY, the answer should be Answer "
            elif type == " desc":
                sample_out["prompt"] = text + " Answer 1 for ASC, Answer 2 for DESC, the answer should be Answer "
            elif type == " limit ":
                sample_out["prompt"] = text + " Answer 1 for empty, Answer 2 for LIMIT, the answer should be Answer "
            else:
                print("Clause type not defined!")
            if ((type == " desc" or type == " limit ") and not query.lower().find(" order by ") == -1) or (type == " having " and not query.lower().find(" group by ") == -1):
                if not query.lower().find(type) == -1:
                    sample_out["target"] = str(2)
                else:
                    sample_out["target"] = str(1)
                if index < len(train):
                    train_clause.append(sample_out)
                else:
                    test_clause.append(sample_out)
            elif type == " where " or type == " group by " or type == " order by ":
                if query.lower().find(type) == -1:
                    sample_out["target"] = str(1)
                else:
                    sample_out["target"] = str(2)
                if index < len(train):
                    train_clause.append(sample_out)
                else:
                    test_clause.append(sample_out)

    with open("data/lms_task2/train_"+type.strip()+"_task2.json", "w") as json_file:
        json.dump(train_clause, json_file, indent=2)
    with open("data/lms_task2/test_"+type.strip()+"_task2.json", "w") as json_file:
        json.dump(test_clause, json_file, indent=2)


def train_op_switcher(train, test):
    all_samples = train + test
    where_op = ["=", "!=", ">", "<", ">=", "<=", "like"]
    have_op = ["=", ">", "<", ">=", "<="]
    keys = ["WHERE", "HAVING"]
    train_op = []
    test_op = []
    for index in range(len(all_samples)):
        text = all_samples[index]["question"]
        query = all_samples[index]["query"]
        query_list = query.lower().split(" ")
        query_list = [item for item in query_list if not item == ""]
        if not "except" in query_list:
            for key in keys:
                if key.lower() in query_list:
                    sample_out = {}
                    if key == "WHERE":
                        domain = where_op
                    else:
                        domain = have_op
                    sample_out["prompt"] = text
                    sample_out["prompt"] += " "+key+" column [operator],"
                    for i in range(len(domain)):
                        sample_out["prompt"] += " Answer " + str(i+1) + " for " + domain[i] + ", "
                    sample_out["prompt"] += "the answer should be Answer "
                    target = query_list[query_list.index(key.lower())+2].strip()
                    sample_out["target"] = str(domain.index(target)+1)
                    sample_out["query"] = query
                    if index < len(train):
                        train_op.append(sample_out)
                    else:
                        test_op.append(sample_out)

    with open("data/lms_task2/train_op_task2.json", "w") as json_file:
        json.dump(train_op, json_file, indent=2)
    with open("data/lms_task2/test_op_task2.json", "w") as json_file:
        json.dump(test_op, json_file, indent=2)



def col_ident(selection, col_idents):
    #print(selection)
    if selection[0] == "*":
        return selection[0]
    else:
        if not selection.find("DISTINCT")==-1:
            col = selection.split()[1]
        else:
            col = selection
        #print(col)
        res = ""
        for c in col_idents:
            if col.lower() == c.lower():
                res = c
        return res

# get the listed version of gt SQL query and change original names to nn names
def get_gt_list(query, tab_col, tab_idents, db, or2nn):
    query = query.split(";")[0]
    tokens = query.split()
    gt = []
    if not tokens[0].lower() == "select":
        print("Alert! Query not start with [select]!")
    else:
        gt.append("'SELECT'")
    from_index = [index for index, value in enumerate(tokens) if value.lower() == "from"]
    table = tokens[from_index[0]+1]
    gt_tab = ""
    for t in tab_idents:
        if table.lower() == t.lower():
            gt_tab = t
    if gt_tab == "":
        print("Alert! Table not found!")
    gt_tab_nn = or2nn[db]["tables"][gt_tab]
    selection = " ".join(tokens[1:from_index[0]])
    gt_sel = []
    if "(" in selection and ")" in selection:
        has_func = True
    else:
        has_func = False
    if not has_func:
        if "DISTINCT" in selection:
            gt_sel.append("'DISTINCT'")
        gt_col_sel = col_ident(selection, tab_col[gt_tab])
        if gt_col_sel == "":
            print("Alert! Column not found!")
        if not gt_col_sel == "*":
            col_nn = ""
            for pair in or2nn[db]["columns"][gt_col_sel]:
                if pair[1] == gt_tab:
                    col_nn = pair[0]
            gt_sel.append("'"+col_nn+"'")
        else:
            gt_sel.append("'*'")
    else:
        if "*" in selection:
            gt_sel.append("'COUNT(*)'")
        else:
            split_func = re.split(r'[()]', selection)
            split_func = [part.strip() for part in split_func if part]
            if not len(split_func) == 2:
                print("Alert! Wrong parse for aggregated column!")
            if split_func[0].lower() == "count":
                gt_sel.append("'COUNT('")
            elif split_func[0].lower() == "sum":
                gt_sel.append("'SUM('")
            elif split_func[0].lower() == "avg":
                gt_sel.append("'AVG('")
            elif split_func[0].lower() == "min":
                gt_sel.append("'MIN('")
            elif split_func[0].lower() == "max":
                gt_sel.append("'MAX('")
            elif split_func[0].lower() == "distinct":
                gt_sel.append("'DISTINCT'")
            else:
                print(query)
                print("Alert! Unknown aggregate function!")
            if "DISTINCT" in split_func[1]:
                gt_sel.append("'DISTINCT'")
                col = col_ident(split_func[1].split()[1], tab_col[gt_tab])
            else:
                col = col_ident(split_func[1], tab_col[gt_tab])
            if col == "":
                print("Alert! Column not found!")
            col_nn = ""
            for pair in or2nn[db]["columns"][col]:
                if pair[1] == gt_tab:
                    col_nn = pair[0]
            if split_func[0].lower() == "distinct":
                gt_sel.append("'" + col_nn + "'")
            else:
                gt_sel.extend(["'"+col_nn+"'", "')'"])
    gt.extend(gt_sel)
    gt.extend(["'FROM'", "'" + gt_tab_nn + "'"])
    if not query.find("GROUP BY") == -1:
        gt.append("'GROUP BY'")
        groupby = query[query.find("GROUP BY"):]
        gt_col_group = groupby.split()[2]
        col = col_ident(gt_col_group, tab_col[gt_tab])
        if col == "":
            print("Alert! Column not found!")
        col_nn = ""
        for pair in or2nn[db]["columns"][col]:
            if pair[1] == gt_tab:
                col_nn = pair[0]
        gt.append("'"+col_nn+"'")
    if not query.find("ORDER BY") == -1:
        gt.append("'ORDER BY'")
        orderby = query[query.find("ORDER BY"):]
        gt_col_order = orderby.split()[2]
        if "*" in gt_col_order:
            gt.append("'COUNT(*)'")
        else:
            col = col_ident(gt_col_order, tab_col[gt_tab])
            if col == "":
                print("Alert! Column not found!")
            col_nn = ""
            for pair in or2nn[db]["columns"][col]:
                if pair[1] == gt_tab:
                    col_nn = pair[0]
            gt.append("'" + col_nn + "'")
        if len(orderby.split())==4:
            gt.append("'" + orderby.split()[3] + "'")
    return gt



def get_schema_mapping():
    # or for original name, nn for nn name
    db_col_tab = {}
    for t in tables:
        db_col_tab[t["db_id"]] = {}
        db_col_tab[t["db_id"]]["columns_or"] = [pair[1] for pair in t["column_names_original"]]
        db_col_tab[t["db_id"]]["columns_nn"] = [pair[1] for pair in t["column_names"]]
        db_col_tab[t["db_id"]]["tables_or"] = t["table_names_original"]
        db_col_tab[t["db_id"]]["tables_nn"] = t["table_names"]
        db_col_tab[t["db_id"]]["tab_col_or"] = {}
        for index in range(len(t["table_names_original"])):
            tab = t["table_names_original"][index]
            db_col_tab[t["db_id"]]["tab_col_or"][tab] = []
            for column in t["column_names_original"]:
                if column[0] == index:
                    db_col_tab[t["db_id"]]["tab_col_or"][tab].append(column[1])
        db_col_tab[t["db_id"]]["tab_col_nn"] = {}
        for index in range(len(t["table_names"])):
            tab = t["table_names"][index]
            db_col_tab[t["db_id"]]["tab_col_nn"][tab] = []
            for column in t["column_names"]:
                if column[0] == index:
                    db_col_tab[t["db_id"]]["tab_col_nn"][tab].append(column[1])
    return db_col_tab

def get_task1(full_set, not_contain):
    index = []
    id = 0
    for sample in full_set:
        flag = True
        for ele in not_contain:
            if (ele in sample) or (ele.lower() in sample):
                flag = False
        # single column
        if "," in sample:
            flag = False
        # remove >1 column, e.g. SELECT avg(active_to_date - active_from_date) FROM customer_contact_channels
        if "(" in sample:
            column = sample[sample.find("(") + 1:sample.find(")")]
            column = column.strip()
            if len(column.split()) > 1:
                flag = False
        if flag:
            index.append(id)
        id = id+1
    return index

def task1_process_write(data, ids, or2nn, db_col_tab, of):
    samples = []
    for id in ids:
        sample = {}
        # remove "'s" in the question to avoid error in prolog
        sample["question"] = "'" + data[id]["question"].replace("'s", "") + "'"
        db = data[id]["db_id"]
        sample["db"] = "'" + db + "'"
        #sample["query_or"] = data[id]["query"]
        sample["query_gt"] = get_gt_list(data[id]["query"], db_col_tab[db]["tab_col_or"],
                                          db_col_tab[db]["tables_or"], db, or2nn)
        samples.append(sample)
    with open(of, "w") as json_file:
        json.dump(samples, json_file, indent=2)




def prepare_t5():
    with open("data/train_task2.json", "r") as train_f:
        train_samples = json.load(train_f)
    with open("data/test_task2.json", "r") as test_f:
        test_samples = json.load(test_f)
    with open("src/task2/schema_task2.json", "r") as schema_f:
        schema = json.load(schema_f)
    all_samples = train_samples + test_samples
    train_t5 = []
    test_t5 = []
    for index in range(len(all_samples)):
        sample_in = all_samples[index]
        sample_out = {}
        db = sample_in["db_id"].replace("'","")
        schema_parts = schema[db].split("\n")
        table_domain = schema_parts[0][schema_parts[0].index("[")+1:schema_parts[0].index("]")].split(", ")
        table_domain = [token.replace("'", "") for token in list(table_domain)]
        column_domain = []
        for part in schema_parts[3:]:
            if part and part.startswith("table("):
                column_domain.append(part[part.index("[") + 1:part.index("]")].split(", "))
        for elem in column_domain:
            for i in range(len(elem)):
                elem[i] = elem[i].replace("'", "")
        prompt = sample_in["question"].replace("'","")+ " database is " + db + ". tables are "
        prompt += ", ".join(table_domain)
        for i in range(len(table_domain)):
            prompt += ". columns in "
            prompt += table_domain[i]
            prompt += " are "
            prompt += ", ".join(column_domain[i])
        prompt += "."
        sample_out["prompt"] = prompt
        sample_out["target"] = sample_in["query"]
        if index < len(train_samples):
            train_t5.append(sample_out)
        else:
            test_t5.append(sample_out)

    with open("data/train_t5_baseline.json", "w") as json_file:
        json.dump(train_t5, json_file, indent=2)
    with open("data/test_t5_baseline.json", "w") as json_file:
        json.dump(test_t5, json_file, indent=2)


def data_preparation(task):
    train_queries = [sample["query"] for sample in train]
    test_queries = [sample["query"] for sample in test]
    db_col_tab = get_schema_mapping()
    with open("src/or2nn.json", "r") as file:
        or2nn = json.load(file)

    if task == 1:
        task1_not_contain = ["WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "JOIN", "INTERSECT", "EXCEPT", "UNION", "NOT IN", "OR", "AND", "EXISTS", "LIKE", "BETWEEN", "AS", "DISTINCT", "DESC", "ASC"]
        task1_train_id = get_task1(train_queries, task1_not_contain)
        task1_test_id = get_task1(test_queries, task1_not_contain)
        task1_process_write(train, task1_train_id, or2nn, db_col_tab, "data/train_task1.json")
        task1_process_write(test, task1_test_id, or2nn, db_col_tab, "data/test_task1.json")
        print("task1 data prepared")
    else:
        sql_keywords = ["SELECT", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "JOIN", "INTERSECT", "EXCEPT",
                        "UNION", "NOT IN", "OR", "AND", "EXISTS", "LIKE", "BETWEEN", "AS", "DISTINCT", "DESC", "ASC"]
        task2_train_id = get_task2(train_queries, sql_keywords)
        task2_test_id = get_task2(test_queries, sql_keywords)
        task2_train = task2_process_write(train, task2_train_id, "data/train_task2.json")
        task2_test = task2_process_write(test, task2_test_id, "data/test_task2.json")

        # prepare supervised training for grammar branches
        # train_type: single selection / two selections connected by "EXCEPT"
        train_type(task2_train, task2_test)
        train_table(task2_train, task2_test, or2nn, db_col_tab)
        train_column(task2_train, task2_test, or2nn, True)
        train_select_switcher(task2_train, task2_test)
        train_clause_switcher(task2_train, task2_test, " where ")
        train_clause_switcher(task2_train, task2_test, " group by ")
        train_clause_switcher(task2_train, task2_test, " order by ")
        train_clause_switcher(task2_train, task2_test, " having ")
        train_clause_switcher(task2_train, task2_test, " desc")
        train_clause_switcher(task2_train, task2_test, " limit ")
        train_op_switcher(task2_train, task2_test)
        print("task2 data prepared")
        print("prepare vanilla T5-small data")
        prepare_t5()
        print("prepare T5-small+CFG data")
        train_column(task2_train, task2_test, or2nn, False)


def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task', type=int, choices=[1, 2], help='Task number (1 or 2)')
    args = arg_parser.parse_args()
    return args


def main():
    args = init()
    print("preparing schema for task"+str(args.task))
    schema(task=args.task)
    print("preparing data for task"+str(args.task))
    data_preparation(task=args.task)


if __name__ == "__main__":
    main()



