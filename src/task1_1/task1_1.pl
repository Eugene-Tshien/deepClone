token(X) --> [X].
pick_tab(NL, DB, T) --> [], {nn(tab_picker, [NL, DB], T), database_tables(DB, Tables), domain(T, Tables)}.
pick_col(NL, DB, T, State) --> {nn(col_picker, [NL, DB, T, State], C), table(DB, T, Columns), domain(C, Columns)}, token(C).
selection_switch(_, _, _, 0) --> ['*'].
selection_switch(_, _, _, 1) --> ['COUNT(*)'].
selection_switch(NL, DB, T, 2) --> pick_col(NL, DB, T, 0).
selection_switch(NL, DB, T, 3) --> ['DISTINCT'], pick_col(NL, DB, T, 0).
selection_switch(NL, DB, T, 4) --> ['COUNT('], pick_col(NL, DB, T, 0), [')'].
selection_switch(NL, DB, T, 5) --> ['COUNT('], ['DISTINCT'], pick_col(NL, DB, T, 0), [')'].
selection_switch(NL, DB, T, 6) --> ['SUM('], pick_col(NL, DB, T, 0), [')'].
selection_switch(NL, DB, T, 7) --> ['AVG('], pick_col(NL, DB, T, 0), [')'].
selection_switch(NL, DB, T, 8) --> ['MIN('], pick_col(NL, DB, T, 0), [')'].
selection_switch(NL, DB, T, 9) --> ['MAX('], pick_col(NL, DB, T, 0), [')'].
groupby_switch(_, _, _, 0) --> [].
groupby_switch(NL, DB, T, 1) --> ['GROUP BY'], pick_col(NL, DB, T, 1).
asc_switch(0) --> ['ASC'].
asc_switch(1) --> ['DESC'].
orderby_switch(_, _, _, 0) --> [].
orderby_switch(NL, DB, _, 1) --> ['ORDER BY'], ['COUNT(*)'], asc_switch(Y), {nn(asc_switcher, [NL, DB], Y), domain(Y, [0,1])}.
orderby_switch(NL, DB, T, 2) --> ['ORDER BY'], pick_col(NL, DB, T, 2), asc_switch(Y), {nn(asc_switcher, [NL, DB], Y), domain(Y, [0,1])}.
selection(NL, DB, T) --> selection_switch(NL, DB, T, Y), {nn(select_switcher, [NL, DB], Y), domain(Y, [0,1,2,3,4,5,6,7,8,9])}.
group_by(NL, DB, T) --> groupby_switch(NL, DB, T, Y), {nn(groupby_switcher, [NL, DB], Y), domain(Y, [0,1])}.
order_by(NL, DB, T) --> orderby_switch(NL, DB, T, Y), {nn(orderby_switcher, [NL, DB], Y), domain(Y, [0,1,2])}.
query(NL, DB) --> pick_tab(NL, DB, T), ['SELECT'], selection(NL, DB, T), ['FROM'], token(T), group_by(NL, DB, T), order_by(NL, DB, T).
