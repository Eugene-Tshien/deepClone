token(X) --> [X].
pick_tab(NL, DB, T) --> [], {nn(tab_picker, [NL, DB], T), database_tables(DB, Tables), domain(T, Tables)}.
pick_col(NL, DB, T, C) --> {nn(col_picker, [NL, DB, T], C), table(DB, T, Columns), domain(C, Columns)}, token(C).
selection_switch(_, _, _, '*', 0) --> ['*'].
selection_switch(_, _, _, '*', 1) --> ['COUNT(*)'].
selection_switch(NL, DB, T, C, 2) --> pick_col(NL, DB, T, C).
selection_switch(NL, DB, T, C, 3) --> ['COUNT('], pick_col(NL, DB, T, C), [')'].
selection_switch(NL, DB, T, C, 4) --> ['SUM('], pick_col(NL, DB, T, C), [')'].
selection_switch(NL, DB, T, C, 5) --> ['AVG('], pick_col(NL, DB, T, C), [')'].
selection_switch(NL, DB, T, C, 6) --> ['MIN('], pick_col(NL, DB, T, C), [')'].
selection_switch(NL, DB, T, C, 7) --> ['MAX('], pick_col(NL, DB, T, C), [')'].
selection(NL, DB, T, C) --> selection_switch(NL, DB, T, C, Y), {nn(select_switcher, [NL, DB], Y), domain(Y, [0,1,2,3,4,5,6,7])}.
query(NL, DB) --> pick_tab(NL, DB, T), ['SELECT'], selection(NL, DB, T, _), ['FROM'], token(T).