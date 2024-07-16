import typing
from collections.abc import Sequence


class Term(object):
    def __init__(self, functor: str, *arguments: "Term"):
        self._functor = functor
        self._arguments = arguments
        self.arity = len(arguments)

        self._repr = str(self._functor) + (
            "({})".format(",".join([str(x) for x in self.arguments]))
            if self.arguments is not None and len(self.arguments) > 0
            else ""
        )

        self._hash = hash(self._repr)

    @property
    def functor(self) -> str:
        return self._functor

    @property
    def arguments(self) -> typing.Tuple["Term"]:
        return self._arguments

    def get_functor_and_arity(self) -> (str, int):
        return self._functor, len(self._arguments)

    def __repr__(self):
        return self._repr

    def __eq__(self, other: "Term"):
        return (
            hash(self) == hash(other)
            and isinstance(other, Term)
            and self._repr == other._repr
        )

    # compare strings?
    def __lt__(self, other):
        return self._repr < other._repr

    def __gt__(self, other):
        return self._repr > other._repr

    def __hash__(self):
        return self._hash


    def with_extra_arguments(
        self, *arguments: typing.Union["Term", str, int]
    ) -> "Term":
        return Term(self._functor, *(self.arguments + arguments))

    def without_difference_list(
        self, long_list_idx: int = -2, difference_list_idx: int = -1
    ) -> "Term":

        if len(self._arguments) < 2:
            return self

        long_list: List["Term"] = self._arguments[long_list_idx]
        diff_list = self._arguments[difference_list_idx]
        try:
            len(long_list)
        except:
            long_list = List(*[long_list])
        try:
            len(diff_list)
        except:
            diff_list = List(*[diff_list])

        new_list = List(
            *long_list.arguments[
                0 : len(long_list) - len(diff_list)
            ]
        )

        new_arguments = list(self._arguments[0:long_list_idx])
        new_arguments.append(new_list)
        if difference_list_idx != -1:
            new_arguments.extend(self._arguments[difference_list_idx:-1])

        return Term(self._functor, *new_arguments)

    def get_generation_output(self) -> "Term":
        return self._arguments[0]

    def can_mask_generation_output(self):
        return len(self.arguments) > 1

    def mask_generation_output(self, mask_index=0, mask_term=None) -> "Term":
        """ Removes the first argument in favor of an underscore, to 'mask' the desired output """
        if not self.can_mask_generation_output():
            # No generation output
            return self
        if not mask_term:
            return self.change_generation_output(wildcard_term, mask_index)
        else:
            return self.change_generation_output(mask_term, mask_index)

    def change_generation_output(self, new_argument: "Term", mask_index=0) -> "Term":
        new_arguments = list(self._arguments)
        new_arguments[mask_index] = new_argument
        return Term(self.functor, *new_arguments)

    def covers(self, el: "Term"):
        return self.functor == el.functor and all(
            self_arg == wildcard_term or self_arg == other_arg
            for (self_arg, other_arg) in zip(self.arguments, el.arguments)
        )

    def contains_mask(self):
        return any(arg == wildcard_term for arg in self.arguments)

    def to_dsl_input(self):
        """ Adds the right arguments to be used as dsl input """
        return self.with_extra_arguments(List())

    def is_list(self):
        return False


wildcard_term = Term("_")

class List(Term, Sequence):
    def __init__(self, *arguments: typing.Union[Term, str, int]):
        super(List, self).__init__(".", *arguments)

    def __str__(self):
        return "[{}]".format(",".join((str(x) for x in self._arguments)))

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.arguments)

    def __getitem__(self, i):
        return self.arguments.__getitem__(i)

    def is_list(self):
        return True


