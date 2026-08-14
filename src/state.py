"""
state.py
--------
Shared, module-level state that several functions read and write, exactly
like the "Shared state" cell in the notebook. Kept as a separate module so
every other file can `import state` and read/mutate the same values instead
of each having its own copy.
"""

# Running totals, updated inside output_generator() and read at the end of
# the main evaluation loop.
correct_answers = 0
# count of Stage 1 predictions that matched the ground-truth answer letter
correct_rationales = 0
# count of Stage 2 predictions that matched the ground-truth rationale letter

output = []
# holds the model's most recent decoded reply; output_generator() overwrites
# this every call, and rPrompt() reads it back

predicted_answer = ""
# holds Stage 1's predicted letter (e.g. "A"), set inside rPrompt()

# The currently active PIL image for the sample being processed. Set in
# main.py before each pair of output_generator() calls, and read inside
# output_generator().
image = None
