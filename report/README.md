# DLH Project Deliverables

## Draft Report

The draft report is: [DL4H_Team_1_draft.ipynb](DL4H_Team_1_draft.ipynb).

## Final Report

Work in Progress

## PDF Generation

To generate locally, need to install the following:

1. `nbconvert==7.16.3` via `jupyter` (higher version may work as well)
2. `pandoc`
3. A $\LaTeX$ installation providing `xelatex`

Then run the following command from this directory (`report/`):

```bash
# for the draft
$ jupyter nbconvert DL4H_Team_1_draft.ipynb --to pdf --template-file draft.tplx --TagRemovePreprocessor.remove_cell_tags="hidden"

# for the final
$ jupyter nbconvert DL4H_Team_1.ipynb --to pdf --template-file final.tplx --TagRemovePreprocessor.remove_cell_tags="hidden"
```

One can mark some cells in the notebook to be hidden in the output PDF. One way to do this is to set the cell tag in an editor like VSCode.

