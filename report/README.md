# DLH Project Deliverables

## Draft Report

The draft report notebook is: [DL4H_Team_1_draft.ipynb](DL4H_Team_1_draft.ipynb).

The draft report PDF is: [DL4H_Team_1_draft.pdf](DL4H_Team_1_draft.pdf).

## Final Report

The final report notebook is: [DL4H_Team_1_final.ipynb](DL4H_Team_1_final.ipynb).

The final report PDF is: [DL4H_Team_1_final.pdf](DL4H_Team_1_final.pdf).

The video is: [Mediaspace](https://mediaspace.illinois.edu/media/t/1_sk6zbm84).

## PDF Generation

To generate locally, first install the following:

1. `nbconvert==7.16.3` via `jupyter` (higher version may work as well)
2. `pandoc`
3. A $\LaTeX$ installation providing `xelatex`

Then run the following command from this directory (`report/`):

```bash
# for the draft
$ jupyter nbconvert DL4H_Team_1_draft.ipynb --to pdf --template-file draft.tplx --TagRemovePreprocessor.remove_cell_tags="hidden"

# for the final
$ jupyter nbconvert DL4H_Team_1_final.ipynb --to pdf --template-file final.tplx --TagRemovePreprocessor.remove_cell_tags="hidden"
```

The output PDF will be located in this directory as well. It will have the same base filename as that of the source notebook.


### Hiding notebook cells from PDF

One can mark some cells in the notebook to be hidden in the output PDF. One way to do this is to set the cell tag in an editor like VSCode. Here, we add a tag "hidden" to the cell metadata to exclude the cell from the output PDF.

