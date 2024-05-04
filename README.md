# Transfer Learning for ECG Classification
This is Team 1's project to perform replication work of the paper by Weimann and Conrad.

The project deliverables can be found in the [report](report) directory.

Citation:

> Weimann, K., Conrad, T.O.F. Transfer learning for ECG classification. Sci Rep 11, 5251 (2021). https://doi.org/10.1038/s41598-021-84374-8

## Running the code

Use Python 3.10 or greater. Install dependencies:

```bash
pip install -r requirements.txt
```

Then obtain the Icentia11K dataset and the PhysioNet/CinC Challenge 2017 dataset.

Then follow the instructions in the Pre-training [directory](pretraining) to pre-train a model.

Finally, follow the instructions in the Fine-tuning [directory](finetuning) to fine-tune a model using the pre-trained weights produced by the previous step.

---

Below this line is the original README.

Scripts for [Pretraining](pretraining) and [Finetuning](finetuning) residual networks on ECG data.

## Installation

Make sure that your virtual environment satisfies the following requirements before running any code:

* Python version: `>=3.10`
* Dependencies: `pip install -r requirements.txt`
