#let Course = [DLH Spring 2024]
#let Team = [Team 1]

#set page(
  paper: "us-letter",
  header: [
    /* Without this, the gap between the text and horizontal line will
       be too large */
    #set block(spacing: 0.5em)
    #Course #h(1fr) #Team
    #line(length: 100%, stroke: 0.5pt)
  ],
  header-ascent: 30%,
  numbering: "1",
  margin: 1in
)
#set par(
  justify: true
)
#set text(
  font: "New Computer Modern",
  size: 10pt
)
#set enum(
  indent: 1.0em
)
#set list(
  indent: 1.0em
)
#set table(
  stroke: 0.5pt
)

#show heading: set block(above: 1.4em, below: 1em)
#show link: underline
#show link: set text(blue)
#show footnote: set text(blue)

#align(center, text(18pt)[
  *Project Proposal*
])

#grid(
  columns: (1fr, 1fr, 1fr),
  align(center)[
    Ted Hsu \
    #link("mailto:thhsu4@illinois.edu")
  ],
  align(center)[
    Myles Iribarne \
    #link("mailto:mylesai2@illinois.edu")
  ],
  align(center)[
    Daniel Xu \
    #link("mailto:dhxu2@illinois.edu")
  ]
)

= Citation to the Original Paper

Weimann, K., Conrad, T.O.F. Transfer learning for ECG classification. Sci Rep 11, 5251 (2021). #footnote[https://www.nature.com/articles/s41598-021-84374-8]

= General Problem

In the original paper, Weimann and Conrad applied transfer learning to improve ECG classification. They first pre-train deep convolutional neural networks (CNN) on the _Icentia11K_ dataset #footnote[https://physionet.org/content/icentia11k-continuous-ecg/1.0/] <icentia11k>, and then fine-tune the pre-trained CNNs on the _PhysioNet/CinC 2017_ dataset #footnote[https://physionet.org/content/challenge-2017/1.0.0/] <physionet2017> to classify Atrial Fibrillation (AF).

= Specific Approach

The visualization of the transfer learning framework in the original paper is shown in @setup below:

#figure(
  image("setup.png"),
  caption: [
    High level framework by Weimann and Conrad.
  ],
) <setup>

The paper designs four different pre-training tasks:
- beat classification
- rhythm classification
- heart rate classification
- future prediction

All tasks are supervised learning with labels (beat and rhythm) from the original dataset or extracted from the ECG signal during the data preprocessing stage (heart rate and future ECG signal). During pre-training, a short (4 to 60 seconds) and continuous ECG signal is sampled from every patient. On average, 4096 samples are collected per patient.

The CNN architectures used by the original paper include ResNet-18v2, ResNet-34v2, and ResNet-50v2, and all of them are 1-D CNNs.

In the fine-tuning stage, the model is initialized with the weights acquired from the pre-training task, replaces its output layer with a fully connected layer whose output matches the classes of the PhysioNet data, and then is trained with PhysioNet data for AF. The paper also tries different sampling frequencies on PhysioNet data.

= Hypotheses to be Tested

The main claim of the paper is that pre-training 1-D CNN models with an extremely large dataset of relatively inexpensively labeled data (such as beat classification) can improve performance of classification based on a smaller set of labeled data with a different classification objective (e.g. AF).

== Objective \#1

We plan on verifying the main claim that transfer-learning improves the final model performance by reproducing results for one of the models/hyperparameter sets explored in the paper.

Specifically:
- Model: 1D ResNet-18v2
- Pre-training Objective: Beat Classification
- Frame size: 4096
- Sample Rate: 250Hz
- Fine-tuning objective: Atrial Fibrillation

To verify the main claim of the paper, we will compare the model and training performance with and without pre-training the model weights.

== Objective \#2

In order to extend this paper’s results, we aim to compare pre-trained and randomly initialized weights using different pre-processing and a different model (2D ResNet). This is motivated by a study on ECG Arrhythmia classification that demonstrates the effectiveness of CNNs trained on spectrograms - a frequency versus time representation of ECG signals. #footnote[J. Huang, B. Chen, B. Yao and W. He, "ECG Arrhythmia Classification Using STFT-Based Spectrogram and Convolutional Neural Network," in IEEE Access, vol. 7] By converting ECG data to spectrograms and utilizing a 2-D ResNet, we intend to illustrate the adaptability of the transfer-learning framework in the original paper across diverse model architectures.

= Ablations Planned

One of the main claims of this paper is that the size of the pre-training data contributes to the effectiveness of transfer learning when fine-tuning is applied. The paper does show that pre-training improves the final results. However, the paper does not explore how significant the effects of the pre-training data _size_ are on the final results.

== Objective \#3

Perform an ablation study comparing the final model performance when different subsets of the pre-training data are used during the pre-training step (for example, 10% versus 20% of the pre-training data).

= Description of Data Access

The training data is the "Icentia11k Single Lead Continuous Raw Electrocardiogram Dataset," which is freely available online as a 188 GB zip file. @icentia11k We do not use this dataset as-is but instead get the compressed version that is available on Academic Torrents. #footnote[https://academictorrents.com/details/af04abfe9a3c96b30e5dd029eb185e19a7055272]

The main validation dataset used for fine-tuning is the "AF Classification from a Short Single Lead ECG Recording: The PhysioNet/Computing in Cardiology Challenge 2017" which is a 1.4GB zip file and freely available online for download. @physionet2017

For all datasets, we save them to Google Drive and will access them from Google Colab via the "mount drive" feature.

= Discussion of the Feasibility of the Computation

The uncompressed _Icentia11k_ dataset is over 1 TiB, and we believe that it is too large and costly to perform pre-training using the entire dataset. We believe that we will be able to perform pre-training of the ResNet18 network on a subset of the data. The exact subset percentage is not yet clear, but will be at the patient level; the dataset contains 11k patient data.

We are thinking initially about 10% and 20%, in order to be able to complete the Objective \#3 ablation study of the effects of the size of pre-training data. We can increase the subset size based on the time taken to pre-train the model.

For the compute, we will use Google Colab Pro and seek to utilize GPUs. However, the amount of GPU capacity that we can attain during training is not yet clear as of this writing. Hence, we plan to start the pre-training with a subset of the data to get an empirical estimate of the feasibility of running pre-training on a larger subset. Then, we expect to spend GPU resources in accordance with our empirical estimate, adjusting as needed.

= Whether you will use the Existing Code or Not

Yes, we plan to use the existing code #footnote[https://github.com/kweimann/ecg-transfer-learning/tree/master], likely with certain modifications to:
- Update deprecated dependencies to be able to reproduce paper results (Objective \#1)
- Extend pre-processing and model architecture to support 2D ResNet (Objective \#2)
- Update code to run on the subsets of the data. for ablation study (Objective \#3)


