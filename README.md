# Meta-Learning-Sperm-classification
Infertility refers to the inability of a couple to achieve a pregnancy that results in the birth of a baby. One of the significant causes of infertility in couples is morphological disorders, or the appearance characteristics, of sperm. In the past, embryologists manually examined microscopic images of sperm to identify these disorders, a process that was time-consuming and prone to errors and subjectivity.<br>

Considering the challenges and difficulties in evaluating sperm morphology, this research aims to automate the identification and analysis process using a deep learning algorithm and image processing solutions. This approach addresses previous challenges in the field.<br>

A supervised approach, titled Relational Network, was presented in this research. It is trained on a smaller number of images using a few-shot learning approach. By learning to compare relationships with a few labeled samples, it attempts to generalize its learning to unseen samples during testing. Finally, by training on half of the images in the training set and testing on three distinct sperm parts, the proposed algorithm achieved an accuracy of 75.88% for the head part, 84.34% for the vacuole part, and 72.25% for the acrosome part of sperm.<be>

In this project, we employed a freely available SMA dataset named MHSMA. The MHSMA dataset contains 1,540 grayscale noisy sperm images with a size of 128×128 pixels.<br>

Sample images of the MHSMA dataset: <br>


![Screenshot 2024-11-19 201125](https://github.com/user-attachments/assets/6614471e-0974-4bb2-b1be-4d739278ba05)





