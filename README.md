# Genre Classification of Lyrics

This is a project associated with the Natural Language Processing course as part of MSc. Computer Science (Data Science) at University of Twente, Netherlands during 2021-2022.

The project is to classify music lyrics based on genres. The file `genre_project_from_colab.ipynb` is the code submitted for the course and this includes basic text processing and classification with TF-IDF, Word2Vec & GloVe features with MNB, SVM, SGB, MLP algorithms.

The file `genre_LSTM.ipynb` is a follow up to this project which implments LSTM and Bidirectional LSTM techniques for the same purpose. The features used are pretrained GloVe vectors. During training, it was observed that the 'Indie' & 'Folk' genres were misclassified very often and hence they were not considered for further training.

To avoid dependency issues, a conda environment can be created and it could be done by running `./myenv.sh`

The images below show the training confusion matrix and also the accuracy graph while training and validation.

![Confusion Matrix](https://i.ibb.co/VqG9tgz/image.png)

![Accuracy](https://i.ibb.co/KLPZBJ8/image.png)

It can be observed that the validation accuracy stops improving after almost 20 epochs saturating around 55%.