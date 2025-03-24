# Requirements
The project focuses on using deep learning techniques to analyze medical OCT images for diagnosing retinal diseases. Its requirements include classifying the images into general categories through supervised learning and identifying specific biomarkers of those categories using unsupervised clustering methods. Two datasets are used in the research: one containing general biomarkers, such as disease categories, and the other with more specific biomarkers.

# Proposed solutions
Several stages are necessary to achieve the desired results. Initially, the research requires a performant model to classify images from the first dataset into four categories: three diseases (CNV, DME, and Drusen) and healthy retinas. For this task, three pre-trained models (VGG16, ResNet50, and ViT/B-16) with different architectures are used. For each, transfer learning is performed, followed by fine-tuning on the specific dataset. After successfully classifying the images, specific features will be extracted from the ResNet50’s average pooling layer, placed just before the final fully connected layer. These features are then grouped using the K-means clustering algorithm to identify similarities between them. After finding their four initial general biomarkers, the features are grouped into more clusters to potentially discover new patterns within each disease, rather than just the disease label. For this task, a second dataset containing images with nine specific biomarkers is used.

# Results obtained
After training the three pre-trained models, varying results have been obtained. Initially, after transfer learning, the models perform moderately well, with an accuracy of around 80%. After fine-tuning, their performance improved significantly. The VGG16 model achieved an accuracy of 93% on the test dataset, while ResNet50 reached almost 95% on the dataset of 10,000 images. The ViT-B/16 model obtained only 91% accuracy. Then, features were extracted from ResNet50, and clusters were formed using the K-means algorithms.

![Screenshot 2025-03-24 154722](https://github.com/user-attachments/assets/8eb5c043-46cb-43c7-93f5-05a2dc0f742b)

# Tests and verifications
By increasing the number of clusters, new groups within the Drusen class were associated with specific biomarkers. The green category shows no fluid, hyperfluorescent spots, reticular drusen, or PR layer disruption. The blue category contains fluid and hyperfluorescent spots, with very few hard drusen. The pink category is primarily characterized by the presence of hard drusen, soft drusen, and reticular drusen.

![Screenshot 2025-03-24 154731](https://github.com/user-attachments/assets/21dd72fd-e7b3-44b1-97c3-84e89e7616f4)

# Personal contributions
- Python scripts to manipulate the datasets and generate plots
- Jupyter Notebooks for training the models, layer freezing and unfreezing, and calculating metrics, using PyTorch library (with GPU)
- Extracting features, applying K-means and analyzing the clustering results

# Documentation sources
Articles providing solutions for both classifying OCT images and discovering unseen biomarkers have been reviewed. I analyzed the differences between building a CNN model and tuning a pre-trained model. Additionally, advantages of residual networks were discussed, along with comparisons of different methods for feature extraction and clustering.

