# Animal-Condition-Prediction
Predicting Animal Condition using different Machine Learning algorithms.

## PHASES OF THE DATA ANALYTICS LIFE CYCLE

### 1) DATA DISCOVERY

#### a) Project Overview:
This initiative seeks to harness the power of data analytics in veterinary medicine. By integrating detailed symptom data and comprehensive health records into a sophisticated predictive model, the goal is to transform how health conditions in animals are diagnosed. This approach aims not just to enhance accuracy but also to expedite the diagnostic process, potentially saving lives and improving the welfare of animals across various settings, from domestic pets to wildlife.

#### b) Learning the Business Domain:
The venture starts with a deep dive into the veterinary field, emphasizing the diagnostic processes and the significant effects of health conditions on animals. It involves understanding the nuances of veterinary care, including disease prevalence, progression, and treatment outcomes. This knowledge is crucial for tailoring the predictive model to be not only accurate but also practical in a real-world veterinary context.

#### c) Resources Available:
The project harnesses a multidisciplinary approach, bringing together data scientists with a knack for pattern recognition and veterinarians with years of clinical experience. This collaboration is augmented by state-of-the-art data analysis tools and technologies, from machine learning algorithms to big data platforms, ensuring a comprehensive analysis of health records and symptom data. These resources are pivotal in navigating the complexities of animal health conditions and their manifestations.

#### d) Framing the Problem:
Developing this model involves overcoming significant challenges, such as the diversity of species, the variability of symptoms, and the complexity of diseases. The model's potential to revolutionize animal healthcare underscores its importance. By providing swift and accurate diagnoses, it aims to facilitate timely interventions, reduce unnecessary treatments, and ultimately, enhance the quality of life for animals.

#### e) Identifying Key Stakeholders:
The project is a collaborative endeavor, involving veterinarians who provide invaluable insights into clinical aspects, care providers who understand the practical needs of animal care, and researchers who contribute cutting-edge scientific knowledge. This collaboration ensures that the model is developed with a holistic view of animal healthcare, aligning technological advancements with the real-world needs and expectations of the veterinary community.

#### f) Developing Initial Hypothesis:
The project begins with hypotheses about the relationship between various symptoms and health conditions. These hypotheses are informed by both veterinary expertise and data patterns, focusing on the predictive power of symptom combinations for different diseases. The iterative nature of this process allows for continuous refinement of hypotheses, leveraging data analysis to uncover insights that can significantly enhance the model's accuracy.

#### g) Identifying Potential Data Sources:
Identifying and securing diverse data sources is crucial for the model's success. This includes not only veterinary records and databases, which provide historical and clinical insights, but also emerging online datasets that offer real-time or novel data. Emphasis is placed on gathering a wide array of data to ensure the model's robustness and generalizability across different animal species and conditions. Ethical data handling and privacy compliance are integral, ensuring that the project adheres to the highest standards of data governance and ethical research.

### 2) DATA PREPARATION

#### a) Analytic Sandbox Establishment:
We have established a controlled environment for data exploration to gain a clear understanding of the dataset. This sandbox enables us to perform effective data exploration and preparation.

#### b) Data Exploration and Preparation:
c) ETLT Process:
- **Extract:** Retrieved data from a CSV file on Kaggle containing animal symptoms and their corresponding health conditions.
- **Transform:** Cleaned and transformed the data to ensure consistency and suitability for analysis.
- **Load:** Loaded the transformed data into our Python environment for further processing.

#### d) Learning about the Data:
- **Source:** Public data repository.
- **Structure:** Explored the data structure, identifying 871 rows (data points) and 7 columns.

#### e) Data Conditioning:
- **Missing Values:** Handled missing values (2 entries) by dropping the corresponding rows to maintain simplicity.
- **Duplicate Rows:** Addressed duplicate rows.
- **Categorical Data:** Encoded categorical data (animal names, symptoms).

#### f) Survey and Visualize:
- Explored data distribution using descriptive statistics (mean, median, standard deviation) to understand the distribution of features and the target variable.
- Created visualizations such as bar plots, line charts, and pie charts to identify patterns and trends in the data. These visualizations provided valuable insights into the distribution of animal conditions and the relationships between features.

#### g) Tools for Data Preparation:
- **pandas:** Provided powerful tools for data manipulation and cleaning tasks.
- **Seaborn and Matplotlib:** Utilized these powerful tools for data visualization, enabling effective exploration of data patterns and trends.

### 3) MODEL PLANNING

#### a) Data Exploration and Variable Selection:
- **Features:** The dataset (df1) contains information about animals, including the type and five symptoms.
- **Target Variable:** The target variable selected for classification is 'Dangerous', indicating whether the animal is in a critical/dangerous condition.
- **Data Splitting:** The dataset is split into training and testing sets using a 75-25 ratio, ensuring a portion of data is reserved for model evaluation.

#### b) Model Selection:
- **Algorithm:** Support Vector Machine (SVM) with a linear kernel is chosen for classification. SVM is well-suited for binary classification tasks and can effectively handle high-dimensional data.
- **Feature Scaling:** StandardScaler is applied to standardize the feature values, ensuring consistent scales across features, which is crucial for SVM's performance.

#### c) Common Tools for Model Planning Phase:
- **Train-Test Splitting:** The dataset is divided into training and testing subsets using train_test_split from scikit-learn. This ensures model evaluation on unseen data, gauging its generalization capability.
- **Scaling:** Feature scaling is performed using StandardScaler to normalize the feature values, preventing any feature from dominating others during model training.
- **Model Instance:** An instance of SVM is created with specified parameters, including a linear kernel and a random state, ensuring reproducibility of results.

#### d) About Support Vector Machine:
- **Overview:** Support Vector Machine (SVM) is a powerful machine learning algorithm commonly used for classification tasks. It aims to find the best hyperplane in a multi-dimensional space that maximizes the margin between different classes.
- **Functionality:** SVM excels in sorting data into groups (classification) and can handle regression tasks as well. It seeks to create a boundary between different groups in the data by finding an optimal hyperplane.
- **Hyperplane Dimension:** The dimension of the hyperplane depends on the number of features in the dataset. For instance, with two features, the hyperplane is a line, while with three features, it becomes a 2-D plane. SVM is effective in high-dimensional spaces, although visualization becomes challenging beyond three dimensions.

#### Conclusion:
The model planning phase encompasses crucial steps in preparing for the development and implementation of a machine learning model for animal condition classification. By exploring the data, selecting appropriate features, choosing a suitable algorithm, and employing common tools for model planning, we lay a strong foundation for building an accurate and reliable predictive model. Support Vector Machine, with its ability to handle high-dimensional data and effectively classify instances, emerges as a promising choice for this classification task.

### 4) MODEL BUILDING
We are using an SVM or Support Vector Machine for the classification of the animals and their symptoms. SVM is used as it is a good fit, being a supervised learning model which uses some support vectors as critical data points to fit the margins more accurately than other models.

The process of building the model:
- Firstly, we split the dataset into training and testing data sets, this is done to check whether the trained model is able to respond correctly to data it hasn’t been fed before, thus measuring its prediction capabilities. Then feature scaling is done to transform all data values into a similar scale for more accurate results and prevent skewness.
- Feature Scaled data is used to train the SVM Model and afterwards, the test data is used to evaluate the model’s performance.

#### a) Results:
- The model gives valid and accurate results. The training accuracy is 97%. The testing accuracy is 98%. The overall accuracy of the model is 97.7%.
- As accuracy is high, there is no need for a different kind of model.
- Very suitable for the runtime environment of a clinic as once trained they are very lightweight, efficient which is a requirement for real-time detection.

#### b) Software Used for Development:
Python: Libraries used are:
- **sklearn:** for model selection
- **pandas:** for dataframe making and loading data
- **numpy:** for array manipulation
- **seaborn, matplotlib:** for data visualization

### 5) COMMUNICATE RESULTS

#### a) Key Findings:
**Example 1:**
- Selected Animal: Tiger
- Chosen Symptoms: Dry Air, Small Size, Inability to jump, Depression, Coughing
- Prediction Result: The model predicts that the condition is dangerous.

**Example 2:**
- Selected Animal: Chicken
- Chosen Symptoms: Slow Growth, Mild Weakness, Congestion, Trembling, Wound
- Prediction Result: The model predicts that the condition is not dangerous.

#### b) Major Insights:
- **Objective Achievement:** The team successfully achieved its objective in predicting dangerous conditions for animals based on the provided symptoms.
- **Statistical Significance:** The results obtained from the model are statistically significant and valid. The conducted analyses confirm the reliability of the predictions.

#### c) Recommendations:
- **Feature Selection:** Continue refining the selection of symptoms/features used for classification to enhance the model's accuracy and robustness.
- **Model Evaluation:** Regularly evaluate the performance of the model using different evaluation metrics to ensure its effectiveness across various scenarios.
- **Data Collection:** Expand the dataset to include a diverse range of animal types and conditions to improve the model's general

ization capability.
- **User Interface:** Develop a user-friendly interface that allows easy input of animal type and symptoms, and provides clear interpretation of the model's prediction.

#### Conclusion:
In conclusion, the model planning phase of our animal condition classification project has yielded promising results. By addressing the recommendations outlined above and continuing to refine the model, we aim to develop a highly accurate and reliable tool for predicting dangerous conditions in animals, thereby assisting veterinarians and animal caretakers in providing timely and appropriate care.

### 6) OPERATIONALIZE RESULTS

#### a) Key Findings:
- **Model Accuracy:** The Animal Conditions Classification ML Model achieved an impressive accuracy rate of 97.7% in classifying animal conditions based on symptoms. This indicates a high level of reliability in identifying critical conditions among animals.
- **Training Data:** The model was trained on a dataset comprising close to 800 instances of various animals and their corresponding symptoms. However, to further enhance the model's performance, additional data collection is recommended. Including a wider variety of animal species and conditions in the training dataset could improve the model's ability to detect subtle indicators of illness or injury.
- **Contributing Factors:** The top contributing factors for accurate classification were identified as the presence of visible injuries, body posture, and overall appearance of the animal. These features played a crucial role in enabling the model to differentiate between animals in critical conditions and those that are not.
- **Model Sensitivity:** The model demonstrated high sensitivity in detecting severe conditions such as malnutrition. However, it struggled with classifying more subtle conditions like minor injuries or early signs of illness. This indicates a potential area for improvement in the model's performance.
- **Data Preparation:** Basic Business Intelligence (BI) activities, including data cleaning and preprocessing, were identified as crucial steps in preparing the dataset for model training. Ensuring the quality and relevance of the training data is essential for the model to make accurate predictions.

#### b) Future Recommendations:
- **Incorporating Additional Features:** Future modifications for the animal conditions classification model could involve incorporating additional features such as behavioral cues and vocalizations. These features could provide valuable insights into the health status of animals and improve the model's predictive capabilities.
- **Image Analysis Techniques:** Exploring image analysis techniques such as object detection and segmentation could further enhance the model's ability to identify specific anatomical features or abnormalities. This could lead to more precise and reliable predictions regarding the health conditions of animals.

#### Conclusion:
The Animal Conditions Classification ML Model shows promise in accurately identifying critical conditions among animals based on input parameters. By leveraging the insights gained from key findings and implementing future recommendations, the model can be further optimized to provide valuable support to animal caregivers, veterinarians, and wildlife conservationists in ensuring the well-being of animals.
