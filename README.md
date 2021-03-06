## Background
The purpose of this project is to introduce the system architecture and construction process of the intelligent customer service system, and to realize the text matching model, the core of the intelligent customer service system, by using the deep learning technology -- BiLstm(GRU) based representation model of securities customer service data
Due to the limitations of human customer service in response time, service time and business knowledge, it is necessary to develop an intelligent customer service system to assist human customer service for users through intelligent means.
There are two types of deep text matching models:
- Representation-based Model -- a representation-based Model. The representational deep text matching model can extract sentence principal components and transform text sequences into vectors. Therefore, in the problem clustering module, we use the representational deep text matching model to preprocess the mining questions and FAQ questions, which is convenient for the subsequent incremental clustering module calculation.In the semantic recall module, we use the deep text matching model of representation to vectorize the questions in FAQ library and build the index, which is convenient for the problem recall module to increase the recall of users' Query.In addition, we use a representation model based on BI-LSTM to capture long dependencies within sentences.
- Lent IterAction-based Model -- an interactive Model. The expressive deep text matching model is easy to lose semantic focus and have semantic deviation when representing sentences, while the interactive deep text matching model has no such problem. It can grasp semantic focus well and model the importance of context reasonably.

## Technique
- Text preprocessing;
- BiLStm(SiaGRU) modeling;
- Lent text matching model training and validation;
- Sino Text matching model evaluation;

## Env
- Windows 10 system；
- Pycharm
- Python3.6
- pytorch 1.6.0
- genism
- pandas
