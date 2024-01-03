import streamlit as st
import pandas as pd
from model import NaiveBayes, DecisionTree, RandomForest, EvaluationMetrics

from sklearn.model_selection import train_test_split


import streamlit as st
import pandas as pd

import time


import numpy as np
def main():
    st.title("Machine Learning Model Evaluation")

    uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        # Check if 'Label' column exists
        if 'Label' in df.columns:
            st.write("Dataset Preview:")
            st.write(df)

            # Define 'Label' column as the target class
            class_col = 'Label'
            y = df[class_col]
            X = df.drop(columns=[class_col])

            st.write("Info Dataset:")
            st.write(f"Jumlah Baris: {len(df)}")
            st.write(f"Jumlah Kolom: {len(df.columns)}")

            # Input split data
            split_ratio = st.sidebar.slider("Training-Testing Split Ratio", 0.1, 0.9, 0.3)
            st.write(f"Pengguna memilih split ratio: {split_ratio}")
 
            # Check if split ratio is chosen
            if split_ratio:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
                st.write(f"Jumlah data training: {len(X_train)}")
                st.write(f"Jumlah data testing: {len(X_test)}")

            if st.button("Tes Naive Bayes"):
                start_time = time.time()
                naive_bayes = NaiveBayes()
                eval_metrics = EvaluationMetrics()
                accuracy, confusion_matrix, execution_time, precision, recall, f1_score, auc = eval_metrics.evaluate_model(naive_bayes, X_train, X_test, y_train, y_test)
                end_time = time.time()
                st.write(f"Waktu Eksekusi Naive Bayes: {end_time - start_time:.2f} detik")
                st.write(f"Akurasi Naive Bayes: {accuracy}")
                st.write(f"Presisi Naive Bayes: {precision}")
                st.write(f"Recall Naive Bayes: {recall}")
                st.write(f"F1 Score Naive Bayes: {f1_score}")
                st.write(f"AUC Naive Bayes: {auc}")
                st.write("Confusion Matrix Naive Bayes")
                st.write(confusion_matrix)

            if st.button("Tes Decision Tree"):
                start_time = time.time()
                decision_tree = DecisionTree()
                eval_metrics = EvaluationMetrics()
                accuracy, confusion_matrix, execution_time, precision, recall, f1_score, auc = eval_metrics.evaluate_model(decision_tree, X_train, X_test, y_train, y_test)
                end_time = time.time()
                st.write(f"Waktu Eksekusi Decision Tree: {end_time - start_time:.2f} detik")
                st.write(f"Akurasi Decision Tree: {accuracy}")
                st.write(f"Presisi Decision Tree: {precision}")
                st.write(f"Recall Decision Tree: {recall}")
                st.write(f"F1 Score Decision Tree: {f1_score}")
                st.write(f"AUC Decision Tree: {auc}")
                st.write("Confusion Matrix Decision Tree")
                st.write(confusion_matrix)

            if st.button("Tes Random Forest"):
                start_time = time.time()
                random_forest = RandomForest()
                eval_metrics = EvaluationMetrics()
                accuracy, confusion_matrix, execution_time, precision, recall, f1_score, auc = eval_metrics.evaluate_model(random_forest, X_train, X_test, y_train, y_test)
                end_time = time.time()
                st.write(f"Waktu Eksekusi Random Forest: {end_time - start_time:.2f} detik")
                st.write(f"Akurasi Random Forest: {accuracy}")
                st.write(f"Presisi Random Forest: {precision}")
                st.write(f"Recall Random Forest: {recall}")
                st.write(f"F1 Score Random Forest: {f1_score}")
                st.write(f"AUC Random Forest: {auc}")
                st.write("Confusion Matrix Random Forest")
                st.write(confusion_matrix)
            
            if st.button("Perbandingan"):
                start_time = time.time()
                naive_bayes = NaiveBayes()
                decision_tree = DecisionTree()
                random_forest = RandomForest()

                eval_metrics_nb = EvaluationMetrics()
                eval_metrics_dt = EvaluationMetrics()
                eval_metrics_rf = EvaluationMetrics()

                accuracy_nb, _, execution_time_nb, precision_nb, recall_nb, f1_score_nb, auc_nb = eval_metrics_nb.evaluate_model(naive_bayes, X_train, X_test, y_train, y_test)
                accuracy_dt, _, execution_time_dt, precision_dt, recall_dt, f1_score_dt, auc_dt = eval_metrics_dt.evaluate_model(decision_tree, X_train, X_test, y_train, y_test)
                accuracy_rf, _, execution_time_rf, precision_rf, recall_rf, f1_score_rf, auc_rf = eval_metrics_rf.evaluate_model(random_forest, X_train, X_test, y_train, y_test)

                end_time = time.time()

                df_comparison = pd.DataFrame({
                    "Model": ["Naive Bayes", "Decision Tree", "Random Forest"],
                    "Akurasi": [
                        accuracy_nb,
                        accuracy_dt,
                        accuracy_rf
                    ],
                    "Presisi": [
                        precision_nb,
                        precision_dt,
                        precision_rf
                    ],
                    "Recall": [
                        recall_nb,
                        recall_dt,
                        recall_rf
                    ],
                    "F1 Score": [
                        f1_score_nb,
                        f1_score_dt,
                        f1_score_rf
                    ],
                    "AUC": [
                        auc_nb,
                        auc_dt,
                        auc_rf
                    ],
                    "Waktu Eksekusi (detik)": [
                        execution_time_nb,
                        execution_time_dt,
                        execution_time_rf
                    ]
                })

                st.write("Hasil Perbandingan Model:")
                st.write(df_comparison)


        else:
            st.write("Mohon pilih file dengan kolom 'Label' untuk menentukan kelas.")
