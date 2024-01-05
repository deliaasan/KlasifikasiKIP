import streamlit as st
import pandas as pd
from model import NaiveBayes, DecisionTree, RandomForest, EvaluationMetrics

from sklearn.model_selection import train_test_split

from PIL import Image
import io


import time


import numpy as np
def main():
    file_path = 'https://raw.githubusercontent.com/deliaasan/repository/main/DataReal.csv'
    st.markdown("<h1 style='text-align: center'>Data Mining Model Evaluation</h1>", unsafe_allow_html=True)
    col, logo2, col = st.columns([1, 1, 1])
    with logo2:
        image = Image.open('logo.png')  
        width = 150  
        st.image(image, caption='Klasifikasi beasiswa Unsri', width=width)
        st.markdown("<p style='text-align: center'>Perbandingan Algortima Decision Tree (C4.5), Random Forest, dan Naive Bayes dalam Klasifikasi Penerima Beasiswa KIP Unsri</p>", unsafe_allow_html=True)
    

    uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        

        # Check if 'Label' column exists
        if 'Label' in df.columns:
            st.write("Dataset Preview:")
            # st.write(df)
            realFile = pd.read_csv(file_path)    
            realFile = pd.DataFrame(realFile)
            st.dataframe(realFile)
            

            # Define 'Label' column as the target class
            class_col = 'Label'
            y = df[class_col]
            X = df.drop(columns=[class_col])

            st.write("Info Dataset:")
            st.write(f"Jumlah Baris: {len(df)}")
            st.write(f"Jumlah Kolom: {len(df.columns)}")

            # Input split data
            split_ratio = st.sidebar.slider("Training-Testing Split Ratio", 0.1, 0.9, 0.2)
            st.write(f"Pengguna memilih split ratio: {split_ratio}")
 
            # Check if split ratio is chosen
            if split_ratio:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
                st.write(f"Jumlah data training: {len(X_train)}")
                st.write(f"Jumlah data testing: {len(X_test)}")
            
            selected_option = st.radio("Pilih Metode Klasifikasi", ("Naive Bayes", "Decision Tree", "Random Forest"))
            # tes Klasifikadi dengan Naive Bayes
            if selected_option == "Naive Bayes":
                def convert_confusion_matrix(confusion_matrix):
                    tn, fp, fn, tp = confusion_matrix.ravel()
                    new_confusion_matrix = pd.DataFrame(
                    {
                        'Actual Positive': [tp, fp],
                        'Actual Negative': [fn, tn]
                        
                    },
                    index=['Predicted Positive', 'Predicted Negative']
                )
                    return new_confusion_matrix
            
                start_time = time.time()
                naive_bayes = NaiveBayes()
                eval_metrics = EvaluationMetrics()
                accuracy, confusion_matrix, execution_time, precision, recall, f1_score, auc = eval_metrics.evaluate_model(naive_bayes, X_train, X_test, y_train, y_test)
                end_time = time.time()
                new_confusion_matrix = convert_confusion_matrix(confusion_matrix)
                st.write(f"Waktu Eksekusi Naive Bayes: {end_time - start_time:.2f} detik")

                st.write(f"Waktu Eksekusi Naive Bayes: {end_time - start_time:.2f} detik")
                st.write(f"Akurasi Naive Bayes: {accuracy}")
                st.write(f"Presisi Naive Bayes: {precision}")
                st.write(f"Recall Naive Bayes: {recall}")
                st.write(f"F1 Score Naive Bayes: {f1_score}")
                st.write(f"AUC Naive Bayes: {auc}")
                st.write("Confusion Matrix Naive Bayes")
                st.write(new_confusion_matrix)
                


            # Tes Klasifikasi dengan Decision Tree
            elif selected_option == "Decision Tree":
                def convert_confusion_matrix(confusion_matrix):
                    tn, fp, fn, tp = confusion_matrix.ravel()
                    new_confusion_matrix = pd.DataFrame(
                    {
                        'Actual Positive': [tp, fp],
                        'Actual Negative': [fn, tn]
                        
                    },
                    index=['Predicted Positive', 'Predicted Negative']
                )
                    return new_confusion_matrix
                start_time = time.time()
                decision_tree = DecisionTree()
                eval_metrics = EvaluationMetrics()
                accuracy, confusion_matrix, execution_time, precision, recall, f1_score, auc = eval_metrics.evaluate_model(decision_tree, X_train, X_test, y_train, y_test)
                end_time = time.time()
                new_confusion_matrix = convert_confusion_matrix(confusion_matrix)
                st.write(f"Waktu Eksekusi Decision Tree: {end_time - start_time:.2f} detik")
                st.write(f"Akurasi Decision Tree: {accuracy}")
                st.write(f"Presisi Decision Tree: {precision}")
                st.write(f"Recall Decision Tree: {recall}")
                st.write(f"F1 Score Decision Tree: {f1_score}")
                st.write(f"AUC Decision Tree: {auc}")
                st.write("Confusion Matrix Decision Tree")
                st.write(new_confusion_matrix)
                
                
            # Tes klasifikasi dengan Random Forest
            elif selected_option == "Random Forest":
                def convert_confusion_matrix(confusion_matrix):
                    tn, fp, fn, tp = confusion_matrix.ravel()
                    new_confusion_matrix = pd.DataFrame(
                    {
                        'Actual Positive': [tp, fp],
                        'Actual Negative': [fn, tn]
                        
                    },
                    index=['Predicted Positive', 'Predicted Negative']
                )
                    return new_confusion_matrix
                start_time = time.time()
                random_forest = RandomForest()
                eval_metrics = EvaluationMetrics()
                accuracy, dd, execution_time, precision, recall, f1_score, auc = eval_metrics.evaluate_model(random_forest, X_train, X_test, y_train, y_test)
                end_time = time.time()
                new_confusion_matrix = convert_confusion_matrix(confusion_matrix)
                st.write(f"Waktu Eksekusi Random Forest: {end_time - start_time:.2f} detik")
                st.write(f"Akurasi Random Forest: {accuracy}")
                st.write(f"Presisi Random Forest: {precision}")
                st.write(f"Recall Random Forest: {recall}")
                st.write(f"F1 Score Random Forest: {f1_score}")
                st.write(f"AUC Random Forest: {auc}")
                st.write("Confusion Matrix Random Forest")
                st.write(new_confusion_matrix)
            
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
                
                y_pred_nb = naive_bayes.predict(X_test)
                y_pred_dt = decision_tree.predict(X_test)
                y_pred_rf = random_forest.predict(X_test)

                # Add predicted labels to original test set
                X_test_with_predictions = X_test.copy()
                X_test_with_predictions['Naive Bayes Prediction'] = y_pred_nb
                X_test_with_predictions['Decision Tree Prediction'] = y_pred_dt
                X_test_with_predictions['Random Forest Prediction'] = y_pred_rf
                X_test_with_predictions['True Label'] = y_test

                # Display comparison table
                st.write("Hasil Perbandingan Prediksi dengan Label Asli:")
                st.dataframe(X_test_with_predictions)
                

        else:
          st.write("Masukkan File dengan dengan format yang benar")
