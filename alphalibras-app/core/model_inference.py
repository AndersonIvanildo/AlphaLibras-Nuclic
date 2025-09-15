# inference.py
import tensorflow as tf
import numpy as np

class SignClassifier:
    def __init__(self, model_path, class_names, confidence_threshold=0.70):
        """
        Inicializa o classificador de sinais.

        Args:
            model_path (str): Caminho para o modelo treinado (.keras).
            class_names (list): Lista de nomes das classes na ordem correta.
            confidence_threshold (float): Limiar de confiança para mostrar uma previsão.
        """
        print("Carregando o modelo de inferência...")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.threshold = confidence_threshold
        print("Modelo carregado com sucesso!")

    def predict(self, landmarks):
        """
        Faz a previsão de um sinal a partir dos landmarks da mão.

        Args:
            landmarks (list): Uma lista com os 63 landmarks normalizados.

        Returns:
            tuple: Uma tupla contendo (letra_prevista, confiança).
                   Retorna (None, None) se nenhuma mão for detectada ou se a
                   confiança for menor que o limiar.
        """
        if landmarks is None:
            return None, None

        input_vector = np.array([landmarks], dtype=np.float32)

        prediction_probs = self.model.predict(input_vector, verbose=0)

        confidence = np.max(prediction_probs)
        predicted_index = np.argmax(prediction_probs)

        if confidence >= self.threshold:
            predicted_label = self.class_names[predicted_index]
            return predicted_label, confidence
        else:
            return None, None