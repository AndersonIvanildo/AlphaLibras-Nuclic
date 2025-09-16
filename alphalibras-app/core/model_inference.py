import tflite_runtime.interpreter as tflite
import numpy as np

class SignClassifier:
    def __init__(self, model_path, class_names, confidence_threshold=0.70):
        """
        Inicializa o classificador de sinais usando o TFLite Runtime.
        """
        print("Carregando o modelo de inferência TFLite...")
        # Carrega o modelo TFLite e aloca os tensores.
        self.interpreter = tflite.Interpreter(model_path=model_path) # <- MUDANÇA AQUI
        self.interpreter.allocate_tensors()

        # Obtém os detalhes dos tensores de entrada e saída.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.class_names = class_names
        self.threshold = confidence_threshold
        print("Modelo TFLite carregado com sucesso!")

    def predict(self, landmarks):
        """
        Faz a previsão de um sinal a partir dos landmarks da mão.
        """
        if landmarks is None:
            return None, None

        input_vector = np.array([landmarks], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_vector)
        self.interpreter.invoke()
        prediction_probs = self.interpreter.get_tensor(self.output_details[0]['index'])

        confidence = np.max(prediction_probs)
        predicted_index = np.argmax(prediction_probs)

        if confidence >= self.threshold:
            predicted_label = self.class_names[predicted_index]
            return predicted_label, confidence
        else:
            return None, None